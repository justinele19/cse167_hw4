struct BVHNode {
    float3 bbox_min;
    float3 bbox_max;

    // which BVH node we should skip
    // to if this node is totally not
    // intersecting the ray
    // if next_index < 0, then we don't
    // have the next node to skip to
    int next_index;

    uint triangle_begin;
    uint triangle_end;

    bool is_leaf;
};

float point_aabb_distance(float3 x, float3 bmin, float3 bmax) {
    return length(x - clamp(x, bmin, bmax));
}

struct ClosestPoint {
    float3 position;
    float3 bary;
};

// From https://iquilezles.org/articles/distfunctions/
ClosestPoint closest_point_triangle(
        float3 p, float3 a, float3 b, float3 c) {
    float3 ba = b - a, pa = p - a;
    float3 cb = c - b, pb = p - b;
    float3 ac = a - c, pc = p - c;
    float3 nor = cross(ba, ac);

    float t0 = clamp(dot(ba, pa) / dot(ba, ba), 0.0, 1.0);
    float t1 = clamp(dot(cb, pb) / dot(cb, cb), 0.0, 1.0);
    float t2 = clamp(dot(ac, pc) / dot(ac, ac), 0.0, 1.0);

    float3 q0 = a + ba * t0;
    float3 q1 = b + cb * t1;
    float3 q2 = c + ac * t2;

    if (sign(dot(cross(ba, nor), pa)) +
        sign(dot(cross(cb, nor), pb)) +
        sign(dot(cross(ac, nor), pc)) < 2.0) {

        float d0 = dot(p - q0, p - q0);
        float d1 = dot(p - q1, p - q1);
        float d2 = dot(p - q2, p - q2);

        if (d0 < d1 && d0 < d2)
            return {q0, float3(1.0 - t0, t0, 0.0)};
        if (d1 < d2)
            return {q1, float3(0.0, 1.0 - t1, t1)};
        return {q2, float3(t2, 0.0, 1.0 - t2)};
    }

    float inv_nor2 = 1.0 / dot(nor, nor);
    float3 q = p - nor * (dot(nor, pa) * inv_nor2);

    float wb = dot(cross(pa, ac), nor) * inv_nor2;
    float wc = dot(cross(pa, ba), nor) * inv_nor2;
    return {q, float3(1.0 - wb - wc, wb, wc)};
}


[shader("compute")]
[numthreads(64, 1, 1)]
void detect_collision(uint tid : SV_DispatchThreadID,
                      StructuredBuffer<float3> positions,
                      StructuredBuffer<float> inv_masses,
                      StructuredBuffer<float4> planes,
                      StructuredBuffer<float4> spheres,
                      StructuredBuffer<float3> mesh_vertices,
                      StructuredBuffer<float3> mesh_vertex_normals,
                      StructuredBuffer<uint3> mesh_triangles,
                      StructuredBuffer<BVHNode> bvh_nodes,
                      RWStructuredBuffer<float4> contact_planes,
                      uniform uint num_particles,
                      uniform uint num_planes,
                      uniform uint num_spheres,
                      uniform uint num_bvh_nodes,
                      uniform float collision_distance,
                      uniform float contact_margin) {
    if (tid >= num_particles) {
        return;
    }
    // TODO: HW 4.6. Your code below.
    float3 x = positions[tid];

    // closest collider so far (smallest signed distance)
    float best_dist = 1e30;
    float4 best_plane = float4(0.0);

    // --- planes and spheres (same as 4.5) ---
    for (uint i = 0; i < num_planes; i++) {
        float4 p = planes[i];
        float dist = dot(p.xyz, x) - p.w;
        if (dist < best_dist) {
            best_dist = dist;
            best_plane = p;
        }
    }

    for (uint i = 0; i < num_spheres; i++) {
        float4 s = spheres[i];
        float3 v = x - s.xyz;
        float len = length(v);
        float3 n = (len > 1e-8) ? v / len : float3(0.0, 1.0, 0.0);
        float dist = len - s.w;
        if (dist < best_dist) {
            best_dist = dist;
            best_plane = float4(n, dot(n, s.xyz) + s.w);
        }
    }

    // --- mesh: find the closest triangle using the BVH ---
    // only care about triangles closer than the contact range
    float closest_dist = contact_margin + collision_distance;
    bool found = false;
    uint3 hit_tri = uint3(0);
    float3 hit_bary = float3(0.0);

    // stackless traversal: first child is node + 1,
    // next_index skips the whole subtree
    uint node = 0;
    while (node < num_bvh_nodes) {
        BVHNode N = bvh_nodes[node];

        // box is farther than our best triangle, skip it
        if (point_aabb_distance(x, N.bbox_min, N.bbox_max) > closest_dist) {
            node = uint(N.next_index);
            continue;
        }

        if (N.is_leaf) {
            for (uint t = N.triangle_begin; t < N.triangle_end; t++) {
                uint3 tri = mesh_triangles[t];
                ClosestPoint cp = closest_point_triangle(
                    x, mesh_vertices[tri.x], mesh_vertices[tri.y], mesh_vertices[tri.z]);
                float d = length(x - cp.position);
                if (d < closest_dist) {
                    closest_dist = d;
                    hit_tri = tri;
                    hit_bary = cp.bary;
                    found = true;
                }
            }
            node = uint(N.next_index);
        } else {
            node = node + 1;
        }
    }

    // treat the closest triangle as an infinite plane
    if (found) {
        float3 p0 = mesh_vertices[hit_tri.x];
        float3 p1 = mesh_vertices[hit_tri.y];
        float3 p2 = mesh_vertices[hit_tri.z];
        float3 n0 = mesh_vertex_normals[hit_tri.x];
        float3 n1 = mesh_vertex_normals[hit_tri.y];
        float3 n2 = mesh_vertex_normals[hit_tri.z];

        // face normal
        float3 ng = normalize(cross(p1 - p0, p2 - p0));
        // interpolated vertex normal at the closest point
        float3 ns = normalize(hit_bary.x * n0 + hit_bary.y * n1 + hit_bary.z * n2);
        // flip face normal so it points outside
        if (dot(ng, ns) < 0.0) {
            ng = -ng;
        }

        float d = dot(ng, p0);
        float dist = dot(ng, x) - d;
        if (dist < best_dist) {
            best_dist = dist;
            best_plane = float4(ng, d);
        }
    }

    // --- store the contact (same as 4.5) ---
    if (best_dist < contact_margin + collision_distance) {
        best_plane.w += collision_distance;
        contact_planes[tid] = best_plane;
    } else {
        contact_planes[tid] = float4(0.0);
    }
}
