[shader("compute")]
[numthreads(64, 1, 1)]
void detect_collision(uint tid : SV_DispatchThreadID,
                      StructuredBuffer<float3> positions,
                      StructuredBuffer<float> inv_masses,
                      // Plane stores (a, b, c, d) where ax+by+cz=d
                      // a, b, c are normalized to a unit 3D vector
                      StructuredBuffer<float4> planes,
                      // Sphere stores center (xyz) + radius (w)
                      StructuredBuffer<float4> spheres,
                      // Contact plane: dot(n, x) = d, with unit normal n
                      // Set to (0, 0, 0, 0) if there is no contact
                      RWStructuredBuffer<float4> contact_planes,
                      uniform uint num_particles,
                      uniform uint num_planes,
                      uniform uint num_spheres,
                      // How much should we shift the generated 
                      // contact plane in the normal direction
                      // (similar to how we shifted the floor by 0.03
                      //  earlier)
                      uniform float collision_distance,
                      // When the signed distance function
                      // evaluates to a value below 
                      // contact_margin + collision_distance
                      // (even when it's positive), we count the
                      // particle to be inside the collider and
                      // a collision event has happened.
                      uniform float contact_margin) {
    if (tid >= num_particles) {
        return;
    }
    // TODO: HW 4.5. Your code below.
    float3 x = positions[tid];

    // closest collider so far (smallest signed distance)
    float best_dist = 1e30;
    float4 best_plane = float4(0.0);

    // planes: signed distance is dot(n, x) - d
    for (uint i = 0; i < num_planes; i++) {
        float4 p = planes[i];
        float dist = dot(p.xyz, x) - p.w;
        if (dist < best_dist) {
            best_dist = dist;
            best_plane = p;
        }
    }

    // spheres: signed distance is |x - c| - r
    for (uint i = 0; i < num_spheres; i++) {
        float4 s = spheres[i];
        float3 v = x - s.xyz;
        float len = length(v);
        // avoid divide by zero if we're exactly at the center
        float3 n = (len > 1e-8) ? v / len : float3(0.0, 1.0, 0.0);
        float dist = len - s.w;
        if (dist < best_dist) {
            best_dist = dist;
            // tangent plane at the closest surface point c + r*n
            best_plane = float4(n, dot(n, s.xyz) + s.w);
        }
    }

    if (best_dist < contact_margin + collision_distance) {
        // shift the plane out a bit so particles don't sit right on the surface
        best_plane.w += collision_distance;
        contact_planes[tid] = best_plane;
    } else {
        contact_planes[tid] = float4(0.0); // no contact
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         StructuredBuffer<float3> old_positions,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         StructuredBuffer<uint> row_ptr,
                         StructuredBuffer<uint> neighbors,
                         StructuredBuffer<float> neighbor_rest,
                         // Contact plane: dot(n, x) = d, with unit normal n
                         StructuredBuffer<float4> contact_planes,
                         uniform uint num_particles,
                         uniform float static_friction,
                         uniform float kinetic_friction) {
    if (tid >= num_particles) {
        return;
    }
    // TODO: HW 4.5. Your code below.
    float3 xi = positions_in[tid];
    float wi = inv_masses[tid];

    if (wi == 0.0) {
        positions_out[tid] = xi;
        return;
    }

    // --- distance constraints (same as 4.4) ---
    uint start = row_ptr[tid];
    uint end = row_ptr[tid + 1];

    float3 total = float3(0.0);
    for (uint e = start; e < end; e++) {
        uint j = neighbors[e];
        float rest = neighbor_rest[e];

        float3 d = positions_in[j] - xi;
        float len = length(d);
        float wsum = wi + inv_masses[j];
        if (len < 1e-8 || wsum == 0.0) {
            continue;
        }
        total += (wi / wsum) * (len - rest) * (d / len);
    }

    uint count = end - start;
    if (count > 0) {
        xi += total / float(count);
    }

    // --- contact + friction ---
    float4 plane = contact_planes[tid];
    float3 n = plane.xyz;
    if (dot(n, n) > 0.0) {
        // negative means we're below the contact plane
        float pen = dot(n, xi) - plane.w;
        if (pen < 0.0) {
            float depth = -pen;

            // normal: push back onto the plane
            xi += n * depth;

            // tangential motion since the start of the step
            float3 dx = xi - old_positions[tid];
            float3 t = dx - dot(dx, n) * n;
            float tl = length(t);

            if (tl <= static_friction * depth) {
                // static friction: cancel the sliding completely
                xi -= t;
            } else {
                // kinetic friction: slow the sliding down
                xi -= (t / tl) * (kinetic_friction * depth);
            }
        }
    }

    positions_out[tid] = xi;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void update_velocity(uint tid : SV_DispatchThreadID,
                     RWStructuredBuffer<float3> positions,
                     StructuredBuffer<float3> old_positions,
                     RWStructuredBuffer<float3> velocities,
                     StructuredBuffer<float> inv_masses,
                     uniform uint num_particles,
                     uniform float dt) {
    if (tid >= num_particles) {
        return;
    }
    // TODO: HW 4.5. Your code below.
    if (inv_masses[tid] == 0.0) {
        return;
    }

    // no more floor hack, collisions handle that now
    velocities[tid] = (positions[tid] - old_positions[tid]) / dt;
}
