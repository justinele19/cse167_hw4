[shader("compute")]
[numthreads(64, 1, 1)]
void compute_main(uint tid : SV_DispatchThreadID,
                  RWStructuredBuffer<float3> positions,
                  RWStructuredBuffer<float3> velocities,
                  uniform uint num_particles,
                  // time step size
                  uniform float dt) {
    if (tid >= num_particles) {
        return;
    }
    // TODO: HW 4.1. Your code below.
    float3 x = positions[tid];
    float3 v = velocities[tid];

    // symplectic euler: update v first, then use the new v for x
    v += float3(0.0, -9.8, 0.0) * dt;
    x += v * dt;

    // floor is really at y=-1 but use -0.97 so particles don't sit inside it
    float floor_y = -0.97;
    if (x.y < floor_y) {
        x.y = floor_y;
        v.y = -v.y; // bounce, keep x/z velocity
    }

    positions[tid] = x;
    velocities[tid] = v;
}
