[shader("compute")]
[numthreads(64, 1, 1)]
void predict(uint tid : SV_DispatchThreadID,
             // Updated position
             RWStructuredBuffer<float3> positions,
             // Store the position before update
             RWStructuredBuffer<float3> old_positions,
             RWStructuredBuffer<float3> velocities,
             StructuredBuffer<float> inv_masses,
             uniform uint num_particles,
             // time step size
             uniform float dt) {
    if (tid >= num_particles) {
        return;
    }
    // TODO: HW 4.2. Your code below.
    float3 x = positions[tid];
    old_positions[tid] = x; // save for velocity update later

    // pinned node, don't touch it
    if (inv_masses[tid] == 0.0) {
        return;
    }

    float3 v = velocities[tid] + float3(0.0, -9.8, 0.0) * dt;
    velocities[tid] = v;
    positions[tid] = x + v * dt;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         StructuredBuffer<float> rest_lengths,
                         uniform uint num_particles) {
    if (tid >= num_particles) {
        return;
    }
    // TODO: HW 4.2. Your code below.
    float3 xi = positions_in[tid];
    float wi = inv_masses[tid];

    // pinned, just copy over
    if (wi == 0.0) {
        positions_out[tid] = xi;
        return;
    }

    float3 total = float3(0.0);
    int count = 0;

    // rope: neighbors are tid-1 and tid+1
    // rest_lengths[k] is the edge between k and k+1
    for (int k = 0; k < 2; k++) {
        int j = (k == 0) ? int(tid) - 1 : int(tid) + 1;
        if (j < 0 || j >= int(num_particles)) {
            continue;
        }
        float rest = rest_lengths[k == 0 ? j : tid];

        float3 d = positions_in[j] - xi;
        float len = length(d);
        float wsum = wi + inv_masses[j];
        count++;
        if (len < 1e-8 || wsum == 0.0) {
            continue; // avoid div by zero
        }
        // dx_i = wi/(wi+wj) * (|d| - l) * d/|d|
        total += (wi / wsum) * (len - rest) * (d / len);
    }

    // jacobi: average the corrections
    if (count > 0) {
        xi += total / float(count);
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
                     // time step size
                     uniform float dt) {
    if (tid >= num_particles) {
        return;
    }
    // TODO: HW 4.2. Your code below.
    if (inv_masses[tid] == 0.0) {
        return;
    }

    // simple floor hack, clip to y=-0.97
    float3 x = positions[tid];
    if (x.y < -0.97) {
        x.y = -0.97;
        positions[tid] = x;
    }

    // velocity from how far we moved this step
    velocities[tid] = (x - old_positions[tid]) / dt;
}
