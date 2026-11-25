[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         // The CSR matrix data structure
                         StructuredBuffer<uint> row_ptr,
                         StructuredBuffer<uint> neighbors,
                         // Rest lengthes of the neighbor edges
                         StructuredBuffer<float> neighbor_rest,
                         uniform uint num_particles) {
    if (tid >= num_particles) {
        return;
    }
    // TODO: HW 4.4. Your code below.
    float3 xi = positions_in[tid];
    float wi = inv_masses[tid];

    if (wi == 0.0) {
        positions_out[tid] = xi;
        return;
    }

    // neighbors of tid are in neighbors[row_ptr[tid] .. row_ptr[tid+1])
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

    // average over all edges
    uint count = end - start;
    if (count > 0) {
        xi += total / float(count);
    }
    positions_out[tid] = xi;
}
