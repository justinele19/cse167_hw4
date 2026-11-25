[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         uniform int2 grid_size,
                         uniform float2 spacing) {
    if (tid >= grid_size.x * grid_size.y) {
        return;
    }
    // TODO: HW 4.3. Your code below.
    float3 xi = positions_in[tid];
    float wi = inv_masses[tid];

    if (wi == 0.0) {
        positions_out[tid] = xi;
        return;
    }

    // row-major grid index
    int x = int(tid) % grid_size.x;
    int y = int(tid) / grid_size.x;

    // neighbor offsets: structural (4), shear (4), bending (4)
    const int2 offsets[12] = {
        int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1),
        int2(1, 1), int2(-1, -1), int2(1, -1), int2(-1, 1),
        int2(2, 0), int2(-2, 0), int2(0, 2), int2(0, -2)
    };

    float3 total = float3(0.0);
    int count = 0;

    for (int k = 0; k < 12; k++) {
        int nx = x + offsets[k].x;
        int ny = y + offsets[k].y;
        // skip if off the edge of the cloth
        if (nx < 0 || nx >= grid_size.x || ny < 0 || ny >= grid_size.y) {
            continue;
        }
        int j = ny * grid_size.x + nx;

        // rest length from the grid spacing
        float rest = length(float2(offsets[k]) * spacing);

        float3 d = positions_in[j] - xi;
        float len = length(d);
        float wsum = wi + inv_masses[j];
        count++;
        if (len < 1e-8 || wsum == 0.0) {
            continue;
        }
        total += (wi / wsum) * (len - rest) * (d / len);
    }

    if (count > 0) {
        xi += total / float(count);
    }
    positions_out[tid] = xi;
}
