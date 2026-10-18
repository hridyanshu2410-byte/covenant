import covenant;
// HartsyInference compute kernels (reduced): two producer ops, and a
// consumer that reads the concatenation of their outputs. Concat itself is
// not a kernel: the backend records it as vkCmdCopyBuffer regions.
struct Params { uint count; float scale; float eps; uint pad; };
[[vk::push_constant]] ConstantBuffer<Params> p;

[[vk::binding(0, 0)]] RWStructuredBuffer<float> src;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dst;

// mul_scalar: dst = src * scale (producer of each concat input)
[shader("compute")] [numthreads(64, 1, 1)]
void mul_scalar(uint3 id : SV_DispatchThreadID) {
    if (id.x < p.count) dst[id.x] = src[id.x] * p.scale;
}

// rms_norm over the whole row (consumer of the concat output), one workgroup
groupshared float partial[64];
[shader("compute")] [numthreads(64, 1, 1)]
void rms_norm(uint3 lid : SV_GroupThreadID) {
    float s = 0;
    for (uint i = lid.x; i < p.count; i += 64) s += src[i] * src[i];
    partial[lid.x] = s;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = 32; off > 0; off >>= 1) {
        if (lid.x < off) partial[lid.x] += partial[lid.x + off];
        GroupMemoryBarrierWithGroupSync();
    }
    float inv = rsqrt(partial[0] / float(p.count) + p.eps);
    for (uint i = lid.x; i < p.count; i += 64) dst[i] = src[i] * inv;
}
