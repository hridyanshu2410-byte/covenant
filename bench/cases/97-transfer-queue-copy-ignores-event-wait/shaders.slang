import covenant;
// Two ggml-vulkan style element-wise kernels: the producer graph on backend A
// (scale) and the consumer graph on backend B (add), each one thread per element.
struct PushConstants { uint ne; float param; };
[[vk::push_constant]] ConstantBuffer<PushConstants> p;

[[vk::binding(0, 0)]] StructuredBuffer<float> data_a;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> data_d;

[shader("compute")] [numthreads(256, 1, 1)]
void scale_f32(uint3 id : SV_DispatchThreadID) {
    if (id.x >= p.ne) return;
    data_d[id.x] = data_a[id.x] * p.param;
}

[shader("compute")] [numthreads(256, 1, 1)]
void add_scalar_f32(uint3 id : SV_DispatchThreadID) {
    if (id.x >= p.ne) return;
    data_d[id.x] = data_a[id.x] + p.param;
}
