import covenant;
// Two guest compute dispatches on the same storage image, then a draw
// samples it. Pass 1 writes every texel; pass 2 reads each texel and writes
// a tone-adjusted result in place (a guest post-process chain).
[[vk::binding(0, 0)]] RWTexture2D<float4> target;

[shader("compute")] [numthreads(8, 8, 1)]
void csFill(uint3 id : SV_DispatchThreadID) {
    target[id.xy] = float4(float(id.x & 1), float(id.y & 1), 0, 1);
}

[shader("compute")] [numthreads(8, 8, 1)]
void csSmooth(uint3 id : SV_DispatchThreadID) {
    float4 c = target[id.xy];
    target[id.xy] = float4(c.rgb * 0.5 + 0.25, c.a);
}

[[vk::binding(0, 1)]] Sampler2D tex;
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0.5, 1); return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return tex.Sample(i.uv); }
