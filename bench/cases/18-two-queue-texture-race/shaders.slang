import covenant;
// Second context: refreshes a texture that the main context composites.
[[vk::binding(0, 0)]] RWTexture2D<float4> overlayOut;
[shader("compute")] [numthreads(8, 8, 1)]
void updateMain(uint3 id : SV_DispatchThreadID) {
    overlayOut[id.xy] = float4(float2(id.xy) / 256.0, 0.5, 1);
}
// Main context: draws the texture.
[[vk::binding(0, 0)]] Sampler2D overlay;
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return overlay.Sample(i.uv); }
