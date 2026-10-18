import covenant;
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsFullscreen(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
[[vk::binding(0, 0)]] Sampler2D streamedTexture;
[[vk::binding(1, 0)]] Sampler2D copiedTexture;
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target {
    float4 a = streamedTexture.Sample(i.uv);
    float4 b = copiedTexture.Sample(i.uv);
    return float4(lerp(a.rgb, b.rgb, step(0.5, i.uv.x)), 1);
}
