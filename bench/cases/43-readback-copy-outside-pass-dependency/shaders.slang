import covenant;
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsFullscreen(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
[shader("fragment")]
float4 fsScene(VOut i) : SV_Target { return float4(i.uv, 0.5, 1); }
[[vk::binding(0, 0)]] Sampler2D frameColor;
[shader("fragment")]
float4 fsPresent(VOut i) : SV_Target { return frameColor.Sample(i.uv); }
