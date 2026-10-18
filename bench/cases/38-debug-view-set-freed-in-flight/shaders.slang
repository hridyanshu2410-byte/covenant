import covenant;
[Color("linear")] Sampler2D texture;              // render target shown by the debug viewer
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 0.5 - 0.9, 0, 1); return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return float4(texture.Sample(i.uv).rgb, 1); }
