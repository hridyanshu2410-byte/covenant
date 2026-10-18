import covenant;
// Draws a quad textured with the destination texture of the blit.
[[vk::binding(0, 0)]] Sampler2D atlas;
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0.5, 1); return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return atlas.Sample(i.uv); }
