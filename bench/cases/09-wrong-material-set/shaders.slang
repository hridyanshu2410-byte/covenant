import covenant;
[[vk::binding(0, 0)]] Sampler2D baseColor;     // set 0: per-material
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) { VOut o; float2 p = float2((vid << 1) & 2, vid & 2); o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o; }
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return baseColor.Sample(i.uv); }
