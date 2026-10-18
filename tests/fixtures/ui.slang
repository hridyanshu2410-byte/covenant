import covenant;
struct Proj { column_major float4x4 ProjMtx; };
[[vk::binding(0, 0)]] ConstantBuffer<Proj> proj;
struct VIn { float2 Position : POSITION; float2 UV : TEXCOORD; float4 Color : COLOR; };
struct VOut { float4 pos : SV_Position; float4 color : COLOR; };
[shader("vertex")]
VOut vsMain(VIn v) { VOut o; o.color = v.Color; o.pos = mul(proj.ProjMtx, float4(v.Position, 0, 1)); return o; }
