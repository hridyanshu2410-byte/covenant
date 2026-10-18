import covenant;
struct VIn { float2 pos : POSITION; float3 color : COLOR; };
struct VOut { float4 pos : SV_Position; float3 color : COLOR; };
[shader("vertex")]
VOut vsMain(VIn v) { VOut o; o.pos = float4(v.pos, 0, 1); o.color = v.color; return o; }
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return float4(i.color, 1); }
