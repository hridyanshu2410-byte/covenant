import covenant;
// vrtr's default.vert / default.frag: a hard-coded coloured triangle.
struct VOut { float4 pos : SV_Position; float3 color : COLOR; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    const float2 p[3] = { float2(1, 1), float2(-1, 1), float2(0, -1) };
    const float3 c[3] = { float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1) };
    VOut o; o.pos = float4(p[vid] * 0.8, 0, 1); o.color = c[vid]; return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return float4(i.color, 1); }
