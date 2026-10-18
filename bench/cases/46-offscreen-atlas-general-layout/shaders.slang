import covenant;
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsFullscreen(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
// item icon quad inside its atlas cell (cell rect in push constants, atlas uv)
struct ItemPush { float4 cell; };
[[vk::push_constant]] ItemPush item;
[shader("vertex")]
VOut vsItem(uint vid : SV_VertexID) {
    const float2 q[6] = { float2(0, 0), float2(1, 0), float2(0, 1), float2(1, 0), float2(1, 1), float2(0, 1) };
    VOut o; float2 p = item.cell.xy + q[vid] * item.cell.zw;
    o.uv = q[vid]; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
[shader("fragment")]
float4 fsItem(VOut i) : SV_Target {
    float d = length(i.uv - 0.5);
    return float4(0.8, 0.6, 0.2, 1) * step(d, 0.45);
}
[[vk::binding(0, 0)]] Sampler2D itemAtlas;
[shader("fragment")]
float4 fsGui(VOut i) : SV_Target {
    float4 icon = itemAtlas.Sample(i.uv);
    return float4(lerp(float3(0.1, 0.1, 0.12), icon.rgb, icon.a), 1);
}
