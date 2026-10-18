import covenant;
struct VOut { float4 pos : SV_Position; };
struct RectPush { float4 rect; float4 color; };   // rect: x0, y0, x1, y1 in [0,1] of the render target
[[vk::push_constant]] RectPush rp;
[shader("vertex")]
VOut vsRect(uint vid : SV_VertexID) {
    const float2 q[6] = { float2(0, 0), float2(1, 0), float2(0, 1), float2(1, 0), float2(1, 1), float2(0, 1) };
    VOut o; float2 p = lerp(rp.rect.xy, rp.rect.zw, q[vid]);
    o.pos = float4(p * 2 - 1, 0, 1); return o;
}
[shader("fragment")]
float4 fsRect(VOut i) : SV_Target { return rp.color; }
