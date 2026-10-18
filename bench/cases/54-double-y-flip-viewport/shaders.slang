import covenant;
// waterui's shaders are written in WGSL (clip space: +y up) and precompiled to
// SPIR-V by shaderloom with naga's spv writer. With naga's default
// WriterFlags::ADJUST_COORDINATE_SPACE the writer appends `pos.y = -pos.y`
// to every write of BuiltIn::Position; shaderloom 0.1.2 clears that flag.
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; float4 color : COLOR; };
struct Rect { float4 bounds; float4 color; };   // bounds in WGSL clip space: x0, y0 (bottom), x1, y1 (top)
[[vk::push_constant]] Rect rect;

[shader("vertex")]
VOut vsRect(uint vid : SV_VertexID) {
    const float2 q[6] = { float2(0, 0), float2(1, 0), float2(0, 1), float2(1, 0), float2(1, 1), float2(0, 1) };
    VOut o;
    float2 p = lerp(rect.bounds.xy, rect.bounds.zw, q[vid]);
    o.pos = float4(p, 0, 1);
    o.uv = float2(q[vid].x, 1 - q[vid].y);     // texture v grows downwards
    o.color = rect.color;
#ifndef FIXED
    o.pos.y = -o.pos.y;                       // emitted by ADJUST_COORDINATE_SPACE
#endif
    return o;
}
[shader("fragment")]
float4 fsRect(VOut i) : SV_Target {
    float edge = step(0.05, min(min(i.uv.x, 1 - i.uv.x), min(i.uv.y, 1 - i.uv.y)));
    return float4(i.color.rgb * lerp(0.6, 1.0, edge), i.color.a);
}
