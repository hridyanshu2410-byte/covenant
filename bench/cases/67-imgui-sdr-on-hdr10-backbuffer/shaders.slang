import covenant;
// imgui_impl_dx12 shaders (vertex: ProjectionMatrix * pos; pixel: col * tex),
// with the pixel shader from imgui_impl_dxgi_hdr.h in the fixed variant
// (IMGUI_COLOR_SPACE 2: HDR10 / ST.2084 back buffer).
struct VIn { float2 pos : POSITION; float2 uv : TEXCOORD; [Color("srgb")] float4 col : COLOR; };
struct VOut { float4 pos : SV_Position; float4 col : COLOR0; float2 uv : TEXCOORD0; };
struct VertexBuffer { column_major float4x4 ProjectionMatrix; };
[[vk::push_constant]] VertexBuffer vb;
[[vk::binding(0, 0)]] Sampler2D texture0;
[shader("vertex")]
VOut vsMain(VIn v) {
    VOut o; o.pos = mul(vb.ProjectionMatrix, float4(v.pos, 0, 1)); o.col = v.col; o.uv = v.uv; return o;
}

#ifdef FIXED
static const float GRAPHICS_WHITE_NITS = 203.0;
float3 SrgbToLinear(float3 color) {
    float3 low = color / 12.92;
    float3 high = pow((color + 0.055) / 1.055, 2.4);
    return lerp(low, high, step(0.04045, color));
}
float3 Rec709ToRec2020(float3 color) {
    const float3x3 transform = {
        0.6274040, 0.3292820, 0.0433136,
        0.0690970, 0.9195400, 0.0113612,
        0.0163916, 0.0880132, 0.8955950 };
    return mul(transform, color);
}
float3 LinearToPq(float3 color) {
    const float m1 = 2610.0 / 16384.0;
    const float m2 = 2523.0 / 32.0;
    const float c1 = 3424.0 / 4096.0;
    const float c2 = 2413.0 / 128.0;
    const float c3 = 2392.0 / 128.0;
    float3 p = pow(max(color, 0.0), m1);
    return pow((c1 + c2 * p) / (1.0 + c3 * p), m2);
}
#endif

[shader("fragment")]
float4 fsMain(VOut input) : SV_Target {
    float4 out_col = input.col * texture0.Sample(input.uv);
#ifdef FIXED
    float3 linear_rec709 = SrgbToLinear(saturate(out_col.rgb));
    float3 linear_rec2020 = Rec709ToRec2020(linear_rec709);
    out_col.rgb = LinearToPq(linear_rec2020 * (GRAPHICS_WHITE_NITS / 10000.0));
#endif
    return out_col;
}
