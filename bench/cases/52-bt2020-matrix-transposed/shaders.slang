import covenant;
// ReShade's ImGui overlay on an HDR10 (PQ) swapchain: imgui_hdr.hlsl, compiled
// for Vulkan through ReShade's HLSL->GLSL compatibility header.
struct VIn { float2 pos : POSITION; float2 uv : TEXCOORD; [Color("srgb")] float4 col : COLOR; };
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; [Color("srgb")] float4 col : COLOR; };
struct ImGuiPush { float4x4 proj; float hdrWhiteNits; float3 pad; };
[[vk::push_constant]] ImGuiPush pc;
[[vk::binding(0, 0)]] Sampler2D fontAtlas;

[shader("vertex")]
VOut vsImGui(VIn v) {
    VOut o; o.pos = mul(pc.proj, float4(v.pos, 0, 1)); o.uv = v.uv; o.col = v.col; return o;
}

float3 srgb_to_linear(float3 c) { return select(c <= 0.04045, c / 12.92, pow((c + 0.055) / 1.055, 2.4)); }
float3 linear_to_pq(float3 c) {
    const float m1 = 0.1593017578125, m2 = 78.84375, c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    float3 y = pow(saturate(c), m1);
    return pow((c1 + c2 * y) / (1.0 + c3 * y), m2);
}
// BT.709 -> BT.2020 primaries, written row by row (HLSL constructor order)
static const float3x3 bt709_to_bt2020 = float3x3(
    0.627403914928436, 0.329283028841019, 0.043313067406416,
    0.069097292423248, 0.919540405273438, 0.011362315155566,
    0.016391439363360, 0.088013306260109, 0.895595252513885);

[shader("fragment")]
float4 fsImGui(VOut i) : SV_Target {
    float4 col = i.col * fontAtlas.Sample(i.uv);
    float3 lin = srgb_to_linear(col.rgb);
#ifdef FIXED
    // compat shim: #define mul(a, b) (b) * (a)  -> the row-major M * col of HLSL
    float3 c2020 = mul(bt709_to_bt2020, lin);
#else
    // compat shim: #define mul(a, b) (a) * (b) on a column-major GLSL mat3
    // built from the same coefficients: M^T * col
    float3 c2020 = mul(lin, bt709_to_bt2020);
#endif
    return float4(linear_to_pq(c2020 * pc.hdrWhiteNits / 10000.0), col.a);
}
