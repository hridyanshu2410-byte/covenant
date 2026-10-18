import covenant;
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsFullscreen(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}

// ---- scene pass: visual / background shaders draw into the RGBA16F intermediate.
// They write their colours unchanged, as the SDR path (plain UNORM swapchain)
// expects.
struct Material { [Color("srgb")] float4 diffuse; };      // Inventor material colour, sRGB code values
[[vk::binding(0, 0)]] ConstantBuffer<Material> material;
[shader("vertex")]
VOut vsVisual(uint vid : SV_VertexID) {
    const float2 p[3] = { float2(-0.6, 0.6), float2(0.6, 0.6), float2(0, -0.6) };
    VOut o; o.uv = p[vid]; o.pos = float4(p[vid], 0, 1); return o;
}
[shader("fragment")]
float4 fsVisual(VOut i) : SV_Target { return material.diffuse; }

// ---- output pass: intermediate -> swapchain
struct OutputParams { uint hdr; float exposure; float paperWhiteNits; float pad; };
[[vk::push_constant]] OutputParams outp;
[[vk::binding(1, 0)]] Sampler2D sceneIntermediate;

float3 srgb_to_linear(float3 c) {                     // IEC 61966-2-1
    return select(c <= 0.04045, c / 12.92, pow((c + 0.055) / 1.055, 2.4));
}
static const float3x3 kL709ToL2020 = float3x3(
    0.6274040, 0.3292820, 0.0433136,
    0.0690970, 0.9195400, 0.0113612,
    0.0163916, 0.0880132, 0.8955950);
float3 tonemap(float3 L) { return L / (1.0 + L); }
float3 pq_inverse_eotf(float3 nits) {                 // SMPTE ST 2084
    const float m1 = 0.1593017578125, m2 = 78.84375, c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    float3 y = pow(saturate(nits / 10000.0), m1);
    return pow((c1 + c2 * y) / (1.0 + c3 * y), m2);
}

[shader("fragment")]
float4 fsOutput(VOut i) : SV_Target {
    float3 c = sceneIntermediate.Sample(i.uv).rgb;
    if (outp.hdr == 0)
        return float4(clamp(c, 0.0, 1.0), 1);
#ifdef FIXED
    float3 lin = mul(kL709ToL2020, srgb_to_linear(clamp(c, 0.0, 1.0)));
    float3 L = lin * outp.exposure;
#else
    float3 L = max(c, 0.0) * outp.exposure;
#endif
    float3 nits = tonemap(L) * outp.paperWhiteNits * 2.0;
    return float4(pq_inverse_eotf(nits), 1);
}
