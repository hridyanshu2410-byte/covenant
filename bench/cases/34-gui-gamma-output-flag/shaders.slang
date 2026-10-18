import covenant;
// SkiaGameRendering-Premultiplied: composites the GUI render texture.
[Color("srgb")] Sampler2D _MainTex;               // GUI render texture, 8-bit sRGB-encoded colours
struct PremultipliedParams { uint _GammaOutput; float _Opacity; float2 pad; };
ConstantBuffer<PremultipliedParams> material;

float3 srgbToLinear(float3 c) {
    return select(c <= 0.04045, c / 12.92, pow((c + 0.055) / 1.055, 2.4));
}

struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target {
    float4 c = _MainTex.Sample(i.uv);
    // _GammaOutput: the GUI already wrote gamma values, so skip the decode
    if (material._GammaOutput == 0)
        c.rgb = srgbToLinear(c.rgb);
    return c * material._Opacity;
}
