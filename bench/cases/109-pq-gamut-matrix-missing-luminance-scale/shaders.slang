import covenant;
// Tint's external-texture lowering (textureSampleBaseClampToEdge on a
// texture_external), as Dawn emits it for a single-plane external texture,
// used by a full-screen video quad.
struct GammaTransferParams { float G; float A; float B; float C; float D; float E; float F; float padding; };
struct ExternalTextureParams {
    uint numPlanes;
    uint doYuvToRgbConversionOnly;
    column_major float4x3 yuvToRgbConversionMatrix;   // WGSL mat3x4f: 3 columns of vec4
    GammaTransferParams gammaDecodeParams;
    GammaTransferParams gammaEncodeParams;
    column_major float3x3 gamutConversionMatrix;   // source linear RGB -> destination linear RGB
};
[[vk::binding(0, 0)]] Texture2D<float4> plane0;          // HDR10 frame: BT.2020 primaries, PQ-encoded
[[vk::binding(1, 0)]] SamplerState smp;
[[vk::binding(2, 0)]] ConstantBuffer<ExternalTextureParams> ext_tex_params;

float3 gammaCorrection(float3 v, GammaTransferParams params) {
    if (params.G == -1.0) {         // HLG inverse OETF
        float3 lo = v * v / params.E;
        float3 hi = (params.B + exp((v - params.C) / params.A)) / params.F;
        return select(v <= params.D, lo, hi);
    }
    if (params.G == -2.0) {         // PQ EOTF, 1.0 = 10000 nits
        float3 p = pow(clamp(v, 0.0, 1.0), 1.0 / params.B);
        p = max(p - params.C, 0.0) / (params.D - params.E * p);
        return pow(p, 1.0 / params.A);
    }
    bool3 cond = abs(v) < params.D;
    float3 t = sign(v) * ((params.C * abs(v)) + params.F);
    float3 f = sign(v) * (pow((params.A * abs(v)) + params.B, params.G) + params.E);
    return select(cond, t, f);
}

float4 textureSampleExternal(float2 coord) {
    float4 color = plane0.SampleLevel(smp, coord, 0.0);
    float3 rgb = mul(float4(color.rgb, 1.0), ext_tex_params.yuvToRgbConversionMatrix);
    if (ext_tex_params.doYuvToRgbConversionOnly == 0) {
        rgb = gammaCorrection(rgb, ext_tex_params.gammaDecodeParams);
        rgb = mul(ext_tex_params.gamutConversionMatrix, rgb);
        rgb = gammaCorrection(rgb, ext_tex_params.gammaEncodeParams);
    }
    return float4(rgb, color.a);
}

struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
[shader("vertex")]
VOut vertMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
[shader("fragment")]
float4 fragMain(VOut i) : SV_Target {
    return textureSampleExternal(i.uv);
}
