import covenant;
// PlayCanvas skybox shader: the cubemap is decoded with the function chosen
// from the texture's encoding (decode chunk): 'linear' (the sampler already
// returns linear values, e.g. an sRGB format), 'srgb' (gamma decode) or
// 'rgbp'. The engine bakes the choice in as a define; here it is a uniform.
struct Skybox { [Maps("view","world")] float4x4 invViewProj; uint encoding; float exposure; float2 pad; };
[[vk::binding(0, 0)]] ConstantBuffer<Skybox> sky;
[[vk::binding(1, 0)]] SamplerCube texture_cubeMap;

static const uint ENCODING_LINEAR = 0, ENCODING_SRGB = 1, ENCODING_RGBP = 2;
float3 decodeLinear(float4 raw) { return raw.rgb; }
float3 decodeGamma(float4 raw) { return pow(raw.rgb, float3(2.2)); }
float3 decodeRGBP(float4 raw) {
    float3 color = raw.rgb * (-raw.a * 7.0 + 8.0);
    return color * color;
}

struct VOut { float4 pos : SV_Position; [Frame("world")] float3 vViewDir : VIEWDIR; };
[shader("vertex")]
VOut vsSkybox(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2) * 2 - 1;
    o.pos = float4(p, 1, 1); o.vViewDir = mul(sky.invViewProj, float4(p, 1, 1)).xyz; return o;
}
[shader("fragment")]
float4 fsSkybox(VOut i) : SV_Target {
    float4 raw = texture_cubeMap.Sample(normalize(i.vViewDir));
    float3 linear = sky.encoding == ENCODING_RGBP ? decodeRGBP(raw)
                  : sky.encoding == ENCODING_SRGB ? decodeGamma(raw) : decodeLinear(raw);
    return float4(linear * sky.exposure, 1);                  // sRGB back buffer encodes on store
}
