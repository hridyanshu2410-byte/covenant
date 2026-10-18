import covenant;
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsFullscreen(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
// sky: gradient from horizon to zenith, HDR radiance
[shader("fragment")]
float4 fsSky(VOut i) : SV_Target {
    float3 horizon = float3(0.9, 1.0, 1.1), zenith = float3(0.15, 0.3, 0.8);
    return float4(lerp(horizon, zenith, saturate(1 - i.uv.y)) * 2.0, 1);
}
// bloom downsample: bright-pass of the sky colour, 4 taps
[[vk::binding(0, 0)]] Sampler2D bloomSource;
[shader("fragment")]
float4 fsBloomDownsample(VOut i) : SV_Target {
    float3 c = 0;
    [unroll] for (int k = 0; k < 4; k++) {
        float2 o = float2(k & 1, k >> 1) * 2 - 1;
        c += bloomSource.Sample(i.uv + o * float2(1.0 / 640, 1.0 / 480)).rgb;
    }
    c *= 0.25;
    return float4(max(c - 1.0, 0), 1);
}
// tonemap: sky + bloom, Reinhard
[[vk::binding(0, 0)]] Sampler2D sceneColor;
[[vk::binding(1, 0)]] Sampler2D bloomColor;
[shader("fragment")]
float4 fsTonemap(VOut i) : SV_Target {
    float3 hdr = sceneColor.Sample(i.uv).rgb + 0.3 * bloomColor.Sample(i.uv).rgb;
    return float4(hdr / (1 + hdr), 1);
}
