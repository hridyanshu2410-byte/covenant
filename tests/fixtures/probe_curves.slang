// Fixture for probe fingerprinting (covenant/probe.py): transfer curves
// written in different shapes, and curves that must stay opaque.
import covenant;
[[vk::binding(0, 0)]] Sampler2D tex;                    // encoding left to inference
struct Params { float paperWhite; float3 pad; };
[[vk::push_constant]] Params pc;
float3 decode_select(float3 c) { return select(c <= 0.04045, c / 12.92, pow((c + 0.055) / 1.055, 2.4)); }
float3 decode_lerp(float3 c) {
    float3 low = c / 12.92;
    float3 high = pow((c + 0.055) / 1.055, 2.4);
    return lerp(low, high, step(0.04045, c));
}
float3 pq(float3 L) {
    const float m1 = 0.1593017578125, m2 = 78.84375, c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    float3 y = pow(saturate(L), m1);
    return pow((c1 + c2 * y) / (1.0 + c3 * y), m2);
}
[shader("fragment")]
float4 fsSelect(float2 uv : TEXCOORD) : SV_Target { return float4(decode_select(tex.Sample(uv).rgb), 1); }
[shader("fragment")]
float4 fsLerp(float2 uv : TEXCOORD) : SV_Target { return float4(decode_lerp(tex.Sample(uv).rgb), 1); }
[shader("fragment")]
float4 fsDecodeThenPQ(float2 uv : TEXCOORD) : SV_Target {
    return float4(pq(decode_lerp(tex.Sample(uv).rgb) * (pc.paperWhite / 10000.0)), 1);
}
[shader("fragment")]
float4 fsGamma22(float2 uv : TEXCOORD) : SV_Target { return float4(pow(tex.Sample(uv).rgb, 1.0 / 2.2), 1); }
[shader("fragment")]
float4 fsOdd(float2 uv : TEXCOORD) : SV_Target { return float4(pow(tex.Sample(uv).rgb, 3.7), 1); }
[shader("fragment")]
float4 fsSmooth(float2 uv : TEXCOORD) : SV_Target { return float4(smoothstep(0.0, 1.0, tex.Sample(uv).rgb), 1); }
