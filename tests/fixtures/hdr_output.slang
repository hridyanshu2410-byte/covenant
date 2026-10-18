// Fixture for the swapchain color space rule and the PQ (ST 2084) heuristic.
import covenant;
[[vk::binding(0, 0)]] [Color("linear")] Sampler2D scene;     // linear light, BT.709
[[vk::binding(1, 0)]] Sampler2D hdrVideo;                       // encoding left to inference
float3 pq_encode(float3 nits) {                                 // inverse EOTF
    const float m1 = 0.1593017578125, m2 = 78.84375, c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    float3 y = pow(saturate(nits / 10000.0), m1);
    return pow((c1 + c2 * y) / (1.0 + c3 * y), m2);
}
float3 pq_decode(float3 e) {                                    // EOTF
    const float m1 = 0.1593017578125, m2 = 78.84375, c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    float3 p = pow(saturate(e), 1.0 / m2);
    return pow(max(p - c1, 0.0) / (c2 - c3 * p), 1.0 / m1) * 10000.0;
}
[shader("fragment")]
float4 fsPQ(float2 uv : TEXCOORD) : SV_Target {          // correct for an HDR10 swapchain
    return float4(pq_encode(scene.Sample(uv).rgb * 203.0), 1);
}
[shader("fragment")]
float4 fsGamma(float2 uv : TEXCOORD) : SV_Target {       // correct for an sRGB UNORM swapchain
    return float4(pow(saturate(scene.Sample(uv).rgb), 1.0 / 2.2), 1);
}
[shader("fragment")]
float4 fsLinear(float2 uv : TEXCOORD) : SV_Target {      // correct for scRGB (extended linear)
    return float4(scene.Sample(uv).rgb, 1);
}
[shader("fragment")]
float4 fsVideo(float2 uv : TEXCOORD) : SV_Target {       // decodes a PQ video frame to linear
    return float4(pq_decode(hdrVideo.Sample(uv).rgb) / 203.0, 1);
}
