// Fixture for the clip depth range dimension: a position rebuilt from a depth
// texture and an inverse projection with the [0, 1] range.
import covenant;
struct View { [Maps("clip","view")] float4x4 invProj; };
[[vk::binding(0, 0)]] ConstantBuffer<View> view;
[[vk::binding(1, 0)]] Sampler2D depthTex;
[shader("fragment")]
float4 fsGood(float2 uv : TEXCOORD) : SV_Target {
    float z = depthTex.Sample(uv).r;
    float4 p = mul(view.invProj, float4(uv * 2 - 1, z, 1));
    return float4(p.xyz / p.w, 1);
}
[shader("fragment")]
float4 fsBad(float2 uv : TEXCOORD) : SV_Target {
    float z = depthTex.Sample(uv).r * 2 - 1;
    float4 p = mul(view.invProj, float4(uv * 2 - 1, z, 1));
    return float4(p.xyz / p.w, 1);
}
