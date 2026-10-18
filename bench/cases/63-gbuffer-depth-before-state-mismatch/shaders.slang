import covenant;
// DeferredShadingD3D12: a G-buffer pass (BasicVS/BasicPS) writes albedo,
// normal and specular render targets plus depth; the light pass
// (ScreenQuadVS/LightPassPS) samples them and the depth buffer.
struct GOut { float4 pos : SV_Position; [Frame("world")] float3 normal : NORMAL; };
struct Camera { [Maps("world","clip")] float4x4 viewProj; };
[[vk::binding(0, 0)]] ConstantBuffer<Camera> camera;
[shader("vertex")]
GOut vsGBuffer(uint vid : SV_VertexID) {
    const float3 p[3] = { float3(-0.7, 0.6, 0.5), float3(0.7, 0.6, 0.5), float3(0, -0.7, 0.5) };
    GOut o; o.pos = mul(camera.viewProj, float4(p[vid], 1)); o.normal = float3(0, 0, -1); return o;
}
struct GBuffer { float4 albedo : SV_Target0; float4 normal : SV_Target1; float4 specular : SV_Target2; };
[shader("fragment")]
GBuffer fsGBuffer(GOut i) {
    GBuffer g; g.albedo = float4(0.7, 0.6, 0.5, 1); g.normal = float4(normalize(i.normal) * 0.5 + 0.5, 0);
    g.specular = float4(0.5, 0.5, 0.5, 32); return g;
}

struct QOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
QOut vsQuad(uint vid : SV_VertexID) {
    QOut o; float2 p = float2((vid << 1) & 2, vid & 2); o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
[[vk::binding(0, 1)]] Sampler2D gAlbedo;
[[vk::binding(1, 1)]] Sampler2D gNormal;
[[vk::binding(2, 1)]] Sampler2D gSpecular;
[[vk::binding(3, 1)]] Sampler2D gDepth;
[shader("fragment")]
float4 fsLight(QOut i) : SV_Target {
    float depth = gDepth.Sample(i.uv).r;
    if (depth >= 1.0) return float4(0, 0, 0, 1);
    float3 n = gNormal.Sample(i.uv).xyz * 2 - 1;
    float ndl = saturate(dot(n, normalize(float3(0.3, 0.5, -1))));
    float4 s = gSpecular.Sample(i.uv);
    return float4(gAlbedo.Sample(i.uv).rgb * ndl + s.rgb * pow(ndl, s.a), 1);
}
