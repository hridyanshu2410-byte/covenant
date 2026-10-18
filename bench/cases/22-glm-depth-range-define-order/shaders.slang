import covenant;
struct Camera {
    [Maps("world","view")] float4x4 view;
    [Maps("view","clip")]  float4x4 proj;
    [Maps("world","clip")] float4x4 lightViewProj;   // spot light shadow projection
};
ConstantBuffer<Camera> camera;
Sampler2D shadowMap;                                  // light-space depth
#ifdef FIXED
static const float SHADOW_BIAS = 0.00005;             // halved with the [0,1] depth range
#else
static const float SHADOW_BIAS = 0.0001;
#endif
struct VOut { float4 pos : SV_Position; float4 lightPos : LIGHTPOS; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    float2 p = float2((vid << 1) & 2, vid & 2);
    [Frame("world")] float3 w = float3(p * 2 - 1, -3);
    VOut o;
    o.pos = mul(camera.proj, mul(camera.view, float4(w, 1)));
    o.lightPos = mul(camera.lightViewProj, float4(w, 1));
    return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target {
    float3 l = i.lightPos.xyz / i.lightPos.w;
    float stored = shadowMap.Sample(l.xy * 0.5 + 0.5).r;
    float lit = (l.z - SHADOW_BIAS > stored) ? 0.2 : 1.0;
    return float4(lit, lit, lit, 1);
}
