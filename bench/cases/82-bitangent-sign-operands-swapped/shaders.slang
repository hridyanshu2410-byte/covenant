import covenant;
// ByroRedux mesh shading (reduced): the vertex carries a tangent whose w is
// the bitangent sign; the fragment shader rebuilds B = tangent.w * cross(N, T)
// and perturbs the normal with the tangent-space normal map.
struct Camera { [Maps("world","clip")] float4x4 viewProj; };
[[vk::binding(0, 0)]] ConstantBuffer<Camera> cam;
[[vk::binding(1, 0)]] Sampler2D normalMap;

struct VIn { [Frame("world")] float3 pos : POSITION; [Frame("world")] float3 normal : NORMAL;
             [Frame("world")] float4 tangent : TANGENT; float2 uv : TEXCOORD; };
struct VOut { float4 pos : SV_Position; [Frame("world")] float3 normal : NORMAL;
              [Frame("world")] float4 tangent : TANGENT; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(VIn v) {
    VOut o; o.pos = mul(cam.viewProj, float4(v.pos, 1)); o.normal = v.normal; o.tangent = v.tangent; o.uv = v.uv; return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target {
    float3 N = normalize(i.normal);
    float3 T = normalize(i.tangent.xyz);
    float3 B = i.tangent.w * cross(N, T);
    float3 m = normalMap.Sample(i.uv).xyz * 2.0 - 1.0;
    float3 n = normalize(T * m.x + B * m.y + N * m.z);
    float3 L = normalize(float3(0.0, -1.0, 0.6));           // key light from the bottom of the screen (+V side)
    return float4(float3(0.75, 0.7, 0.65) * (0.1 + saturate(dot(n, L))), 1);
}
