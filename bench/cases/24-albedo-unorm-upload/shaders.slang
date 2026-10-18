import covenant;
[Color("srgb")] Sampler2D albedoMap;           // colour map, authored in sRGB
Sampler2D normalMap;                           // data map (tangent-space normals)
struct Light { float3 direction; float pad0; [Color("linear")] float3 color; float pad1; };
ConstantBuffer<Light> light;
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target {
    float3 n = normalize(normalMap.Sample(i.uv).xyz * 2 - 1);
    float ndl = max(dot(n, -light.direction), 0.0);
    float3 albedo = albedoMap.Sample(i.uv).rgb;
    return float4(albedo * light.color * ndl, 1);
}
