import covenant;
struct Light { [Frame("world")] float3 direction; [Color("linear")] float3 color; };
ConstantBuffer<Light> sun;
Sampler2D lightmap;                            // meaning inferred from the bake pass that wrote it
[Color("srgb")] Sampler2D albedo;
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; [Frame("world")] float3 normal : NORMAL; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); o.normal = float3(0, 0, 1); return o;
}
// Bake pass: writes linear irradiance into the lightmap.
[shader("fragment")]
float4 bakeMain(VOut i) : SV_Target {
    return float4(sun.color * max(dot(normalize(i.normal), -sun.direction), 0), 1);
}
// Shading pass: albedo times baked light.
[shader("fragment")]
float4 shadeMain(VOut i) : SV_Target {
    return float4(albedo.Sample(i.uv).rgb * lightmap.Sample(i.uv).rgb, 1);
}
