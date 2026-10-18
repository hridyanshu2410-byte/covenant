import covenant;
// Premultiplied output for a (ONE, ONE_MINUS_SRC_ALPHA) blend, with an encode.
struct Material { float4 baseColor; };
[[vk::binding(0, 0)]] ConstantBuffer<Material> material;
struct VOut { float4 pos : SV_Position; };
[shader("fragment")]
float4 encodeAfterPremultiply(VOut i) : SV_Target {
    float4 c = material.baseColor;
    c.rgb *= c.a;
    c.rgb = pow(c.rgb, float3(0.45454));
    return c;
}
[shader("fragment")]
float4 encodeThenPremultiply(VOut i) : SV_Target {
    float4 c = material.baseColor;
    c.rgb = pow(c.rgb, float3(0.45454));
    c.rgb *= c.a;
    return c;
}
[shader("fragment")]
float4 straightOut(VOut i) : SV_Target { return material.baseColor; }
