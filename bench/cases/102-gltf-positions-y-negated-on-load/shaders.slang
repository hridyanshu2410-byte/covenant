import covenant;
// Vulkan-Tutorial chapter 35 (glTF + KTX2) shader: model/view/proj UBO and a
// textured mesh (the base-colour texture is replaced by the vertex colour).
struct VSInput {
    [Frame("object")] float3 inPosition;
    [Color("linear")] float3 inColor;
    float2 inTexCoord;
};

struct UniformBuffer {
    [Maps("object", "world")] column_major float4x4 model;
    [Maps("world", "view")]   column_major float4x4 view;
    [Maps("view", "clip")]    column_major float4x4 proj;
};
[[vk::binding(0, 0)]] ConstantBuffer<UniformBuffer> ubo;

struct VSOutput {
    float4 pos : SV_Position;
    float3 fragColor;
    float2 fragTexCoord;
};

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    output.pos = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.inPosition, 1.0))));
    output.fragColor = input.inColor;
    output.fragTexCoord = input.inTexCoord;
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
    return float4(vertIn.fragColor, 1.0);
}
