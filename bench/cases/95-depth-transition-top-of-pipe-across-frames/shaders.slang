import covenant;
// Vulkan-Tutorial 27_depth_buffering shader (Slang version, reduced): two
// textured quads at different depths, transformed by the per-frame UBO.
struct UniformBufferObject {
    [Maps("object", "world")] column_major float4x4 model;
    [Maps("world", "view")]   column_major float4x4 view;
    [Maps("view", "clip")]    column_major float4x4 proj;
};
[[vk::binding(0, 0)]] ConstantBuffer<UniformBufferObject> ubo;

struct VSInput {
    [Frame("object")] float3 inPosition : POSITION;
    float3 inColor : COLOR;
};
struct VSOutput { float4 pos : SV_Position; float3 fragColor : COLOR; };

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput o;
    o.pos = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.inPosition, 1.0))));
    o.fragColor = input.inColor;
    return o;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_Target {
    return float4(input.fragColor, 1.0);
}
