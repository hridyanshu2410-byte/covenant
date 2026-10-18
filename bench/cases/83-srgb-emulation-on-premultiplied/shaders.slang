import covenant;
// Filament surface_main.fs (reduced) at feature level 0: the swapchain has
// no sRGB format, so CONFIG_SRGB_SWAPCHAIN_EMULATION encodes in the shader.
// Transparent materials output premultiplied colour (BLEND_MODE_TRANSPARENT:
// blend ONE, ONE_MINUS_SRC_ALPHA).
struct FrameUniforms { int rec709; int3 pad; };
struct Material { [Color("linear")] float4 baseColor; };
[[vk::binding(0, 0)]] ConstantBuffer<FrameUniforms> frameUniforms;
[[vk::binding(1, 0)]] ConstantBuffer<Material> material;

struct VOut { float4 pos : SV_Position; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2); o.pos = float4(p - 0.5, 0.5, 1); return o;
}

void unpremultiply(inout float4 color) { color.rgb /= max(color.a, 1.175494e-38); }

[shader("fragment")]
float4 fsMain(VOut i) : SV_Target {
    float4 fragColor = material.baseColor;
    fragColor.rgb *= fragColor.a;                    // BLEND_MODE_TRANSPARENT: premultiplied output
    if (frameUniforms.rec709 != 0) {                 // CONFIG_SRGB_SWAPCHAIN_EMULATION
#ifdef FIXED
        unpremultiply(fragColor);
        fragColor.rgb = pow(fragColor.rgb, float3(0.45454));
        fragColor.rgb *= fragColor.a;
#else
        fragColor.rgb = pow(fragColor.rgb, float3(0.45454));
#endif
    }
    return fragColor;
}
