import covenant;
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsFullscreen(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
// layer pass: antialiased particle, premultiplied output
[shader("fragment")]
float4 fsLayer(VOut i) : SV_Target {
    float r = length(i.uv - 0.5);
    float cov = saturate((0.3 - r) * 48.0);
    float3 color = float3(1.0, 1.0, 1.0);
    return float4(color * cov, cov);
}
[shader("fragment")]
float4 fsBackground(VOut i) : SV_Target { return float4(1, 1, 1, 1); }

[[vk::binding(0, 0)]] [Convention("alpha=premultiplied")] Sampler2D layerTexture;
[[vk::binding(1, 0)]] [Convention("alpha=premultiplied")] Sampler2D accumTexture;
struct CompositeParams { float opacity; uint blendMode; float2 pad; };
[[vk::binding(2, 0)]] ConstantBuffer<CompositeParams> composite;

[shader("fragment")]
float4 fsComposite(VOut i) : SV_Target {
    float4 fg = layerTexture.Sample(i.uv);
    float4 bg = accumTexture.Sample(i.uv);
    float opacity = composite.opacity;
    if (composite.blendMode == 1)            // add
        return float4(bg.rgb + fg.rgb * opacity, saturate(bg.a + fg.a * opacity));
    if (composite.blendMode == 2)            // screen
        return float4(1 - (1 - bg.rgb) * (1 - fg.rgb * opacity), max(bg.a, fg.a * opacity));
    // normal
#ifdef FIXED
    float a = saturate(fg.a) * opacity;
    float3 rgb = fg.rgb * opacity + bg.rgb * (1.0 - a);
    return float4(rgb, a + bg.a * (1.0 - a));
#else
    float a = fg.a * opacity;
    float3 rgb = fg.rgb * opacity * fg.a + bg.rgb * (1.0 - a);
    return float4(rgb, max(a, bg.a));
#endif
}
