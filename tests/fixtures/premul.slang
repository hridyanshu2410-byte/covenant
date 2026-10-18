import covenant;
[[vk::binding(0, 0)]] [Convention("alpha=premultiplied")] Sampler2D layer;
[[vk::binding(1, 0)]] Sampler2D straight;
[shader("fragment")]
float4 twice(float2 uv : TEXCOORD) : SV_Target {
    float4 fg = layer.Sample(uv);
    return float4(fg.rgb * 0.5 * fg.a, fg.a);
}
[shader("fragment")]
float4 once(float2 uv : TEXCOORD) : SV_Target {
    float4 fg = layer.Sample(uv);
    float4 s = straight.Sample(uv);
    return float4(fg.rgb * 0.5 + s.rgb * s.a, fg.a);
}
