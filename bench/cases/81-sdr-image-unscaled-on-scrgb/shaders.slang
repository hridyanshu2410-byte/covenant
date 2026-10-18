import covenant;
// sldshow2 transition.wgsl (reduced): a full-screen pass that cross-fades
// slide A into slide B. The slides are 8-bit sRGB images (sampled through
// an sRGB view, so the shader sees linear values).
struct TransitionUniform {
    float blend;
    int mode;
    float2 aspect_ratio;
    float4 bg_color;
    int display_mode;      // 0 = SDR (clamp to [0,1]), 1 = HDR (allow > 1.0)
#ifdef FIXED
    float sdr_scale_a;     // SDR brightness scale for texture A (1.0 or ~2.54)
    float sdr_scale_b;     // SDR brightness scale for texture B (1.0 or ~2.54)
    float pad;
#else
    float3 pad;
#endif
};
[[vk::binding(0, 0)]] ConstantBuffer<TransitionUniform> material;
[[vk::binding(1, 0)]] [Color("srgb")] Sampler2D texture_a;
[[vk::binding(3, 0)]] [Color("srgb")] Sampler2D texture_b;

struct VertexOutput { float4 position : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VertexOutput vs_main(uint in_vertex_index : SV_VertexID) {
    VertexOutput o;
    float x = float(int(in_vertex_index) & 1);
    float y = float(int(in_vertex_index) >> 1);
    o.uv = float2(x * 2.0, y * 2.0);
    o.position = float4(x * 4.0 - 1.0, 1.0 - y * 4.0, 0.0, 1.0);
    return o;
}

float4 sample_a(float2 uv) {
    float4 c = texture_a.Sample(uv);
#ifdef FIXED
    c.rgb *= material.sdr_scale_a;
#endif
    return c;
}
float4 sample_b(float2 uv) {
    float4 c = texture_b.Sample(uv);
#ifdef FIXED
    c.rgb *= material.sdr_scale_b;
#endif
    return c;
}

[shader("fragment")]
float4 fs_main(VertexOutput i) : SV_Target {
    float4 color = lerp(sample_a(i.uv), sample_b(i.uv), material.blend);
    if (material.display_mode == 0) color = saturate(color);
    return color;
}
