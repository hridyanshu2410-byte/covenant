import covenant;
#ifdef FIXED
struct GlassCompositeUniforms {
    float ior; float blur_radius; float dispersion; float rim_power;
    float sheen_intensity; float light_angle; float roughness; float saturation;
    float brightness; float thickness; float curvature; float corner_radius;
    float4 tint_color;
    float4 glass_bounds;
    float2 resolution;
    float2 _padding;
};
[[vk::binding(0, 0)]] ConstantBuffer<GlassCompositeUniforms> uniforms;
#else
struct OpticalUniforms {
    float ior; float blur_radius; float dispersion; float rim_power; float sheen_intensity;
    float light_angle; float roughness; float thickness; float curvature;
    float padding;
    float4 tint_color;
    float2 resolution;
    float4 glass_bounds;
    float corner_radius;
    float3 padding2;
};
[[vk::binding(0, 0)]] ConstantBuffer<OpticalUniforms> uniforms;
#endif
[[vk::binding(1, 0)]] Sampler2D blurred_texture;

float roundedRectSdf(float2 p, float2 center, float2 halfSize, float r) {
    float2 q = abs(p - center) - halfSize + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target {
    float2 px = i.uv * uniforms.resolution;
    float2 center = uniforms.glass_bounds.xy + uniforms.glass_bounds.zw * 0.5;
    float d = roundedRectSdf(px, center, uniforms.glass_bounds.zw * 0.5, uniforms.corner_radius);
    if (d > 0.0) discard;
    float edge = saturate(1.0 + d / uniforms.thickness);
    float bend = pow(edge, 1.0 / uniforms.curvature) * (1.0 - 1.0 / uniforms.ior);
    float2 offset = normalize(px - center + 1e-3) * bend * uniforms.thickness / uniforms.resolution;
    float3 c;
    c.r = blurred_texture.Sample(i.uv - offset * (1.0 + uniforms.dispersion)).r;
    c.g = blurred_texture.Sample(i.uv - offset).g;
    c.b = blurred_texture.Sample(i.uv - offset * (1.0 - uniforms.dispersion)).b;
    c = lerp(c, uniforms.tint_color.rgb, uniforms.tint_color.a);
#ifdef FIXED
    float luma = dot(c, float3(0.2126, 0.7152, 0.0722));
    c = max(lerp(float3(luma), c, uniforms.saturation), 0.0) * uniforms.brightness;
#endif
    float2 l = float2(cos(uniforms.light_angle), sin(uniforms.light_angle));
    float rim = pow(edge, uniforms.rim_power) * saturate(dot(normalize(px - center + 1e-3), l));
    c += rim * uniforms.sheen_intensity * (1.0 - uniforms.roughness);
    return float4(c, 1);
}
