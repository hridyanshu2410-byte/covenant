import covenant;
// Godot tonemap.glsl (reduced): the final pass of the 3D renderer. Reads
// the HDR colour buffer, applies exposure and (when the host sets the flag)
// converts linear -> sRGB because the target is a UNORM 2D/viewport buffer.
#define FLAG_USE_BCS               (1 << 0)
#define FLAG_USE_GLOW              (1 << 1)
#define FLAG_USE_AUTO_EXPOSURE     (1 << 2)
#define FLAG_USE_COLOR_CORRECTION  (1 << 3)
#define FLAG_USE_FXAA              (1 << 4)
#define FLAG_USE_8_BIT_DEBANDING   (1 << 5)
#define FLAG_CONVERT_TO_SRGB       (1 << 6)

struct Params {
    float exposure;
    float white;
    uint flags;
    float pad;
};
[[vk::push_constant]] Params params;

[[vk::binding(0, 0)]] [Color("linear")] Sampler2D source_color;

struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0.5, 1); return o;
}

float3 linear_to_srgb(float3 c) {
    c = saturate(c);
    return select(c < 0.0031308, c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055);
}

[shader("fragment")]
float4 fsMain(VOut i) : SV_Target {
    float3 color = source_color.Sample(i.uv).rgb * params.exposure;
    color = color / (1.0 + color / (params.white * params.white));   // Reinhard (white-point)
    if ((params.flags & FLAG_CONVERT_TO_SRGB) != 0) {
        color = linear_to_srgb(color);
    }
    return float4(color, 1.0);
}
