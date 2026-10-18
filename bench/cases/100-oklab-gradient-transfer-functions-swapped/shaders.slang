import covenant;
// gpui_windows shaders.hlsl, quad pipeline reduced to the background fill
// (solid and linear gradient; no borders, corners, or pattern).
struct GlobalParams {
    float2 global_viewport_size;
    float2 global_pad;
};
[[vk::binding(1, 0)]] ConstantBuffer<GlobalParams> globals;

struct Bounds { float2 origin; float2 size; };
struct Hsla { float h; float s; float l; float a; };
struct LinearColorStop { [Color("srgb")] Hsla color; float percentage; };
struct Background {
    uint tag;            // 0 solid, 1 linear gradient
    uint color_space;    // 0 sRGB, 1 Oklab
    [Color("srgb")] Hsla solid;
    float gradient_angle_or_pattern_height;
    LinearColorStop colors[2];
    uint pad;
};
struct Quad {
    uint order;
    uint border_style;
    Bounds bounds;
    Background background;
};
[[vk::binding(0, 0)]] StructuredBuffer<Quad> quads;

static const float M_PI_F = 3.141592653f;

float4 to_device_position(float2 unit_vertex, Bounds bounds) {
    float2 position = unit_vertex * bounds.size + bounds.origin;
    float2 device_position = position / globals.global_viewport_size * float2(2.0, -2.0) + float2(-1.0, 1.0);
    return float4(device_position, 0., 1.);
}

// Convert linear RGB to sRGB
float3 linear_to_srgb(float3 color) {
#ifdef FIXED
    return pow(color, float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));
#else
    return pow(color, float3(2.2, 2.2, 2.2));
#endif
}

// Convert sRGB to linear RGB
float3 srgb_to_linear(float3 color) {
#ifdef FIXED
    return pow(color, float3(2.2, 2.2, 2.2));
#else
    return pow(color, float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));
#endif
}

/// Hsla to linear RGBA conversion.
float4 hsla_to_rgba(Hsla hsla) {
    float h = hsla.h * 6.0;
    float s = hsla.s, l = hsla.l, a = hsla.a;
    float c = (1.0 - abs(2.0 * l - 1.0)) * s;
    float x = c * (1.0 - abs(fmod(h, 2.0) - 1.0));
    float m = l - c / 2.0;
    float r = 0.0, g = 0.0, b = 0.0;
    if (h >= 0.0 && h < 1.0) { r = c; g = x; }
    else if (h >= 1.0 && h < 2.0) { r = x; g = c; }
    else if (h >= 2.0 && h < 3.0) { g = c; b = x; }
    else if (h >= 3.0 && h < 4.0) { g = x; b = c; }
    else if (h >= 4.0 && h < 5.0) { r = x; b = c; }
    else { r = c; b = x; }
    return float4(r + m, g + m, b + m, a);
}

// Converts a sRGB color to the Oklab color space.
float4 srgb_to_oklab(float4 color) {
    // Convert non-linear sRGB to linear sRGB
    color = float4(srgb_to_linear(color.rgb), color.a);
    float l = 0.4122214708 * color.r + 0.5363325363 * color.g + 0.0514459929 * color.b;
    float m = 0.2119034982 * color.r + 0.6806995451 * color.g + 0.1073969566 * color.b;
    float s = 0.0883024619 * color.r + 0.2817188376 * color.g + 0.6299787005 * color.b;
    float l_ = pow(l, 1.0 / 3.0), m_ = pow(m, 1.0 / 3.0), s_ = pow(s, 1.0 / 3.0);
    return float4(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        color.a);
}

// Converts an Oklab color to the sRGB color space.
float4 oklab_to_srgb(float4 color) {
    float l_ = color.r + 0.3963377774 * color.g + 0.2158037573 * color.b;
    float m_ = color.r - 0.1055613458 * color.g - 0.0638541728 * color.b;
    float s_ = color.r - 0.0894841775 * color.g - 1.2914855480 * color.b;
    float l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
    float3 linear_rgb = float3(
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s);
    // Convert linear sRGB to non-linear sRGB
    return float4(linear_to_srgb(linear_rgb), color.a);
}

struct GradientColor { float4 solid; float4 color0; float4 color1; };

GradientColor prepare_gradient_color(uint tag, uint color_space, Hsla solid, LinearColorStop colors[2]) {
    GradientColor output;
    output.solid = float4(0, 0, 0, 0); output.color0 = output.solid; output.color1 = output.solid;
    if (tag == 0) {
        output.solid = hsla_to_rgba(solid);
    } else if (tag == 1) {
        output.color0 = hsla_to_rgba(colors[0].color);
        output.color1 = hsla_to_rgba(colors[1].color);
        // Prepare color space in vertex for avoid conversion
        // in fragment shader for performance reasons
        if (color_space == 1) {
            output.color0 = srgb_to_oklab(output.color0);
            output.color1 = srgb_to_oklab(output.color1);
        }
    }
    return output;
}

float4 gradient_color(Background background, float2 position, Bounds bounds,
                      float4 solid_color, float4 color0, float4 color1) {
    float4 color = solid_color;
    if (background.tag == 1) {
        // -90 degrees to match the CSS gradient angle.
        float gradient_angle = background.gradient_angle_or_pattern_height;
        float radians = (fmod(gradient_angle, 360.0) - 90.0) * (M_PI_F / 180.0);
        float2 direction = float2(cos(radians), sin(radians));
        if (bounds.size.x > bounds.size.y) direction.y *= bounds.size.y / bounds.size.x;
        else direction.x *= bounds.size.x / bounds.size.y;
        float2 half_size = bounds.size * 0.5;
        float2 center = bounds.origin + half_size;
        float t = dot(position - center, direction) / length(direction);
        if (abs(direction.x) > abs(direction.y)) t = (t + half_size.x) / bounds.size.x;
        else t = (t + half_size.y) / bounds.size.y;
        t = (t - background.colors[0].percentage) / (background.colors[1].percentage - background.colors[0].percentage);
        t = clamp(t, 0.0, 1.0);
        if (background.color_space == 0) {
            color = lerp(color0, color1, t);
        } else {
            float4 oklab_color = lerp(color0, color1, t);
            color = oklab_to_srgb(oklab_color);
        }
    }
    return color;
}

struct QuadVertexOutput {
    float4 position : SV_Position;
    nointerpolation float4 background_solid : COLOR0;
    nointerpolation float4 background_color0 : COLOR1;
    nointerpolation float4 background_color1 : COLOR2;
    nointerpolation uint quad_id : TEXCOORD0;
};

[shader("vertex")]
QuadVertexOutput quad_vertex(uint vertex_id : SV_VertexID, uint quad_id : SV_InstanceID) {
    static const float2 corners[6] = { float2(0, 0), float2(1, 0), float2(0, 1), float2(0, 1), float2(1, 0), float2(1, 1) };
    float2 unit_vertex = corners[vertex_id];
    Quad quad = quads[quad_id];
    GradientColor gradient = prepare_gradient_color(quad.background.tag, quad.background.color_space,
                                                     quad.background.solid, quad.background.colors);
    QuadVertexOutput output;
    output.position = to_device_position(unit_vertex, quad.bounds);
    output.background_solid = gradient.solid;
    output.background_color0 = gradient.color0;
    output.background_color1 = gradient.color1;
    output.quad_id = quad_id;
    return output;
}

[shader("fragment")]
float4 quad_fragment(QuadVertexOutput input) : SV_Target {
    Quad quad = quads[input.quad_id];
    float4 background_color = gradient_color(quad.background, input.position.xy, quad.bounds,
        input.background_solid, input.background_color0, input.background_color1);
    return float4(background_color.rgb * background_color.a, background_color.a);
}
