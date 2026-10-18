import covenant;
// ghostty renderer shaders.metal, cell_text pipeline (grayscale atlas only).
struct Uniforms {
    [Maps("world", "clip")] column_major float4x4 projection_matrix;
    float2 screen_size;
    float2 cell_size;
    uint2 grid_size;
    float4 grid_padding;
    float min_contrast;
    uint2 cursor_pos;
    [Color("srgb")] uint cursor_color;    // packed uchar4 (r, g, b, a)
    [Color("srgb")] uint bg_color;        // packed uchar4
    uint cursor_wide;
    uint use_display_p3;
    uint use_linear_blending;
    uint use_linear_correction;
};
[[vk::binding(0, 0)]] ConstantBuffer<Uniforms> uniforms;
[[vk::binding(1, 0)]] Texture2D<float4> textureGrayscale;

static const uint ATLAS_GRAYSCALE = 0;
static const uint IS_CURSOR_GLYPH = 2;

float4 unpack_uchar4(uint c) {
    return float4(float(c & 0xff), float((c >> 8) & 0xff), float((c >> 16) & 0xff), float(c >> 24));
}

// Converts a color from sRGB gamma encoding to linear.
float4 linearize(float4 srgb) {
    bool3 cutoff = srgb.rgb <= 0.04045;
    float3 lower = srgb.rgb / 12.92;
    float3 higher = pow((srgb.rgb + 0.055) / 1.055, 2.4);
    srgb.rgb = select(cutoff, lower, higher);
    return srgb;
}
// Converts a color from linear to sRGB gamma encoding.
float4 unlinearize(float4 linear) {
    bool3 cutoff = linear.rgb <= 0.0031308;
    float3 lower = linear.rgb * 12.92;
    float3 higher = pow(linear.rgb, 1.0 / 2.4) * 1.055 - 0.055;
    linear.rgb = select(cutoff, lower, higher);
    return linear;
}

float4 load_color(float4 in_color, bool display_p3, bool linear) {
    // 0 .. 255 -> 0.0 .. 1.0
    float4 color = in_color / 255.0f;
    if (display_p3 && !linear) {
        color.rgb *= color.a;
        return color;
    }
    color = linearize(color);
    // If we're not doing linear blending, then we need to
    // unlinearize after doing the color space conversion.
    if (!linear) {
        color = unlinearize(color);
    }
    // Premultiply our color by its alpha.
    color.rgb *= color.a;
    return color;
}

struct CellTextVertexIn {
    uint2 glyph_pos : GLYPH_POS;     // position in the atlas
    uint2 glyph_size : GLYPH_SIZE;
    int2 bearings : BEARINGS;
    uint2 grid_pos : GRID_POS;
    [Color("srgb")] uint4 color : COLOR;   // uchar4
    uint atlas : ATLAS;
    uint bools : BOOLS;
};
struct CellTextVertexOut {
    float4 position : SV_Position;
    nointerpolation uint atlas : ATLAS;
    nointerpolation float4 color : COLOR0;
    nointerpolation float4 bg_color : COLOR1;
    float2 tex_coord : TEXCOORD0;
};

[shader("vertex")]
CellTextVertexOut cell_text_vertex(uint vid : SV_VertexID, CellTextVertexIn input) {
    float2 cell_pos = uniforms.cell_size * float2(input.grid_pos);
    float2 corner;
    corner.x = float(vid == 1 || vid == 3);
    corner.y = float(vid == 2 || vid == 3);

    CellTextVertexOut o;
    o.atlas = input.atlas;
    float2 size = float2(input.glyph_size);
    float2 offset = float2(input.bearings);
    offset.y = uniforms.cell_size.y - offset.y;
    cell_pos = cell_pos + size * corner + offset;
    o.position = mul(uniforms.projection_matrix, float4(cell_pos.x, cell_pos.y, 0.0f, 1.0f));
    o.tex_coord = float2(input.glyph_pos) + float2(input.glyph_size) * corner;

    bool display_p3 = uniforms.use_display_p3 != 0;
    // Get our color. We always fetch a linearized version to
    // make it easier to handle minimum contrast calculations.
    o.color = load_color(float4(input.color), display_p3, true);
    // Get the BG color (global background; per-cell backgrounds omitted)
    o.bg_color = load_color(unpack_uchar4(uniforms.bg_color), display_p3, true);

    // Check if current position is under cursor (including wide cursor)
    bool is_cursor_pos = (input.grid_pos.x == uniforms.cursor_pos.x ||
                          uniforms.cursor_wide != 0 && input.grid_pos.x == uniforms.cursor_pos.x + 1) &&
                         input.grid_pos.y == uniforms.cursor_pos.y;
    // If this cell is the cursor cell, but we're not processing
    // the cursor glyph itself, then we need to change the color.
    if ((input.bools & IS_CURSOR_GLYPH) == 0 && is_cursor_pos) {
#ifdef FIXED
        o.color = load_color(unpack_uchar4(uniforms.cursor_color), display_p3, true);
#else
        o.color = load_color(unpack_uchar4(uniforms.cursor_color), display_p3, false);
#endif
    }
    return o;
}

[shader("fragment")]
float4 cell_text_fragment(CellTextVertexOut input) : SV_Target {
    // Our input color is always linear.
    float4 color = input.color;
    // If we're not doing linear blending, then we need to
    // re-apply the gamma encoding to our color manually.
    if (uniforms.use_linear_blending == 0) {
        color.rgb /= color.a;
        color = unlinearize(color);
        color.rgb *= color.a;
    }
    // Fetch our alpha mask for this pixel (pixel-coordinate sampling).
    float a = textureGrayscale.Load(int3(int2(input.tex_coord), 0)).r;
    color *= a;
    return color;
}
