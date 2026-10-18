import covenant;
// deck.gl SolidPolygonLayer + FillStyleExtension (fill-style shader module),
// reduced: positions are already in deck.gl common space (Web Mercator
// world units, +y north) relative to the coordinate origin.
struct ProjectUniforms {
    [Maps("world", "clip")] column_major float4x4 viewProjectionMatrix;
    float opacity;              // layer.opacity
};
[[vk::binding(0, 0)]] ConstantBuffer<ProjectUniforms> project;

struct FillUniforms {
    float2 patternTextureSize;
    uint patternEnabled;
    uint patternMask;
    float2 uvCoordinateOrigin;
    float2 uvCoordinateOrigin64Low;
};
[[vk::binding(1, 0)]] ConstantBuffer<FillUniforms> fill;
[[vk::binding(2, 0)]] [Color("srgb")] Sampler2D fill_patternTexture;

struct VSIn {
    [Frame("world")] float3 positions : POSITION;     // common space, +y = north
    [Color("srgb")] float4 fillColors : COLOR;        // normalized from Uint8
    float4 fillPatternFrames : PATTERN_FRAME;          // x, y, w, h in the atlas (pixels)
    float fillPatternScales : PATTERN_SCALE;
    float2 fillPatternOffsets : PATTERN_OFFSET;
    [Color("srgb")] float4 fillPatternBackgroundColors : PATTERN_BG;
};
struct VSOut {
    float4 position : SV_Position;
    float4 vColor : COLOR0;
    float2 fill_uv : TEXCOORD0;
    nointerpolation float4 fill_patternBounds : TEXCOORD1;
    nointerpolation float4 fill_patternPlacement : TEXCOORD2;
    nointerpolation float4 fill_backgroundColor : COLOR1;
};

static const float FILL_UV_SCALE = 512.0 / 40000000.0;

float2 glsl_mod(float2 x, float2 y) { return x - y * floor(x / y); }

[shader("vertex")]
VSOut vertMain(VSIn i) {
    VSOut o;
    o.position = mul(project.viewProjectionMatrix, float4(i.positions, 1.0));
    // vs:DECKGL_FILTER_GL_POSITION
    o.fill_uv = i.positions.xy;
    o.vColor = i.fillColors;
    // vs:DECKGL_FILTER_COLOR
    o.fill_patternBounds = i.fillPatternFrames / float4(fill.patternTextureSize, fill.patternTextureSize);
    o.fill_patternPlacement.xy = i.fillPatternOffsets;
    o.fill_patternPlacement.zw = i.fillPatternScales * i.fillPatternFrames.zw;
    o.fill_backgroundColor = i.fillPatternBackgroundColors;
    return o;
}

// Draw the pattern over the background fill (Porter-Duff source-over)
float4 fill_blendOverBackground(float4 pattern, float4 background) {
    if (background.a == 0.0 || project.opacity == 0.0) return pattern;
    float patternAlpha = pattern.a / project.opacity;
    float blendedAlpha = patternAlpha + background.a * (1.0 - patternAlpha);
    float3 blendedRGB = lerp(background.rgb, pattern.rgb, patternAlpha / blendedAlpha);
    return float4(blendedRGB, blendedAlpha * project.opacity);
}

[shader("fragment")]
float4 fragMain(VSOut i) : SV_Target {
    float4 color = float4(i.vColor.rgb, i.vColor.a * project.opacity);
    // fs:DECKGL_FILTER_COLOR
    if (fill.patternEnabled != 0) {
        float2 scale = FILL_UV_SCALE * i.fill_patternPlacement.zw;
        float2 patternUV = glsl_mod(glsl_mod(fill.uvCoordinateOrigin, scale) + fill.uvCoordinateOrigin64Low + i.fill_uv, scale) / scale;
        patternUV = glsl_mod(i.fill_patternPlacement.xy + patternUV, float2(1.0, 1.0));
#ifdef FIXED
        float2 texCoords = i.fill_patternBounds.xy + i.fill_patternBounds.zw * float2(patternUV.x, 1.0 - patternUV.y);
#else
        float2 texCoords = i.fill_patternBounds.xy + i.fill_patternBounds.zw * patternUV;
#endif
        float4 patternColor = fill_patternTexture.Sample(texCoords);
        color.a *= patternColor.a;
        if (fill.patternMask == 0) {
            color.rgb = patternColor.rgb;
        }
        color = fill_blendOverBackground(color, i.fill_backgroundColor);
    }
    return color;
}
