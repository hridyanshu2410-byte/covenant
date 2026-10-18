import covenant;
// Ghostty custom shader pipeline: shadertoy_prefix.glsl's Globals block and
// a user "cursor highlight" shader (in the style of the community cursor
// shaders that read iCurrentCursor), run as a full-screen post pass over the
// terminal image.
struct Globals {
    float3 iResolution;
    float  iTime;
    float  iTimeDelta;
    float  iFrameRate;
    int    iFrame;
    float  iChannelTime[4];
    float3 iChannelResolution[4];
    float4 iMouse;
    float4 iDate;
    float  iSampleRate;
    float4 iCurrentCursor;        // x, y = left / +Y edge in fragCoord pixels; z, w = size
    float4 iPreviousCursor;
    [Color("srgb")] float4 iCurrentCursorColor;
    [Color("srgb")] float4 iPreviousCursorColor;
    float  iTimeCursorChange;
};
[[vk::binding(1, 0)]] ConstantBuffer<Globals> globals;
[[vk::binding(0, 0)]] [Color("srgb")] Sampler2D iChannel0;   // the rendered terminal

struct VOut { float4 pos : SV_Position; };
[shader("vertex")]
VOut full_screen_vertex(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.pos = float4(p * 2 - 1, 0, 1); return o;
}

// rectangle given by its left / +Y edge corner and size, in fragCoord space
float sdRect(float2 p, float4 rect) {
    float2 minc = float2(rect.x, rect.y - rect.w);
    float2 maxc = float2(rect.x + rect.z, rect.y);
    float2 d = max(minc - p, p - maxc);
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

void mainImage(out float4 fragColor, float2 fragCoord) {
    float2 uv = fragCoord / globals.iResolution.xy;
    fragColor = iChannel0.Sample(uv);
    // glow around the current cursor, fading since the last cursor change
    float d = sdRect(fragCoord, globals.iCurrentCursor);
    float fade = clamp(1.0 - (globals.iTime - globals.iTimeCursorChange) * 2.0, 0.0, 1.0);
    float glow = (1.0 - smoothstep(0.0, 6.0, d)) * fade;
    fragColor.rgb = lerp(fragColor.rgb, globals.iCurrentCursorColor.rgb, glow * 0.6);
}

[shader("fragment")]
float4 main(VOut i) : SV_Target {
    float4 c;
    mainImage(c, i.pos.xy);
    return c;
}
