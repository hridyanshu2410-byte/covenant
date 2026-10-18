import covenant;
// Ghostty shadertoy_prefix.glsl Globals block (current layout) and a user
// custom shader that tints selected text using the selection colours.
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
    float4 iCurrentCursor;
    float4 iPreviousCursor;
    [Color("srgb")] float4 iCurrentCursorColor;
    [Color("srgb")] float4 iPreviousCursorColor;
    int    iCurrentCursorStyle;
    int    iPreviousCursorStyle;
    int    iCursorVisible;
    float  iTimeCursorChange;
    float  iTimeFocus;
    int    iFocus;
    [Color("srgb")] float3 iPalette[256];
    [Color("srgb")] float3 iBackgroundColor;
    [Color("srgb")] float3 iForegroundColor;
    [Color("srgb")] float3 iCursorColor;
    [Color("srgb")] float3 iCursorText;
    [Color("srgb")] float3 iSelectionForegroundColor;
    [Color("srgb")] float3 iSelectionBackgroundColor;
};
[[vk::binding(1, 0)]] ConstantBuffer<Globals> globals;
[[vk::binding(0, 0)]] [Color("srgb")] Sampler2D iChannel0;

struct VOut { float4 pos : SV_Position; };
[shader("vertex")]
VOut full_screen_vertex(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.pos = float4(p * 2 - 1, 0, 1); return o;
}

// "selection glow": pixels whose colour matches the selection background get
// a soft halo of the selection foreground colour
void mainImage(out float4 fragColor, float2 fragCoord) {
    float2 uv = fragCoord / globals.iResolution.xy;
    float4 c = iChannel0.Sample(uv);
    float isSel = 1.0 - smoothstep(0.02, 0.08, distance(c.rgb, globals.iSelectionBackgroundColor));
    float pulse = 0.5 + 0.5 * sin(globals.iTime * 3.0);
    fragColor = float4(lerp(c.rgb, globals.iSelectionForegroundColor, isSel * 0.25 * pulse), c.a);
}

[shader("fragment")]
float4 main(VOut i) : SV_Target {
    float4 c;
    mainImage(c, i.pos.xy);
    return c;
}
