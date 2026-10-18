import covenant;
// A Babylon.js screen-space shader (written for WebGL) that addresses a
// screen-sized texture by fragment position: uv = gl_FragCoord.xy / size.
// Babylon's screen-sized textures follow the WebGL convention: v = 0 is the
// bottom row of the screen.
struct Params {
    float2 screenSize;
#ifdef FIXED
    float2 bnFragCoordTargetSize;   // set per draw from the bound framebuffer (FlipFragCoordY)
#else
    float2 pad;
#endif
};
[[vk::binding(0, 0)]] ConstantBuffer<Params> params;
[[vk::binding(1, 0)]] Sampler2D screenTexture;

struct VOut { float4 pos : SV_Position; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2); o.pos = float4(p * 2 - 1, 0.5, 1); return o;
}

[shader("fragment")]
float4 fsMain(float4 fragCoord : SV_Position) : SV_Target {
#ifdef FIXED
    float4 glFragCoord = float4(fragCoord.x, params.bnFragCoordTargetSize.y - fragCoord.y, fragCoord.z, fragCoord.w);
#else
    float4 glFragCoord = fragCoord;
#endif
    float2 uv = glFragCoord.xy / params.screenSize;
    return screenTexture.Sample(uv);
}
