import covenant;
// RetroArch Vulkan (reduced): the menu/frame is drawn into the offscreen
// buffer (SDR, 8-bit), then the HDR pass converts it to the HDR10 back
// buffer (sRGB BT.709 -> linear -> BT.2020 -> PQ at the paper-white level).
struct MenuPush { float4 rect; [Color("srgb")] float4 color; };
[[vk::push_constant]] MenuPush menu;
[[vk::binding(0, 0)]] [Color("srgb")] Sampler2D offscreen;

struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0.5, 1); return o;
}

// menu quad pipeline (built for the offscreen format)
[shader("fragment")]
float4 fsMenu(VOut i) : SV_Target {
    float2 q = i.uv;
    bool inside = all(q >= menu.rect.xy) && all(q <= menu.rect.zw);
    return inside ? menu.color : float4(0.05, 0.05, 0.08, 1);
}

static const float PAPER_WHITE_NITS = 200.0;
float3 SrgbToLinear(float3 c) { return select(c <= 0.04045, c / 12.92, pow((c + 0.055) / 1.055, 2.4)); }
float3 Rec709ToRec2020(float3 c) {
    const float3x3 m = { 0.6274040, 0.3292820, 0.0433136,
                         0.0690970, 0.9195400, 0.0113612,
                         0.0163916, 0.0880132, 0.8955950 };
    return mul(m, c);
}
float3 LinearToPq(float3 c) {
    const float m1 = 2610.0 / 16384.0, m2 = 2523.0 / 32.0;
    const float c1 = 3424.0 / 4096.0, c2 = 2413.0 / 128.0, c3 = 2392.0 / 128.0;
    float3 p = pow(max(c, 0.0), m1);
    return pow((c1 + c2 * p) / (1.0 + c3 * p), m2);
}

// HDR pass pipeline (built for the HDR10 swapchain format)
[shader("fragment")]
float4 fsHdr(VOut i) : SV_Target {
    float3 c = offscreen.Sample(i.uv).rgb;
    c = Rec709ToRec2020(SrgbToLinear(c)) * (PAPER_WHITE_NITS / 10000.0);
    return float4(LinearToPq(c), 1);
}
