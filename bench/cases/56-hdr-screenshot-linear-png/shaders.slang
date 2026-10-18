import covenant;
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsFullscreen(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
// canvas item (Sprite2D) drawn into the HDR 2D viewport: linear colour out
[[vk::binding(0, 0)]] [Color("srgb")] Sampler2D spriteTexture;    // 8-bit sRGB PNG, R8G8B8A8_SRGB view
[shader("vertex")]
VOut vsCanvasItem(uint vid : SV_VertexID) {
    const float2 q[6] = { float2(0, 0), float2(1, 0), float2(0, 1), float2(1, 0), float2(1, 1), float2(0, 1) };
    VOut o; o.uv = q[vid]; o.pos = float4(q[vid] * 1.6 - 0.8, 0, 1); return o;
}
[shader("fragment")]
float4 fsCanvasItem(VOut i) : SV_Target { return spriteTexture.Sample(i.uv); }

// viewport -> window (sRGB swapchain format encodes on store)
[[vk::binding(0, 0)]] Sampler2D viewportTexture;
[shader("fragment")]
float4 fsBlit(VOut i) : SV_Target { return viewportTexture.Sample(i.uv); }
