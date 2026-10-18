import covenant;
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsFullscreen(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
// sprite batch: textured quad with a tint (rgba) and a screen rect
struct SpritePush { float4 tint; float4 rect; };
[[vk::push_constant]] SpritePush sprite;
[[vk::binding(0, 0)]] [Convention("alpha=straight")] Sampler2D spriteTexture;   // PNG sprite, straight alpha
[shader("vertex")]
VOut vsSprite(uint vid : SV_VertexID) {
    const float2 q[6] = { float2(0, 0), float2(1, 0), float2(0, 1), float2(1, 0), float2(1, 1), float2(0, 1) };
    VOut o; float2 p = sprite.rect.xy + q[vid] * sprite.rect.zw;
    o.uv = q[vid]; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
[shader("fragment")]
float4 fsSprite(VOut i) : SV_Target { return spriteTexture.Sample(i.uv) * sprite.tint; }

// present pass: the render target's texel, unchanged; the pipeline's blend layers it
[[vk::binding(0, 0)]] Sampler2D layerTexture;
[shader("fragment")]
float4 fsLayer(VOut i) : SV_Target { return layerTexture.Sample(i.uv); }
