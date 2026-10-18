import covenant;
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
VOut fullscreen(uint vid) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
// Skia: vector UI into the surface
[shader("vertex")] VOut vsSkia(uint vid : SV_VertexID) { return fullscreen(vid); }
[shader("fragment")]
float4 fsSkia(VOut i) : SV_Target {
    float r = length(i.uv - 0.5);
    float a = saturate((0.4 - r) * 64.0);
    return float4(0.2 * a, 0.5 * a, 0.9 * a, a);   // premultiplied
}
// Stride SpriteBatch: composites the Skia texture
[[vk::binding(0, 0)]] Sampler2D spriteTexture;
[shader("vertex")] VOut vsSprite(uint vid : SV_VertexID) { return fullscreen(vid); }
[shader("fragment")]
float4 fsSprite(VOut i) : SV_Target { return spriteTexture.Sample(i.uv); }
