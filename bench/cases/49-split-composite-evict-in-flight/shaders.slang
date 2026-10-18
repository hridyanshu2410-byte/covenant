import covenant;
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
struct LayerRect { float4 rect; };   // x, y, w, h in [0,1] window space
[[vk::push_constant]] LayerRect layerRect;
[shader("vertex")]
VOut vsQuad(uint vid : SV_VertexID) {
    const float2 q[6] = { float2(0, 0), float2(1, 0), float2(0, 1), float2(1, 0), float2(1, 1), float2(0, 1) };
    VOut o; float2 p = layerRect.rect.xy + q[vid] * layerRect.rect.zw;
    o.uv = q[vid]; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
[[vk::binding(0, 0)]] [Color("srgb")] Sampler2D layerImage;
[shader("fragment")]
float4 fsLayer(VOut i) : SV_Target { return layerImage.Sample(i.uv); }
