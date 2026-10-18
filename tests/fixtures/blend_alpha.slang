// Fixture for the blend-state / alpha-convention rule.
import covenant;
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
struct Tint { float4 color; };
[[vk::push_constant]] Tint tint;
[[vk::binding(0, 0)]] [Convention("alpha=straight")] Sampler2D sprite;   // PNG: straight alpha
[[vk::binding(1, 0)]] Sampler2D plain;                                    // convention unknown
[[vk::binding(2, 0)]] Sampler2D layer;                                    // an earlier pass's target
[[vk::binding(3, 0)]] [Convention("alpha=straight")] Sampler2D layerStraight;
[shader("fragment")]
float4 fsSprite(VOut i) : SV_Target { return sprite.Sample(i.uv) * tint.color; }     // straight
[shader("fragment")]
float4 fsPremul(VOut i) : SV_Target {                     // premultiplies in the shader
    float4 c = plain.Sample(i.uv);
    return float4(c.rgb * c.a, c.a);
}
[shader("fragment")]
float4 fsLayer(VOut i) : SV_Target { return layer.Sample(i.uv); }          // passes the texel on
[shader("fragment")]
float4 fsLayerStraight(VOut i) : SV_Target { return layerStraight.Sample(i.uv); }
