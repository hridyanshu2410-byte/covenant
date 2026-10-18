import covenant;
struct Params { float sdrScale; float3 pad; };
[[vk::binding(0, 0)]] ConstantBuffer<Params> params;
[[vk::binding(1, 0)]] Sampler2D slide;
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("fragment")]
float4 composite(VOut i) : SV_Target { float4 c = slide.Sample(i.uv); c.rgb *= params.sdrScale; return c; }
[shader("fragment")]
float4 lit(VOut i) : SV_Target { float4 c = slide.Sample(i.uv); c.rgb = c.rgb * 0.8 + 0.2; return c; }
