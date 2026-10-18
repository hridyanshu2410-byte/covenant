import covenant;
// A textured echo material: uniforms at set 0 binding 0, the base colour
// texture (combined image sampler) at set 0 binding 1.
struct Uniforms { [Maps("object","clip")] float4x4 u_WorldViewProjMatrix; float4 u_Color; };
[[vk::binding(0, 0)]] ConstantBuffer<Uniforms> ub;
[[vk::binding(1, 0)]] Sampler2D u_BaseColorSampler;
struct VIn { [Frame("object")] float3 a_Position : POSITION; float2 a_UV : TEXCOORD; };
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(VIn v) { VOut o; o.pos = mul(ub.u_WorldViewProjMatrix, float4(v.a_Position, 1)); o.uv = v.a_UV; return o; }
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return ub.u_Color * u_BaseColorSampler.Sample(i.uv); }
