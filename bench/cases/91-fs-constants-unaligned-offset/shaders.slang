import covenant;
// bgfx predefined/user uniforms (reduced): the vertex and fragment shader
// each have their own constant block; the renderer packs both into one
// per-draw region of the constant ring buffer.
struct VsUniforms {
    [Maps("object", "clip")] float4x4 u_modelViewProj;
    [Maps("object", "world")] float4x4 u_model;
    [Maps("world", "view")] float4x4 u_view;
    [Maps("view", "clip")] float4x4 u_proj;
    float4 u_viewRect;
};
struct FsUniforms {
    [Color("linear")] float4 u_tint;
    float4 u_params;          // x = alpha ref, y = time
};
[[vk::binding(0, 0)]] ConstantBuffer<VsUniforms> vsc;
[[vk::binding(1, 0)]] ConstantBuffer<FsUniforms> fsc;

struct VIn { float3 pos : POSITION; };
struct VOut { float4 pos : SV_Position; float3 wpos : TEXCOORD; };
[shader("vertex")]
VOut vsMain(VIn v) {
    VOut o;
    o.pos = mul(vsc.u_modelViewProj, float4(v.pos, 1));
    o.wpos = mul(vsc.u_model, float4(v.pos, 1)).xyz;
    return o;
}

[shader("fragment")]
float4 fsMain(VOut i) : SV_Target {
    float a = saturate(1.0 - length(i.wpos.xy));
    if (a < fsc.u_params.x) discard;
    return float4(fsc.u_tint.rgb * a, 1);
}
