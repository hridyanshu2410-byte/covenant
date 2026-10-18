import covenant;
// mesh_view_bindings, with DEPTH_PREPASS and MOTION_VECTOR_PREPASS defined
struct View { [Maps("world","clip")] float4x4 viewProj; float4 viewport; };
[[vk::binding(0, 0)]]  ConstantBuffer<View> view;
[[vk::binding(17, 0)]] Sampler2D depthPrepassTexture;
[[vk::binding(19, 0)]] Sampler2D motionVectorPrepassTexture;

struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p;
    [Frame("world")] float3 w = float3(p * 2 - 1, 0.5);
    o.pos = mul(view.viewProj, float4(w, 1));
    return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target {
    float2 mv = motionVectorPrepassTexture.Sample(i.uv).rg;
    float d = depthPrepassTexture.Sample(i.uv - mv).r;
    return float4(float3(0.6, 0.6, 0.6) * (1.0 - d * 0.5), 1);
}
