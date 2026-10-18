import covenant;
[Color("linear")] Sampler2D hdrScene;          // HDR scene color, linear
struct PostFx { float gamma; float exposure; float2 pad; };
ConstantBuffer<PostFx> postfx;
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
// fs_composite: exposure, then pow(c, 1/gamma)
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target {
    float3 c = hdrScene.Sample(i.uv).rgb * postfx.exposure;
    c = pow(max(c, 0.0), 1.0 / postfx.gamma);
    return float4(c, 1);
}
