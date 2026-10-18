import covenant;
// BRITE-Engine pbr.vs / pbr.fs (raylib PBR shader), reduced to the albedo
// path and one directional light. raylib hands material and vertex colours to
// the shader as the engine's sRGB colours; pbr.fs linearizes them and
// gamma-encodes the result for the UNORM framebuffer.
struct Uniforms {
    [Maps("object","clip")] float4x4 mvp;
    [Color("srgb")] float4 colDiffuse;           // material albedo colour (raylib Color / 255)
};
[[vk::binding(0, 0)]] ConstantBuffer<Uniforms> u;
struct VIn { [Frame("object")] float3 vertexPosition : POSITION; [Frame("object")] float3 vertexNormal : NORMAL;
             [Color("srgb")] float4 vertexColor : COLOR; };
struct VOut { float4 pos : SV_Position; float3 fragNormal : NORMAL; float4 fragColor : COLOR; };
[shader("vertex")]
VOut vsMain(VIn v) {
    VOut o; o.pos = mul(u.mvp, float4(v.vertexPosition, 1)); o.fragNormal = v.vertexNormal; o.fragColor = v.vertexColor;
    return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target {
    float3 albedo = pow(u.colDiffuse.rgb * i.fragColor.rgb, float3(2.2));          // pbr.fs:155
    float3 L = normalize(float3(0.3, 0.6, -0.7));
    float3 color = albedo * (0.1 + saturate(dot(normalize(i.fragNormal), L)));
    color = pow(color, float3(1.0 / 2.2));                                        // gamma correction
    return float4(color, 1.0);
}
