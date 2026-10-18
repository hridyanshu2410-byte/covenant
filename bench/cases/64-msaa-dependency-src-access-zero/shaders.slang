import covenant;
// Sascha Willems' multisampling example (mesh.vert / mesh.frag), reduced to a
// lit triangle.
struct UBO { [Maps("object","clip")] float4x4 mvp; float4 lightPos; };
[[vk::binding(0, 0)]] ConstantBuffer<UBO> ubo;
struct VOut { float4 pos : SV_Position; float3 color : COLOR; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    const float3 p[3] = { float3(-0.8, 0.7, 0.3), float3(0.8, 0.7, 0.3), float3(0, -0.8, 0.6) };
    VOut o; o.pos = mul(ubo.mvp, float4(p[vid], 1));
    o.color = float3(0.8, 0.8, 0.8) * saturate(dot(normalize(ubo.lightPos.xyz), float3(0, 0, -1)) * 0.75 + 0.25);
    return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return float4(i.color, 1); }
