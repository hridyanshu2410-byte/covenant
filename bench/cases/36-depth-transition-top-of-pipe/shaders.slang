import covenant;
struct UBOVS {
    [Maps("view","clip")]   float4x4 projection;
    [Maps("object","view")] float4x4 modelview;
    [Maps("world","view")]  float4x4 skyboxModelview;
};
ConstantBuffer<UBOVS> ubo;
struct VOut { float4 pos : SV_Position; float3 dir : DIRECTION; };
[shader("vertex")]
VOut vsSkybox(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2) * 2 - 1;
    o.pos = float4(p, 1, 1);
    o.dir = float3(p, 1);
    return o;
}
[shader("fragment")]
float4 fsSkybox(VOut i) : SV_Target { return float4(normalize(i.dir) * 0.5 + 0.5, 1); }
[shader("vertex")]
VOut vsModel(uint vid : SV_VertexID) {
    const float3 tri[3] = { float3(-0.5, -0.5, 0), float3(0.5, -0.5, 0), float3(0, 0.5, 0) };
    [Frame("object")] float3 p = tri[vid];
    VOut o;
    o.pos = mul(ubo.projection, mul(ubo.modelview, float4(p, 1)));
    o.dir = float3(0, 0, 1);
    return o;
}
[shader("fragment")]
float4 fsModel(VOut i) : SV_Target { return float4(0.9, 0.9, 0.9, 1); }
