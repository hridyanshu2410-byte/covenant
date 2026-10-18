import covenant;
struct CameraUniforms {
    [Maps("world","view")] float4x4 view;
    [Maps("view","clip")]  float4x4 projection;
};
[[vk::binding(0, 0)]] ConstantBuffer<CameraUniforms> camera;
[[vk::binding(1, 0)]] StructuredBuffer<float4> visibleChunks;   // world-space chunk origins
struct VOut { float4 pos : SV_Position; float3 normal : NORMAL; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID, uint inst : SV_InstanceID) {
    float4 chunkOrigin = visibleChunks[inst];
    const int3 corner[8] = { int3(0,0,0), int3(1,0,0), int3(1,1,0), int3(0,1,0),
                             int3(0,0,1), int3(1,0,1), int3(1,1,1), int3(0,1,1) };
    const int idx[36] = { 0,1,2, 0,2,3, 5,4,7, 5,7,6, 4,0,3, 4,3,7, 1,5,6, 1,6,2, 3,2,6, 3,6,7, 4,5,1, 4,1,0 };
    uint v = vid;
    [Frame("world")] float3 w = chunkOrigin.xyz + float3(corner[idx[v]]) * float3(16, 16, 16);
    VOut o;
    o.pos = mul(camera.projection, mul(camera.view, float4(w, 1)));
    o.normal = float3(0, 1, 0);
    return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return float4(0.3, 0.6, 0.25, 1); }
