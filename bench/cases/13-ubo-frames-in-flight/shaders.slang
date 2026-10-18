import covenant;
struct Camera { [Maps("world","clip")] float4x4 viewProj; };
ConstantBuffer<Camera> camera;
[shader("vertex")]
float4 vsMain(uint vid : SV_VertexID) : SV_Position {
    float2 p = float2((vid << 1) & 2, vid & 2);
    [Frame("world")] float3 world = float3(p * 2 - 1, 0);
    return mul(camera.viewProj, float4(world, 1));
}
[shader("fragment")]
float4 fsMain() : SV_Target { return float4(1, 0.5, 0.2, 1); }
