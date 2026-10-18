import covenant;
struct Camera {
    [Maps("world","clip")] [Convention("depth=reverse_z")] float4x4 viewProj;   // near = 1, far = 0
};
ConstantBuffer<Camera> camera;
[shader("vertex")]
float4 vsMain(uint vid : SV_VertexID) : SV_Position {
    float2 p = float2((vid << 1) & 2, vid & 2);
    [Frame("world")] float3 w = float3(p * 2 - 1, 1);
    return mul(camera.viewProj, float4(w, 1));
}
[shader("fragment")]
float4 fsMain() : SV_Target { return float4(0.5, 0.5, 0.5, 1); }
