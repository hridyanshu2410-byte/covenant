import covenant;
struct Camera { [Maps("object","world")] float4x4 model; [Maps("world","view")] float4x4 view; [Maps("view","clip")] float4x4 proj; };
ConstantBuffer<Camera> camera;
[shader("vertex")]
float4 vsMain(float3 p : POSITION) : SV_Position {
    float4x4 mvp = mul(camera.proj, mul(camera.model, camera.view));
    return mul(mvp, float4(p, 1));
}
