// Fixture for the projection witness: clip = viewProj * world position.
import covenant;
struct Camera { [Maps("world","clip")] float4x4 viewProj; };
[[vk::binding(0, 0)]] ConstantBuffer<Camera> camera;
struct VIn { [Frame("world")] float3 position : POSITION; };
[shader("vertex")]
float4 vsMain(VIn v) : SV_Position { return mul(camera.viewProj, float4(v.position, 1)); }
