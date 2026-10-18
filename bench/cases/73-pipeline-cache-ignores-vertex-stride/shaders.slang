import covenant;
// WebGPURenderer node material for MeshStandardMaterial on an InstancedMesh,
// reduced to position + instance offset and a flat colour.
struct Camera { [Maps("world","clip")] float4x4 viewProj; };
[[vk::binding(0, 0)]] ConstantBuffer<Camera> camera;
[[vk::binding(1, 0)]] StructuredBuffer<float4> instanceOffsets;
struct VIn { [Frame("object")] float3 position : POSITION; uint instance : SV_InstanceID; };
struct VOut { float4 pos : SV_Position; };
[shader("vertex")]
VOut vsMain(VIn v) {
    VOut o; o.pos = mul(camera.viewProj, float4(v.position + instanceOffsets[v.instance].xyz, 1)); return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return float4(0.6, 0.7, 0.9, 1); }
