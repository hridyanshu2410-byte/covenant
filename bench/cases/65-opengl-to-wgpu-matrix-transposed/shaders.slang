import covenant;
// learn-wgpu tutorial shader.wgsl (camera uniform + vertex position).
struct CameraUniform { [Maps("world","clip")] column_major float4x4 view_proj; };
[[vk::binding(0, 0)]] ConstantBuffer<CameraUniform> camera;
struct VIn { [Frame("world")] float3 position : POSITION; float3 color : COLOR; };
struct VOut { float4 clip_position : SV_Position; float3 color : COLOR; };
[shader("vertex")]
VOut vsMain(VIn v) {
    VOut o; o.color = v.color; o.clip_position = mul(camera.view_proj, float4(v.position, 1.0)); return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return float4(i.color, 1.0); }
