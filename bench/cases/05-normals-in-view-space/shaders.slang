import covenant;
struct Camera { [Maps("object","world")] float4x4 model; [Maps("world","clip")] float4x4 viewProj; };
struct Light { [Frame("world")] float3 direction; [Color("linear")] float3 color; };
ConstantBuffer<Camera> camera;
ConstantBuffer<Light> sun;
struct VIn { [Frame("object")] float3 position : POSITION; [Frame("object")] float3 normal : NORMAL; };
struct VOut { float4 pos : SV_Position; float3 normal : NORMAL; };
[shader("vertex")]
VOut vsMain(VIn v) {
    VOut o;
    o.pos = mul(camera.viewProj, mul(camera.model, float4(v.position, 1)));
    o.normal = mul(camera.model, float4(v.normal, 0)).xyz;
    return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target {
    return float4(sun.color * max(dot(normalize(i.normal), -sun.direction), 0), 1);
}
