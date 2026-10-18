import covenant;
struct Camera { [Maps("object","world")] float4x4 model; [Maps("world","view")] float4x4 view; [Maps("view","clip")] float4x4 proj; };
struct Light { [Frame("world")] [Unit("m")] float3 position; [Color("linear")] float3 color; };
ConstantBuffer<Camera> camera;
StructuredBuffer<Light> lights;
[Color("srgb")] Sampler2D albedo;
struct VIn { float3 position : POSITION; float3 normal : NORMAL; float2 uv : TEXCOORD; };
struct VOut { float4 pos : SV_Position; float3 worldPos : WORLDPOS; float3 normal : NORMAL; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(VIn v) {
    VOut o;
    float4 wp = mul(camera.model, float4(v.position, 1));
    o.worldPos = wp.xyz;
    o.normal = mul(camera.model, float4(v.normal, 0)).xyz;
    o.pos = mul(camera.proj, mul(camera.view, wp));
    o.uv = v.uv;
    return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target {
    float3 L = normalize(lights[0].position - i.worldPos);
    float ndl = max(dot(normalize(i.normal), L), 0);
    float4 base = albedo.Sample(i.uv);
    return float4(ndl * base.rgb * lights[0].color, 1);
}
