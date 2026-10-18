import covenant;
struct Light { [Color("linear")] float3 color; };
StructuredBuffer<Light> lights;
// normals packed into an RGBA8 texture: not color, a deliberate idiom
[Color("data")] Sampler2D packedNormals;
[shader("fragment")]
float4 fsMain(float2 uv : TEXCOORD) : SV_Target {
    float3 n = packedNormals.Sample(uv).xyz * 2 - 1;
    return float4(n * lights[0].color, 1);
}
