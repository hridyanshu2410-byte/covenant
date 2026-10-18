import covenant;
struct Light { [Frame("world")] [Unit("m")] float3 position; [Color("linear")] float3 color; [Unit("m")] float radius; };
StructuredBuffer<Light> lights;
[shader("fragment")] float4 fsMain(float3 wp : WORLDPOS) : SV_Target { return float4(lights[0].color * length(lights[0].position - wp) / lights[0].radius, 1); }
