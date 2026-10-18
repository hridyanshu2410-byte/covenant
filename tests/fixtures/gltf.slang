import covenant;
struct Uniforms { [Color("srgb")] float4 colDiffuse; };
[[vk::binding(0, 0)]] ConstantBuffer<Uniforms> u;
[shader("fragment")]
float4 fsMain(float4 pos : SV_Position) : SV_Target { return float4(pow(u.colDiffuse.rgb, float3(2.2)), 1); }
