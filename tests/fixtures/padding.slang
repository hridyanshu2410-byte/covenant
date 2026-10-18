// Fixture for the uniform padding witness: 'b' is 16-byte aligned, so bytes
// 4..15 are padding.
import covenant;
struct Params { float a; float4 b; };
[[vk::binding(0, 0)]] ConstantBuffer<Params> params;
[shader("fragment")]
float4 fsMain(float2 uv : TEXCOORD) : SV_Target { return params.b * params.a; }
