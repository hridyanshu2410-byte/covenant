// Fixture for the color-matrix witness: a BT.709 -> BT.2020 conversion
// applied as M * c (correct) or c * M (transposed).
import covenant;
[[vk::binding(0, 0)]] [Color("linear")] Sampler2D scene;
static const float3x3 bt709_to_bt2020 = float3x3(
    0.627403914928436, 0.329283028841019, 0.043313067406416,
    0.069097292423248, 0.919540405273438, 0.011362315155566,
    0.016391439363360, 0.088013306260109, 0.895595252513885);
static const float3x3 rot = float3x3(0, -1, 0, 1, 0, 0, 0, 0, 1);
[shader("fragment")]
float4 fsRight(float2 uv : TEXCOORD) : SV_Target { return float4(mul(bt709_to_bt2020, scene.Sample(uv).rgb), 1); }
[shader("fragment")]
float4 fsTransposed(float2 uv : TEXCOORD) : SV_Target { return float4(mul(scene.Sample(uv).rgb, bt709_to_bt2020), 1); }
[shader("fragment")]
float4 fsNotColor(float2 uv : TEXCOORD) : SV_Target { return float4(mul(rot, scene.Sample(uv).rgb), 1); }
