import covenant;
// PlayCanvas lit shader (reduced): one spot light, inverse-squared falloff,
// lighting a floor quad. The light's colour uniform already includes the
// intensity (Light._updateLinearColor: colorLinear = color * intensity).
struct Camera { [Maps("world","clip")] float4x4 viewProj; };
struct SpotLight {
    [Frame("world")] float4 position;      // xyz
    [Frame("world")] float4 direction;     // xyz, unit
    float4 colorLinear;                    // rgb * intensity
    float innerConeCos; float outerConeCos; float2 pad;
};
[[vk::binding(0, 0)]] ConstantBuffer<Camera> cam;
[[vk::binding(1, 0)]] ConstantBuffer<SpotLight> light;

struct VIn { [Frame("world")] float3 pos : POSITION; };
struct VOut { float4 pos : SV_Position; [Frame("world")] float3 wpos : WORLDPOS; };
[shader("vertex")]
VOut vsMain(VIn v) { VOut o; o.wpos = v.pos; o.pos = mul(cam.viewProj, float4(v.pos, 1)); return o; }

[shader("fragment")]
float4 fsMain(VOut i) : SV_Target {
    float3 toLight = light.position.xyz - i.wpos;
    float d2 = dot(toLight, toLight);
    float3 L = toLight * rsqrt(d2);
    float cosAngle = dot(-L, light.direction.xyz);
    float spot = smoothstep(light.outerConeCos, light.innerConeCos, cosAngle);
    float ndotl = saturate(L.y);                          // floor normal +Y
    float3 radiance = light.colorLinear.rgb * spot * ndotl / d2;
    return float4(radiance * float3(0.6, 0.6, 0.6), 1);
}
