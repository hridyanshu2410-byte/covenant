import covenant;
// three.js MeshStandardMaterial normal mapping for geometry without a tangent
// attribute (normal_fragment_maps + perturbNormal2Arb): the tangent frame is
// derived from screen-space derivatives of the view position and the UV.
struct Material {
    [Maps("object","view")] float4x4 modelViewMatrix;
    [Maps("view","clip")] float4x4 projectionMatrix;
    float2 normalScale;
    float2 pad;
};
[[vk::binding(0, 0)]] ConstantBuffer<Material> m;
[[vk::binding(1, 0)]] Sampler2D normalMap;

struct VIn { [Frame("object")] float3 position : POSITION; [Frame("object")] float3 normal : NORMAL; float2 uv : TEXCOORD; };
struct VOut { float4 pos : SV_Position; [Frame("view")] float3 vViewPosition : VIEWPOS;
              [Frame("view")] float3 vNormal : NORMAL; float2 vNormalMapUv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(VIn v) {
    VOut o;
    float4 mv = mul(m.modelViewMatrix, float4(v.position, 1));
    o.vViewPosition = -mv.xyz;
    o.vNormal = normalize(mul((float3x3)m.modelViewMatrix, v.normal));
    o.vNormalMapUv = v.uv;
    o.pos = mul(m.projectionMatrix, mv);
    return o;
}

float3 perturbNormal2Arb(float3 eye_pos, float3 surf_norm, float3 mapN, float faceDirection, float2 uv) {
    float3 q0 = ddx(eye_pos.xyz); float3 q1 = ddy(eye_pos.xyz);
    float2 st0 = ddx(uv); float2 st1 = ddy(uv);
    float3 N = normalize(surf_norm);
    float3 q1perp = cross(q1, N); float3 q0perp = cross(N, q0);
    float3 T = q1perp * st0.x + q0perp * st1.x;
    float3 B = q1perp * st0.y + q0perp * st1.y;
    float det = max(dot(T, T), dot(B, B));
    float scale = (det == 0.0) ? 0.0 : faceDirection * rsqrt(det);
    return normalize(T * (mapN.x * scale) + B * (mapN.y * scale) + N * mapN.z);
}

[shader("fragment")]
float4 fsMain(VOut i, bool front : SV_IsFrontFace) : SV_Target {
    float faceDirection = front ? 1.0 : -1.0;
    float3 mapN = normalMap.Sample(i.vNormalMapUv).xyz * 2.0 - 1.0;
    mapN.xy *= m.normalScale;
    float3 normal = perturbNormal2Arb(-i.vViewPosition, i.vNormal, mapN, faceDirection, i.vNormalMapUv);
    float3 lightDir = normalize(float3(0.0, 1.0, 0.3));            // key light from above (view space)
    float d = saturate(dot(normal, lightDir));
    return float4(float3(0.8, 0.8, 0.8) * (0.15 + d), 1);
}
