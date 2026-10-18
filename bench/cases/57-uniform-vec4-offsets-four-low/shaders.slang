import covenant;
// brain_viz render uniforms (WGSL `struct Uniforms`, uniform address space).
// slicePlane is a vec4 and is 16-byte aligned: it follows dopamineTrails
// (float 36) at float 40, with floats 37-39 padding.
struct Uniforms {
    [Maps("world","clip")]  float4x4 viewProj;       // floats 0-15
    [Maps("object","world")] float4x4 model;         // floats 16-31
    float time;                                      // 32
    float pulse;                                     // 33
    float glow;                                      // 34
    float colorShift;                                // 35
    float dopamineTrails;                            // 36
    [Frame("world")] float4 slicePlane;              // 40-43: clip plane (normal, distance)
    float sparkle;                                   // 44
    float growth;                                    // 45
    float aberration;                                // 46
    float grain;                                     // 47
};
[[vk::binding(0, 0)]] ConstantBuffer<Uniforms> u;

struct VIn { [Frame("object")] float3 position : POSITION; };
struct VOut { float4 pos : SV_Position; [Frame("world")] float3 world : WORLDPOS; };
[shader("vertex")]
VOut vsMesh(VIn v) {
    VOut o;
    float4 w = mul(u.model, float4(v.position * (1 + 0.1 * u.growth), 1));
    o.world = w.xyz; o.pos = mul(u.viewProj, w); return o;
}
[shader("fragment")]
float4 fsMesh(VOut i) : SV_Target {
    if (dot(u.slicePlane.xyz, i.world) > u.slicePlane.w) discard;          // slice / clip plane
    float3 base = float3(0.8, 0.5, 0.6) + 0.2 * sin(u.time * u.pulse + float3(0, 2, 4) + u.colorShift);
    base += u.dopamineTrails * float3(0.3, 0.2, 0.0) + u.glow * 0.1;
    float n = frac(sin(dot(i.pos.xy, float2(12.9898, 78.233))) * 43758.5453);
    base += (n - 0.5) * u.grain + u.sparkle * step(0.995, n) + u.aberration * 0.01;
    return float4(base, 1);
}
