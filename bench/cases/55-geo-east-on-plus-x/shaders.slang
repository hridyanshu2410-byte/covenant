import covenant;
struct Camera { [Maps("world","clip")] float4x4 viewProj; };
[[vk::binding(0, 0)]] ConstantBuffer<Camera> camera;
struct VOut { float4 pos : SV_Position; float shade : SHADE; };

struct RoadIn { [Frame("world")] float3 position : POSITION; };
[shader("vertex")]
VOut vsRoad(RoadIn v) {
    VOut o; o.pos = mul(camera.viewProj, float4(v.position, 1)); o.shade = frac(v.position.z * 0.05); return o;
}
[shader("fragment")]
float4 fsRoad(VOut i) : SV_Target { return float4(float3(0.35, 0.35, 0.38) + 0.1 * step(0.5, i.shade), 1); }

// HUD plan: north-up map in screen space (+x east, +y north), drawn as thin quads
[shader("vertex")]
VOut vsHud(float2 position : POSITION) {
    VOut o; o.pos = float4(position.x, -position.y, 0, 1); o.shade = 1; return o;
}
[shader("fragment")]
float4 fsHud(VOut i) : SV_Target { return float4(1, 0.8, 0.1, 1); }
