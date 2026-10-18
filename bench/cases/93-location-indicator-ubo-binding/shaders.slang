import covenant;
// maplibre-native WebGPU location_indicator shader (WGSL, ported to Slang;
// untextured variant). The renderer's shared bind group has the global
// paint params UBO at binding 0 and the per-drawable UBO at binding 4.
struct LocationIndicatorUBO {
    [Maps("object", "clip")] float4x4 matrix;
    [Color("linear")] float4 color;
};
#ifdef FIXED
[[vk::binding(4, 0)]] ConstantBuffer<LocationIndicatorUBO> ubo;
#else
[[vk::binding(0, 0)]] ConstantBuffer<LocationIndicatorUBO> ubo;
#endif

struct VIn { float2 a_pos : POSITION; };
struct VOut { float4 pos : SV_Position; };
[shader("vertex")]
VOut vsMain(VIn v) {
    VOut o; o.pos = mul(ubo.matrix, float4(v.a_pos, 0, 1)); return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return ubo.color; }
