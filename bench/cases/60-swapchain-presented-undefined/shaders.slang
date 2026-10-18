import covenant;
// Razix's forward test scene, reduced to one coloured triangle drawn straight
// into the swapchain image.
struct VOut { float4 pos : SV_Position; float3 color : COLOR; };
struct Push { float time; };
[[vk::push_constant]] Push pc;
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    const float2 p[3] = { float2(0, -0.5), float2(0.5, 0.5), float2(-0.5, 0.5) };
    const float3 c[3] = { float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1) };
    VOut o; o.pos = float4(p[vid] * (0.9 + 0.1 * sin(pc.time)), 0, 1); o.color = c[vid]; return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return float4(i.color, 1); }
