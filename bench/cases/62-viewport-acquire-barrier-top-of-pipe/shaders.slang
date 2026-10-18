import covenant;
// imgui_impl_vulkan's glsl_shader.vert / .frag: textured, vertex-coloured
// triangles in display coordinates.
struct VIn { float2 pos : POSITION; float2 uv : TEXCOORD; float4 col : COLOR; };
struct VOut { float4 pos : SV_Position; float4 col : COLOR; float2 uv : TEXCOORD; };
struct PC { float2 scale; float2 translate; };
[[vk::push_constant]] PC pc;
[[vk::binding(0, 0)]] Sampler2D sTexture;
[shader("vertex")]
VOut vsMain(VIn v) {
    VOut o; o.col = v.col; o.uv = v.uv; o.pos = float4(v.pos * pc.scale + pc.translate, 0, 1); return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return i.col * sTexture.Sample(i.uv); }
