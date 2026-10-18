import covenant;
// Azahar renderer_vulkan present shader (single screen, no 3D/interlace
// variants): draws the emulated top screen texture into the output frame.
struct PresentUniformData {
    column_major float4x4 modelview;     // screen rect in output pixels -> clip
    float4 i_resolution;
    float4 o_resolution;
};
[[vk::push_constant]] ConstantBuffer<PresentUniformData> push;
[[vk::binding(0, 0)]] [Color("srgb")] Sampler2D screen_textures;   // emulated LCD output, RGBA8

struct VOut { float4 pos : SV_Position; float2 frag_tex_coord : TEXCOORD0; };

[shader("vertex")]
VOut vertMain(float2 vert_position : POSITION, float2 vert_tex_coord : TEXCOORD0) {
    VOut o;
    o.pos = mul(push.modelview, float4(vert_position, 0.0, 1.0));
    o.frag_tex_coord = vert_tex_coord;
    return o;
}

[shader("fragment")]
float4 fragMain(VOut i) : SV_Target {
    return float4(screen_textures.Sample(i.frag_tex_coord).rgb, 1.0);
}
