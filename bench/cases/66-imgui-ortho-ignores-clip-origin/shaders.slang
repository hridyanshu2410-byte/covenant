import covenant;
// imgui_impl_opengl3 vertex/fragment shaders (GLSL 130 variant):
//   Frag_UV = UV; Frag_Color = Color; gl_Position = ProjMtx * vec4(Position.xy, 0, 1);
struct VIn {
    float2 Position : POSITION;   // ImGui display coordinates: origin top-left, y down
    float2 UV : TEXCOORD;
    float4 Color : COLOR;
};
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; float4 color : COLOR; };
struct Proj { column_major float4x4 ProjMtx; };
[[vk::binding(0, 0)]] ConstantBuffer<Proj> proj;
[[vk::binding(1, 0)]] Sampler2D Texture;
[shader("vertex")]
VOut vsMain(VIn v) {
    VOut o; o.uv = v.UV; o.color = v.Color; o.pos = mul(proj.ProjMtx, float4(v.Position, 0, 1)); return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return i.color * Texture.Sample(i.uv); }
