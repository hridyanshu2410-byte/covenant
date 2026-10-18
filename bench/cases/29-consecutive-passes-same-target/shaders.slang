import covenant;
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
// Emulated game draws into its framebuffers
[shader("fragment")]
float4 gameFs(VOut i) : SV_Target { return float4(i.uv, 0.25, 1); }
// Final output: the emulated display framebuffer is copied to the screen
Sampler2D displayFb;
[shader("fragment")]
float4 presentFs(VOut i) : SV_Target { return displayFb.Sample(i.uv); }
