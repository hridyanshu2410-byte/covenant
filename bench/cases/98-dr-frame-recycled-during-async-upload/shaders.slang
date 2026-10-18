import covenant;
// vo_gpu_next main pass, reduced: the uploaded software-decoded RGB frame is
// sampled through an sRGB view and drawn full-screen.
[[vk::binding(0, 0)]] [Color("srgb")] Sampler2D frameTex;   // decoded video frame, sRGB-encoded
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return float4(frameTex.Sample(i.uv).rgb, 1.0); }
