import covenant;
[Color("srgb")] Sampler2D cameraFrame;          // 8-bit camera frame, sRGB-encoded
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return float4(cameraFrame.Sample(i.uv).rgb, 1); }
