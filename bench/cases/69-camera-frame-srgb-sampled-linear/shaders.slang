import covenant;
// Fabric: an Image Mesh draws the Camera Provider / Syphon Client texture
// into the document, which renders in linear rgba16Float; the output pass
// writes the document to the sRGB drawable (the hardware encodes linear ->
// sRGB on store).
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsQuad(uint vid : SV_VertexID) {
    const float2 q[6] = { float2(0, 0), float2(1, 0), float2(0, 1), float2(1, 0), float2(1, 1), float2(0, 1) };
    VOut o; o.uv = q[vid]; o.pos = float4(lerp(float2(-0.8, -0.6), float2(0.8, 0.6), q[vid]), 0, 1); return o;
}
[[vk::binding(0, 0)]] Sampler2D imageTexture;           // Image Mesh material input
[shader("fragment")]
float4 fsImageMesh(VOut i) : SV_Target { return imageTexture.Sample(i.uv); }

[shader("vertex")]
VOut vsFullscreen(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2); o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
[[vk::binding(1, 0)]] [Color("linear")] Sampler2D documentColor;   // the document's linear rgba16Float render
[shader("fragment")]
float4 fsOutput(VOut i) : SV_Target { return documentColor.Sample(i.uv); }
