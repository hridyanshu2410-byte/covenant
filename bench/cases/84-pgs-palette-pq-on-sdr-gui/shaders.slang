import covenant;
// Kodi overlay renderer (GLES, reduced): draws the subtitle texture over the
// GUI (blend state omitted). The GUI surface is SDR.
[[vk::binding(0, 0)]] Sampler2D subtitle;
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p.x * 1.2 - 0.6, p.y * 0.3 + 0.5, 0.5, 1); return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return subtitle.Sample(i.uv); }
