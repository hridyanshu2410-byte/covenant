import covenant;
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
VOut fullscreen(uint vid, float z) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, z, 1); return o;
}
// Opaque pass
[shader("vertex")]
VOut opaqueVs(uint vid : SV_VertexID) { return fullscreen(vid, 0.6); }
[shader("fragment")]
float4 opaqueFs(VOut i) : SV_Target { return float4(0.3, 0.3, 0.35, 1); }
// Transparent pass: soft particles fade near opaque geometry
Sampler2D sceneDepth;
[shader("vertex")]
VOut particleVs(uint vid : SV_VertexID) { return fullscreen(vid, 0.4); }
[shader("fragment")]
float4 particleFs(VOut i) : SV_Target {
    float scene = sceneDepth.Sample(i.uv).r;
    float fade = saturate((scene - i.pos.z) * 20.0);
    return float4(1, 0.6, 0.2, 0.5 * fade);
}
