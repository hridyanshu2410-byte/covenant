import covenant;
[Color("srgb")] Sampler2D albedo;          // texture content is sRGB-encoded
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { float3 c = albedo.Sample(i.uv).rgb;
#ifndef FIXED
    c = pow(c, 2.2);                           // BUG: the sRGB view already decoded
#endif
    return float4(c, 1); }
