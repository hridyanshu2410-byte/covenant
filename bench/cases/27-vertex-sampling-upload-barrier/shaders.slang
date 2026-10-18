import covenant;
// Flutter GPU-style heightfield: the vertex shader displaces a grid by a texture.
Sampler2D heightMap;
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    float h = heightMap.SampleLevel(p, 0).r;
    o.uv = p; o.pos = float4(p * 2 - 1, 0.5 - 0.25 * h, 1);
    return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return float4(i.uv, 0.6, 1); }
