import covenant;
// One 2D vector builds both a world-space point and a clip-space position.
struct VOut { float4 pos : SV_Position; [Frame("world")] float3 world : WORLDPOS; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) { VOut o; float2 p = float2((vid << 1) & 2, vid & 2); o.world = float3(p, 0); o.pos = float4(p * 2 - 1, 0.5, 1); return o; }
