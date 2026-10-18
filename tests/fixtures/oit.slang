struct Fragment { float4 color; float depth; uint next; };
[[vk::binding(0, 0)]] RWStructuredBuffer<Fragment> fragments;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> counter;
struct VOut { float4 pos : SV_Position; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) { VOut o; o.pos = float4(0, 0, 0.5, 1); return o; }
void append(uint slot, float z) { fragments[slot].depth = z; }
[shader("fragment")]
void appendMain(VOut i) { uint slot; InterlockedAdd(counter[0], 1, slot); append(slot, i.pos.z); }
[shader("fragment")]
float4 resolveMain(VOut i) : SV_Target { return fragments[uint(i.pos.x) % 64].color; }
