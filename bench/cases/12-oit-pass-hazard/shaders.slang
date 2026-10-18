import covenant;
struct Fragment { float4 color; float depth; uint next; };
[[vk::binding(0, 0)]] RWStructuredBuffer<Fragment> fragments;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> counter;
struct VOut { float4 pos : SV_Position; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) { VOut o; float2 p = float2((vid << 1) & 2, vid & 2); o.pos = float4(p * 2 - 1, 0.5, 1); return o; }
// Pass 1: append transparent fragments to a list.
[shader("fragment")]
void appendMain(VOut i) {
    uint slot; InterlockedAdd(counter[0], 1, slot);
    fragments[slot].color = float4(1, 0, 0, 0.5); fragments[slot].depth = i.pos.z; fragments[slot].next = 0;
}
// Pass 2: resolve the list.
[shader("fragment")]
float4 resolveMain(VOut i) : SV_Target { return fragments[uint(i.pos.x) % 64].color; }
