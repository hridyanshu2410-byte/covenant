import covenant;
struct Particle { float3 pos; float pad; };
[[vk::binding(0, 0)]] RWStructuredBuffer<Particle> particles;
[shader("compute")] [numthreads(64, 1, 1)]
void simMain(uint3 id : SV_DispatchThreadID) { particles[id.x].pos += float3(0, -0.01, 0); }
struct VIn { float3 pos : POSITION; float pad : PAD; };
[shader("vertex")]
float4 vsMain(VIn v) : SV_Position { return float4(v.pos, 1); }
[shader("fragment")]
float4 fsMain() : SV_Target { return float4(1, 1, 1, 1); }
