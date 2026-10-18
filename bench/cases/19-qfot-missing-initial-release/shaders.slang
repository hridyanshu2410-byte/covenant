import covenant;
struct Particle { float4 pos; float4 vel; };
[[vk::binding(0, 0)]] RWStructuredBuffer<Particle> particles;
[shader("compute")] [numthreads(256, 1, 1)]
void integrateMain(uint3 id : SV_DispatchThreadID) {
    Particle p = particles[id.x];
    p.pos.xyz += p.vel.xyz * 0.001;
    particles[id.x] = p;
}
struct VIn { float4 pos : POSITION; float4 vel : VELOCITY; };
[shader("vertex")]
float4 vsMain(VIn v) : SV_Position { return float4(v.pos.xyz, 1); }
[shader("fragment")]
float4 fsMain() : SV_Target { return float4(1, 0.9, 0.6, 1); }
