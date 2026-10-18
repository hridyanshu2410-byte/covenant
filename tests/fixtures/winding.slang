import covenant;
// One vertex stage, a per-draw model matrix: the second draw mirrors the mesh
// (scale.y = -1), which reverses its window-space winding.
struct U { [Maps("object", "clip")] column_major float4x4 mvp; };
[[vk::binding(0, 0)]] ConstantBuffer<U> u;
[shader("vertex")]
float4 vsMain(float3 pos : POSITION) : SV_Position { return mul(u.mvp, float4(pos, 1)); }
