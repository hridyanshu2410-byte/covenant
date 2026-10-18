import covenant;
// The replayed draw: reads the restored vertex buffer.
struct VIn { float2 pos : POSITION; };
[shader("vertex")]
float4 vsMain(VIn v) : SV_Position { return float4(v.pos, 0.5, 1); }
[shader("fragment")]
float4 fsMain() : SV_Target { return float4(1, 0.5, 0, 1); }
