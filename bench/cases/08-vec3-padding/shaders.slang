import covenant;
struct FrameInfo { float3 lightDir; float intensity; float3 tint; float time; };
ConstantBuffer<FrameInfo> frameInfo;           // std140 (uniform buffer)
[shader("vertex")]
float4 vsMain(uint vid : SV_VertexID) : SV_Position { float2 p = float2((vid << 1) & 2, vid & 2); return float4(p * 2 - 1, 0, 1); }
[shader("fragment")]
float4 fsMain() : SV_Target { return float4(frameInfo.tint * frameInfo.intensity * (0.5 + 0.5 * sin(frameInfo.time)), 1); }
