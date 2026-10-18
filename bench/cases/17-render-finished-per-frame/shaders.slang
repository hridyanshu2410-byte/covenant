import covenant;
[shader("vertex")]
float4 vsMain(uint vid : SV_VertexID) : SV_Position {
    float2 p = float2((vid << 1) & 2, vid & 2);
    return float4(p * 2 - 1, 0, 1);
}
[shader("fragment")]
float4 fsMain() : SV_Target { return float4(0.2, 0.4, 0.8, 1); }
