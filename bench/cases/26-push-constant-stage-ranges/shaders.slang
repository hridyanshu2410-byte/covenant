import covenant;
// Overlay push constants: the vertex stage reads the first 80 bytes, the
// fragment stage the last 32 (rpcs3 declares one range per stage).
struct OverlayPush {
    float4x4 transform;      // vertex: bytes 0..63
    float4   viewport;       // vertex: bytes 64..79
    float4   color;          // fragment: bytes 80..95
    float    time; float blur; float2 pad;   // fragment: bytes 96..111
};
[[vk::push_constant]] ConstantBuffer<OverlayPush> pc;
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p;
    float2 px = p * pc.viewport.zw + pc.viewport.xy;
    o.pos = mul(pc.transform, float4(px, 0, 1));
    return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target {
    float pulse = 0.5 + 0.5 * sin(pc.time);
    return pc.color * lerp(1.0, pulse, pc.blur);
}
