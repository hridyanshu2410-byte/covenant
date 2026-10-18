import covenant;
struct Camera { [Maps("world","clip")] float4x4 viewProj; };
[[vk::binding(0, 0)]] ConstantBuffer<Camera> camera;
[[vk::binding(0, 1)]] Sampler2D lightmap;      // set 1: per-surface lightmap
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    [Frame("world")] float3 w = float3(p * 2 - 1, 0);
    o.uv = p; o.pos = mul(camera.viewProj, float4(w, 1)); return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return float4(lightmap.Sample(i.uv).rgb, 1); }
