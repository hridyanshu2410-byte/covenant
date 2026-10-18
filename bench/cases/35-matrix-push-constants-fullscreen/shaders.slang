import covenant;
// vsg phong shader set: matrices in push constants
struct PushConstants {
    [Maps("view","clip")]   float4x4 projection;
    [Maps("object","view")] float4x4 modelView;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

struct VOut { float4 pos : SV_Position; float3 normal : NORMAL; };
[shader("vertex")]
VOut vsPhong(uint vid : SV_VertexID) {
    const float3 tri[3] = { float3(-0.5, -0.5, 0), float3(0.5, -0.5, 0), float3(0, 0.5, 0) };
    [Frame("object")] float3 p = tri[vid];
    VOut o;
    o.pos = mul(pc.projection, mul(pc.modelView, float4(p, 1)));
    o.normal = mul(pc.modelView, float4(0, 0, 1, 0)).xyz;
    return o;
}
[shader("fragment")]
float4 fsPhong(VOut i) : SV_Target {
    float ndl = saturate(dot(normalize(i.normal), normalize(float3(0.3, 0.5, 1))));
    return float4(float3(0.8, 0.7, 0.6) * (0.2 + ndl), 1);
}

// fullscreen tonemap: its own push constants, fragment stage only
struct TonemapParams { float exposure; float gamma; float vignette; float time; };
[[vk::push_constant]] ConstantBuffer<TonemapParams> tonemap;
struct FOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
FOut vsFullscreen(uint vid : SV_VertexID) {
    FOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
[shader("fragment")]
float4 fsTonemap(FOut i) : SV_Target {
    float2 d = i.uv - 0.5;
    float v = 1.0 - tonemap.vignette * dot(d, d) * 4.0;
    return float4(0, 0, 0, 1.0 - saturate(v * tonemap.exposure));
}
