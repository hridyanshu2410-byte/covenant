import covenant;
#ifdef FIXED
struct SkyboxUniforms {
    [Maps("view","world")] float4x4 transform;   // mat4 first: no implicit padding
    float brightness;
    uint _webgl2_padding_8b; uint _webgl2_padding_12b; uint _webgl2_padding_16b;
};
#else
struct SkyboxUniforms {
    float brightness;                            // BUG: the host struct has transform first
    [Maps("view","world")] float4x4 transform;   // 16-byte aligned: implicit padding after brightness
    uint _webgl2_padding_8b; uint _webgl2_padding_12b; uint _webgl2_padding_16b;
};
#endif
ConstantBuffer<SkyboxUniforms> skybox;
SamplerCube skyCube;
struct VOut { float4 pos : SV_Position; float3 dir : DIRECTION; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2) * 2 - 1;
    o.pos = float4(p, 0, 1);
    [Frame("view")] float3 ray = float3(p, -1);
    o.dir = mul(skybox.transform, float4(ray, 0)).xyz;
    return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return float4(skyCube.Sample(i.dir).rgb * skybox.brightness, 1); }
