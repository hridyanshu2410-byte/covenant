import covenant;
struct Shadow { [Maps("world","clip")] [Convention("depth=reverse_z")] float4x4 lightViewProj; float bias; };
ConstantBuffer<Shadow> shadow;
Sampler2D shadowMap;
struct VOut { float4 pos : SV_Position; [Frame("world")] float3 world : WORLDPOS; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) { VOut o; float2 p = float2((vid << 1) & 2, vid & 2); o.world = float3(p, 0); o.pos = float4(p * 2 - 1, 0.5, 1); return o; }
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target {
    float4 lp = mul(shadow.lightViewProj, float4(i.world, 1));
    float d = lp.z / lp.w;
#ifdef FIXED
    float lit = (d + shadow.bias >= shadowMap.Sample(lp.xy * 0.5 + 0.5).r) ? 1 : 0;   // reverse-Z: larger is nearer
#else
    float lit = (d - shadow.bias >= shadowMap.Sample(lp.xy * 0.5 + 0.5).r) ? 1 : 0;   // BUG: forward-Z bias sign
#endif
    return float4(lit.xxx, 1);
}
