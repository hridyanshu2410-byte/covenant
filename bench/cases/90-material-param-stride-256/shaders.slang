import covenant;
// rx material pass (reduced): one Params block per material, bound as a
// dynamic uniform buffer at param_index * stride. Field names are the
// reduced shader's; the last row is the skin subsurface-scattering row
// that grew the block from 256 to 272 bytes.
struct Params {
    [Color("linear")] float4 base_color;
    [Color("linear")] float4 emissive;
    float4 pbr;                // metallic, roughness, occlusion, alpha cutoff
    float4 uv_transform;       // scale.xy, offset.xy
    float4 clearcoat;
    float4 sheen;
    float4 transmission;
    float4 anisotropy;
    float4 iridescence;
    uint4  texture_indices0;
    uint4  texture_indices1;
    uint4  flags;
    float4 reserved[4];
    [Color("linear")] float4 sss_color_radius;   // skin SSS: rgb scatter colour, w = radius (mm)
};
[[vk::binding(0, 0)]] ConstantBuffer<Params> material;

struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p * material.uv_transform.xy + material.uv_transform.zw;
    o.pos = float4(p * 2 - 1, 0.5, 1); return o;
}

[shader("fragment")]
float4 fsMain(VOut i) : SV_Target {
    float3 c = material.base_color.rgb;
    if ((material.flags.x & 1u) != 0) {          // skin: cheap wrap-lit SSS tint
        float w = saturate(material.sss_color_radius.w / 4.0);
        c = lerp(c, c * material.sss_color_radius.rgb, w);
    }
    return float4(c + material.emissive.rgb, material.base_color.a);
}
