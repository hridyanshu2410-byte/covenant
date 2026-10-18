import covenant;
// Kodi GLES GUI shader SM_TEXTURE_NOBLEND (gles_shader_texture_noblend.frag)
// built with KODI_LIMITED_RANGE, plus its default vertex shader.
struct GUIUniforms {
    [Maps("world", "clip")] column_major float4x4 m_proj;   // includes the model-view
    float m_sdrPeak;
#ifdef FIXED
    float m_pma;
#endif
};
[[vk::binding(0, 0)]] ConstantBuffer<GUIUniforms> gui;
[[vk::binding(1, 0)]] [Color("srgb")] Sampler2D m_samp0;

struct VOut { float4 pos : SV_Position; float4 m_cord0 : TEXCOORD0; };

struct VIn { [Frame("world")] float4 m_attrpos : POSITION; float4 m_attrcord0 : TEXCOORD0; };

[shader("vertex")]
VOut vertMain(VIn v) {
    VOut o;
    o.pos = mul(gui.m_proj, v.m_attrpos);
    o.m_cord0 = v.m_attrcord0;
    return o;
}

[shader("fragment")]
float4 fragMain(VOut i) : SV_Target {
    float4 rgb = m_samp0.Sample(i.m_cord0.xy);
    // KODI_LIMITED_RANGE
    rgb.rgb *= (235.0 - 16.0) / 255.0;
#ifdef FIXED
    rgb.rgb += lerp(1.0, rgb.a, gui.m_pma) * 16.0 / 255.0;
#else
    rgb.rgb += 16.0 / 255.0;
#endif
    return rgb;
}
