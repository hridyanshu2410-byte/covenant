import covenant;
// labelle-bgfx fs_yuv (reduced): converts the decoder's Y, U and V planes
// to RGB for display.
[[vk::binding(0, 0)]] Sampler2D s_texY;
[[vk::binding(1, 0)]] Sampler2D s_texU;
[[vk::binding(2, 0)]] Sampler2D s_texV;
#ifdef FIXED
struct YuvUniforms {
    float4 u_yuvOffsetGain;   // x = Y offset, y = Y gain, z = chroma offset
    float4 u_yuvCoeffs;       // x = R.Cr, y = G.Cb, z = G.Cr, w = B.Cb
};
[[vk::binding(3, 0)]] ConstantBuffer<YuvUniforms> yuv;
#endif

struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0.5, 1); return o;
}

[shader("fragment")]
float4 fsMain(VOut i) : SV_Target {
    float y = s_texY.Sample(i.uv).r;
    float u = s_texU.Sample(i.uv).r;
    float v = s_texV.Sample(i.uv).r;
#ifdef FIXED
    float Y = (y - yuv.u_yuvOffsetGain.x) * yuv.u_yuvOffsetGain.y;
    float Cb = u - yuv.u_yuvOffsetGain.z, Cr = v - yuv.u_yuvOffsetGain.z;
    float3 rgb = float3(Y + yuv.u_yuvCoeffs.x * Cr,
                        Y + yuv.u_yuvCoeffs.y * Cb + yuv.u_yuvCoeffs.z * Cr,
                        Y + yuv.u_yuvCoeffs.w * Cb);
#else
    // BT.601, limited range
    float Y = (y - 16.0 / 255.0) * 1.164383;
    float Cb = u - 0.5, Cr = v - 0.5;
    float3 rgb = float3(Y + 1.596027 * Cr,
                        Y - 0.391762 * Cb - 0.812968 * Cr,
                        Y + 2.017232 * Cb);
#endif
    return float4(saturate(rgb), 1);
}
