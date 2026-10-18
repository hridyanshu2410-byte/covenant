import covenant;
// moonlight-qt D3D11VA video shaders: d3d11_vertex.hlsl and the YUV 4:4:4
// Y410 pixel shader (d3d11_yuv444_pixel_start/end.hlsli + d3d11_y410_pixel.hlsl).
[[vk::binding(0, 0)]] Texture2D<float4> videoTex;        // Y410: U, Y, V, A in R, G, B, A
[[vk::binding(1, 0)]] SamplerState theSampler;

struct CSC_CONST_BUF {
    column_major float3x3 cscMatrix;   // YUV -> RGB (row-vector convention), range-scaled
    float3 offsets;                    // YUV offsets, normalized
};
[[vk::binding(2, 0)]] ConstantBuffer<CSC_CONST_BUF> csc;

struct ShaderInput { float4 pos : SV_Position; float2 tex : TEXCOORD0; };

[shader("vertex")]
ShaderInput vertMain(float2 vpos : POSITION, float2 vtex : TEXCOORD0) {
    ShaderInput o;
    o.pos = float4(vpos, 0.0, 1.0);
    o.tex = vtex;
    return o;
}

float3 swizzle(float3 input) {
    // Y410 SRVs are in UYVA order
    return input.grb;
}

[shader("fragment")]
float4 fragMain(ShaderInput input) : SV_Target {
    float3 yuv = swizzle(videoTex.Sample(theSampler, input.tex).rgb);
    // Subtract the YUV offset for limited vs full range
    yuv -= csc.offsets;
    // Multiply by the conversion matrix for this colorspace
    yuv = mul(yuv, csc.cscMatrix);
    return float4(yuv, 1.0);
}
