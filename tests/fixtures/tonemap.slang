import covenant;
[Color("linear")] Sampler2D hdr;
float3 aces(float3 x) { return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)); }
[shader("fragment")]
float4 fsMain(float2 uv : TEXCOORD) : SV_Target {
    float3 c = aces(hdr.Sample(uv).rgb);
#ifdef MANUAL_ENCODE
    c = pow(c, 1.0 / 2.2);
#endif
    return float4(c, 1);
}
