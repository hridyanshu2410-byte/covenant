import covenant;
// BT.709 limited-range YCbCr -> RGB with 8-bit levels (16/255, 128/255).
// Correct for an 8-bit view; on a 16-bit UNORM view black is 4096/65535.
[[vk::binding(0, 0)]] Sampler2D yuv;
[shader("fragment")]
float4 fsMain(float2 uv : TEXCOORD) : SV_Target {
    float3 s = yuv.Sample(uv).rgb;
    float y = (s.x - 16.0 / 255.0) * (255.0 / 219.0);
    float cb = (s.y - 128.0 / 255.0) * (255.0 / 224.0);
    float cr = (s.z - 128.0 / 255.0) * (255.0 / 224.0);
    return float4(y + 1.5748 * cr, y - 0.1873 * cb - 0.4681 * cr, y + 1.8556 * cb, 1.0);
}
