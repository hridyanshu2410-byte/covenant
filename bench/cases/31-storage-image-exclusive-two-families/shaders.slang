import covenant;
struct ComputeParams { float time; float strength; float2 pad; };
[[vk::binding(0, 0)]] ConstantBuffer<ComputeParams> params;
[[vk::binding(1, 0)]] [format("rgba8")] RWTexture2D<float4> resultImage;

// edgedetect over a procedural pattern
float pattern(int2 p) { return frac(sin(dot(float2(p), float2(12.9898, 78.233)) + params.time) * 43758.5453); }
[shader("compute")] [numthreads(16, 16, 1)]
void csMain(uint3 id : SV_DispatchThreadID) {
    int2 p = int2(id.xy);
    float c = pattern(p);
    float e = 8.0 * c - pattern(p + int2(1, 0)) - pattern(p - int2(1, 0)) - pattern(p + int2(0, 1))
            - pattern(p - int2(0, 1)) - pattern(p + int2(1, 1)) - pattern(p - int2(1, 1))
            - pattern(p + int2(1, -1)) - pattern(p - int2(1, -1));
    float v = saturate(0.5 + e * params.strength / 8.0);
    resultImage[id.xy] = float4(v, v, v, 1);
}

[[vk::binding(0, 0)]] Sampler2D samplerColor;
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return samplerColor.Sample(i.uv); }
