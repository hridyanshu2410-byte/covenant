import covenant;
// PixiJS mesh shader (WGSL default mesh program, reduced) plus the sprite
// that later shows the render texture on screen.
struct GlobalUniforms {
    [Maps("world", "clip")] column_major float3x3 uProjectionMatrix;   // includes the flipY of the bound target
    column_major float3x3 uWorldTransformMatrix;
    float4 uWorldColorAlpha;
    float2 uResolution;
};
[[vk::binding(0, 0)]] ConstantBuffer<GlobalUniforms> globalUniforms;

struct LocalUniforms {
    [Maps("object", "world")] column_major float3x3 uTransformMatrix;
    [Color("linear")] float4 uColor;
};
[[vk::binding(1, 0)]] ConstantBuffer<LocalUniforms> localUniforms;

struct VSOut { float4 position : SV_Position; float2 vUV : TEXCOORD0; };

struct MeshIn { [Frame("object")] float2 aPosition : POSITION; float2 aUV : TEXCOORD0; };

[shader("vertex")]
VSOut meshVertex(MeshIn i) {
    float3x3 mvp = mul(globalUniforms.uProjectionMatrix, mul(globalUniforms.uWorldTransformMatrix, localUniforms.uTransformMatrix));
    VSOut o;
    o.position = float4(mul(mvp, float3(i.aPosition, 1.0)).xy, 0.0, 1.0);
    o.vUV = i.aUV;
    return o;
}

[shader("fragment")]
float4 meshFragment(VSOut i) : SV_Target {
    return localUniforms.uColor * globalUniforms.uWorldColorAlpha;
}

// screen pass: full-screen triangle sampling the render texture
[[vk::binding(2, 0)]] [Color("linear")] Sampler2D uTexture;
struct BlitOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
BlitOut blitVertex(uint vid : SV_VertexID) {
    BlitOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
[shader("fragment")]
float4 blitFragment(BlitOut b) : SV_Target { return uTexture.Sample(b.uv); }
