import covenant;
// PixiJS WebGPU mesh program (reduced): global uniforms in group 0, each
// mesh's local uniform group in group 1, served from the shared uniform batch
// buffer at the mesh's offset.
struct GlobalUniforms {
    [Maps("world", "clip")] column_major float3x3 uProjectionMatrix;
    column_major float3x3 uWorldTransformMatrix;
    float4 uWorldColorAlpha;
    float2 uResolution;
};
[[vk::binding(0, 0)]] ConstantBuffer<GlobalUniforms> globalUniforms;

struct LocalUniforms {
    [Maps("object", "world")] column_major float3x3 uTransformMatrix;
    [Color("linear")] float4 uColor;
    float uRound;
};
[[vk::binding(0, 1)]] ConstantBuffer<LocalUniforms> localUniforms;

struct MeshIn { [Frame("object")] float2 aPosition : POSITION; };
struct VSOut { float4 position : SV_Position; };

[shader("vertex")]
VSOut mainVertex(MeshIn i) {
    float3x3 mvp = mul(globalUniforms.uProjectionMatrix, mul(globalUniforms.uWorldTransformMatrix, localUniforms.uTransformMatrix));
    float2 p = mul(mvp, float3(i.aPosition, 1.0)).xy;
    if (localUniforms.uRound == 1.0) p = round(p * globalUniforms.uResolution * 0.5) / (globalUniforms.uResolution * 0.5);
    VSOut o;
    o.position = float4(p, 0.0, 1.0);
    return o;
}

[shader("fragment")]
float4 mainFragment(VSOut i) : SV_Target {
    return localUniforms.uColor * globalUniforms.uWorldColorAlpha;
}
