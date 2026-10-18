import covenant;
// Flax Engine material shader (MaterialCommon.hlsl ViewData + a generated
// surface material that pans an emissive pattern with time and adds a
// view-dependent rim), reduced to one pass.
struct ViewData {
    [Maps("world", "view")] column_major float4x4 ViewMatrix;
    [Maps("world", "clip")] column_major float4x4 ViewProjectionMatrix;
    [Maps("world", "clip")] column_major float4x4 PrevViewProjectionMatrix;
    [Maps("world", "clip")] column_major float4x4 MainViewProjectionMatrix;
    float4 MainScreenSize;
    [Frame("world")] float3 ViewPos;
    float ViewFar;
    [Frame("world")] float3 ViewDir;
#ifdef FIXED
    float TimeParam;
#else
    float UnscaledTimeParam;
    float ScaledTimeParam;
#endif
    float4 ViewInfo;
    float4 ScreenSize;
    float4 TemporalAAJitter;
    float3 LargeWorldsChunkIndex;
    float LargeWorldsChunkSize;
#ifdef FIXED
    float3 ViewPadding0;
    float UnscaledTimeParam;
#endif
};
[[vk::binding(1, 0)]] ConstantBuffer<ViewData> viewData;

struct DrawData {
    [Maps("object", "world")] column_major float4x4 WorldMatrix;
};
[[vk::binding(2, 0)]] ConstantBuffer<DrawData> drawData;

struct ModelInput {
    [Frame("object")] float3 Position : POSITION;
    float2 TexCoord : TEXCOORD0;
    [Frame("object")] float3 Normal : NORMAL;
};
struct VertexOutput {
    float4 Position : SV_Position;
    float3 WorldPosition : TEXCOORD0;
    float2 TexCoord : TEXCOORD1;
    float3 WorldNormal : TEXCOORD2;
};

[shader("vertex")]
VertexOutput VS(ModelInput input) {
    VertexOutput o;
    float4 world = mul(drawData.WorldMatrix, float4(input.Position, 1.0));
    o.WorldPosition = world.xyz;
    o.Position = mul(viewData.ViewProjectionMatrix, world);
    o.TexCoord = input.TexCoord;
    o.WorldNormal = normalize(mul((float3x3)drawData.WorldMatrix, input.Normal));
    return o;
}

[shader("fragment")]
float4 PS_GBuffer(VertexOutput input) : SV_Target {
    // Material graph: Time -> Panner(UV, speed 0.25) -> stripes; rim = 1 - dot(N, V)
#ifdef FIXED
    float2 uv = input.TexCoord + float2(viewData.TimeParam * 0.25, 0.0);
#else
    float2 uv = input.TexCoord + float2(viewData.ScaledTimeParam * 0.25, 0.0);
#endif
    float stripes = step(0.5, frac(uv.x * 8.0));
    float3 V = normalize(viewData.ViewPos - input.WorldPosition);
    float rim = pow(1.0 - saturate(dot(normalize(input.WorldNormal), V)), 3.0);
    // screen-space vignette from ScreenSize (xy = size, zw = 1/size)
    float2 screenUV = input.Position.xy * viewData.ScreenSize.zw;
    float vignette = 1.0 - 0.5 * dot(screenUV - 0.5, screenUV - 0.5);
    float3 color = (float3(0.1, 0.3, 0.8) * stripes + float3(1.0, 0.6, 0.2) * rim) * vignette;
    return float4(color, 1.0);
}
