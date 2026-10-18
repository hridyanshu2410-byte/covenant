import covenant;
struct ViewUniforms {
    [Maps("world","view")] float4x4 view;
    [Maps("view","clip")] [Convention("depth=reverse_z")] float4x4 projection;           // reversed depth
    [Maps("clip","view")] [Convention("depth=reverse_z")] float4x4 projectionInverse;
    uint depthZeroToOne;
    uint3 pad;
};
[[vk::binding(0, 0)]] ConstantBuffer<ViewUniforms> camera;
[[vk::binding(1, 0)]] Sampler2D depthTexture;

// scene: a ground quad
[shader("vertex")]
float4 vsScene(uint vid : SV_VertexID) : SV_Position {
    const float2 q[6] = { float2(-1, -1), float2(1, -1), float2(1, 1), float2(-1, -1), float2(1, 1), float2(-1, 1) };
    [Frame("world")] float3 w = float3(q[vid].x * 5, 0, q[vid].y * 5);
    return mul(camera.projection, mul(camera.view, float4(w, 1)));
}
[shader("fragment")]
void fsDepth() {}

// getViewPosition( uv, depth, projectionMatrixInverse )
float3 getViewPosition(float2 uv, float depth) {
    float clipZ = camera.depthZeroToOne != 0 ? depth : depth * 2.0 - 1.0;
    float4 clip = float4(uv * 2.0 - 1.0, clipZ, 1.0);
    float4 v = mul(camera.projectionInverse, clip);
    return v.xyz / v.w;
}

struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsFullscreen(uint vid : SV_VertexID) {
    VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
    o.uv = p; o.pos = float4(p * 2 - 1, 0, 1); return o;
}
// history rejection weight from the reconstructed view distance
[shader("fragment")]
float4 fsTraa(VOut i) : SV_Target {
    float d = depthTexture.Sample(i.uv).r;
    float3 viewPos = getViewPosition(i.uv, d);
    float dist = length(viewPos);
    float w = saturate(dist / 20.0);
    return float4(w, w, w, 1);
}
