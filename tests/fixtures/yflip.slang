// Fixture for the clip-space Y convention rule.
import covenant;
struct Camera { [Convention("clip_y=down")] float4x4 projVk; float4x4 projGl; };
[[vk::binding(0, 0)]] ConstantBuffer<Camera> camera;
[shader("vertex")]
float4 vsPlain(float3 p : POSITION) : SV_Position { return float4(p, 1); }
[shader("vertex")]
float4 vsNegate(float3 p : POSITION) : SV_Position {
    float4 o = float4(p, 1);
    o.y = -o.y;
    return o;
}
[shader("vertex")]
float4 vsScale(float3 p : POSITION) : SV_Position { return float4(p, 1) * float4(1, -1, 1, 1); }
[shader("vertex")]
float4 vsProjVk(float3 p : POSITION) : SV_Position { return mul(camera.projVk, float4(p, 1)); }
[shader("vertex")]
float4 vsProjGl(float3 p : POSITION) : SV_Position { return mul(camera.projGl, float4(p, 1)); }
[shader("vertex")]
float4 vsProjVkNegate(float3 p : POSITION) : SV_Position {
    float4 o = mul(camera.projVk, float4(p, 1));
    o.y = -o.y;
    return o;
}
[shader("vertex")]
float4 vsConstFlip(float3 p : POSITION) : SV_Position {
    const float4x4 flip = float4x4(1, 0, 0, 0,  0, -1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1);
    return mul(flip, float4(p, 1));
}
