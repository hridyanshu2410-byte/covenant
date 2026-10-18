import covenant;
// One push-constant block shared by both stages: the vertex stage reads the
// matrix (bytes 0-63), the fragment stage reads one float at byte 64.
// (The issue gives only offsets, sizes and stages; field names are ours.)
struct PushConstants {
    [Maps("world","clip")] float4x4 viewProj;
    float fragmentValue;
};
[[vk::push_constant]] PushConstants pc;

struct VIn { [Frame("world")] float3 position : POSITION; float2 uv : TEXCOORD; };
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD; };
[shader("vertex")]
VOut vsWorld(VIn v) { VOut o; o.pos = mul(pc.viewProj, float4(v.position, 1)); o.uv = v.uv; return o; }

[[vk::binding(0, 0)]] [Color("srgb")] Sampler2D surfaceTexture;
[shader("fragment")]
float4 fsWorld(VOut i) : SV_Target {
    float4 c = surfaceTexture.Sample(i.uv);
    return float4(c.rgb * pc.fragmentValue, 1);
}
