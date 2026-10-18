import covenant;
// VTK WebGPU polydata shading (vtkWebGPUActor / SceneLight.wgsl), reduced to
// per-fragment Blinn-Phong over the scene lights in view coordinates.
struct SceneTransform {
    [Maps("world", "view")] column_major float4x4 view;
    [Maps("view", "clip")]  column_major float4x4 projection;
};
struct ActorTransform {
    [Maps("object", "world")] column_major float4x4 world;
    [Maps("object", "view")]  column_major float3x3 normal;   // padded columns
};
struct SceneLight {
    uint light_type;     // 0 deferred, 1 headlight, 2 lightkit, 3 positional
    uint positional;
    float cone_angle;
    float exponent;
    [Color("linear")] float3 color;
    [Frame("view")] float3 direction_vc;
    [Frame("view")] float3 position_vc;
    float3 attenuation;
};
struct SceneLights {
    uint count;
    SceneLight values[8];
};
[[vk::binding(0, 0)]] ConstantBuffer<SceneTransform> scene_transform;
[[vk::binding(1, 0)]] StructuredBuffer<SceneLights> scene_lights;
[[vk::binding(2, 0)]] ConstantBuffer<ActorTransform> actor;

struct VSIn { [Frame("object")] float3 position : POSITION; [Frame("object")] float3 normal : NORMAL; };
struct VSOut { float4 pos : SV_Position; float3 position_vc : POSITION_VC; float3 normal_vc : NORMAL_VC; };

[shader("vertex")]
VSOut vertexMain(VSIn i) {
    VSOut o;
    float4 vc = mul(scene_transform.view, mul(actor.world, float4(i.position, 1.0)));
    o.pos = mul(scene_transform.projection, vc);
    o.position_vc = vc.xyz;
    o.normal_vc = normalize(mul(actor.normal, i.normal));
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOut i) : SV_Target {
    const float3 ambient = float3(0.05, 0.05, 0.05);
    const float3 diffuse_color = float3(0.8, 0.8, 0.8);
    const float specular_power = 30.0;
    float3 n = normalize(i.normal_vc);
    float3 v = normalize(-i.position_vc);
    float3 rgb = ambient;
    uint count = scene_lights[0].count;
    for (uint li = 0; li < count; li++) {
        SceneLight light = scene_lights[0].values[li];
        float3 l;
        float att = 1.0;
        if (light.positional == 1) {
            float3 d = light.position_vc - i.position_vc;
            float dist = length(d);
            l = d / dist;
            att = 1.0 / (light.attenuation.x + light.attenuation.y * dist + light.attenuation.z * dist * dist);
        } else {
            l = -light.direction_vc;
        }
        float df = max(dot(n, l), 0.0);
        float sf = pow(max(dot(n, normalize(l + v)), 0.0), specular_power);
        rgb += att * light.color * (diffuse_color * df + sf * 0.3);
    }
    return float4(rgb, 1.0);
}
