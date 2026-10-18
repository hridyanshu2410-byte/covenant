import covenant;
// iced custom_shader example, cubes.wgsl reduced: instanced cubes, lit in
// world space against a fixed point light (the normal map and the skybox
// reflection are dropped).
struct Uniforms {
    [Maps("world", "clip")] column_major float4x4 projection;
    [Frame("world")] float4 camera_pos;
    [Color("linear")] float4 light_color;
};
[[vk::binding(0, 0)]] ConstantBuffer<Uniforms> uniforms;

static const float3 LIGHT_POS = float3(0.0, 3.0, 3.0);
static const float4 CUBE_BASE_COLOR = float4(0.294118, 0.462745, 0.611765, 0.6);

struct Vertex {
    [Frame("object")] float3 position : POSITION;
    [Frame("object")] float3 normal : NORMAL;
};
struct Cube {
    [Maps("object", "world")] float4 matrix_0 : MATRIX0;
    float4 matrix_1 : MATRIX1;
    float4 matrix_2 : MATRIX2;
    float4 matrix_3 : MATRIX3;
};
struct Output {
    float4 clip_pos : SV_Position;
    float3 world_pos : WORLDPOS;
    float3 world_normal : NORMAL;
};

[shader("vertex")]
Output vs_main(Vertex vertex, Cube cube) {
    float4x4 cube_matrix = transpose(float4x4(cube.matrix_0, cube.matrix_1, cube.matrix_2, cube.matrix_3));
    float4 world_pos = mul(cube_matrix, float4(vertex.position, 1.0));
    Output o;
    o.clip_pos = mul(uniforms.projection, world_pos);
    o.world_pos = world_pos.xyz;
    o.world_normal = normalize(mul((float3x3)cube_matrix, vertex.normal));
    return o;
}

[shader("fragment")]
float4 fs_main(Output i) : SV_Target {
    float3 n = normalize(i.world_normal);
    float3 to_light = normalize(LIGHT_POS - i.world_pos);
    float3 to_camera = normalize(uniforms.camera_pos.xyz - i.world_pos);
    float diffuse = max(dot(n, to_light), 0.0);
    float specular = pow(max(dot(reflect(-to_light, n), to_camera), 0.0), 16.0) * 0.8;
    float3 color = CUBE_BASE_COLOR.rgb * (0.2 + diffuse) * uniforms.light_color.rgb + specular * uniforms.light_color.rgb;
    return float4(color, 1.0);
}
