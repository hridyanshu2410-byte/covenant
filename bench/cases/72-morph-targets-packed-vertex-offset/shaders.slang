import covenant;
// bevy_pbr mesh.wgsl with MORPH_TARGETS: morph_vertex() adds the weighted
// target displacements for this vertex, read from the morph target data by
// vertex index. (bevy stores the targets in a texture_3d; a storage buffer
// with the same vertex-major indexing stands in for it.)
struct Mesh {
    [Maps("object","world")] float4x4 world_from_local;
    uint flags;
#ifdef FIXED
    uint first_vertex_index;
    uint pad_a;
    uint pad_b;
#else
    uint pad_a;
    uint pad_b;
    uint pad_c;
#endif
};
[[vk::binding(0, 1)]] StructuredBuffer<Mesh> mesh;

struct View { [Maps("world","clip")] float4x4 clip_from_world; };
[[vk::binding(0, 0)]] ConstantBuffer<View> view;

struct MorphWeights { float4 weights[4]; uint layer_count; uint vertex_count; uint2 pad; };
[[vk::binding(1, 1)]] ConstantBuffer<MorphWeights> morph_weights;
[[vk::binding(2, 1)]] StructuredBuffer<float> morph_targets;    // [layer][vertex][xyz] position offsets

float weight_at(uint i) { return morph_weights.weights[i / 4][i % 4]; }
float3 morph(uint vertex_index, uint weight_index) {
    uint base = (weight_index * morph_weights.vertex_count + vertex_index) * 3;
    return float3(morph_targets[base], morph_targets[base + 1], morph_targets[base + 2]);
}

struct Vertex { uint instance_index : SV_InstanceID; uint index : SV_VertexID; [Frame("object")] float3 position : POSITION; };
struct VOut { float4 pos : SV_Position; float3 color : COLOR; };

[shader("vertex")]
VOut vsMain(Vertex vertex_in) {
    float3 position = vertex_in.position;
#ifdef FIXED
    let first_vertex = mesh[vertex_in.instance_index].first_vertex_index;
    let vertex_index = vertex_in.index - first_vertex;
#else
    let vertex_index = vertex_in.index;
#endif
    for (uint i = 0; i < morph_weights.layer_count; i++) {
        float weight = weight_at(i);
        if (weight == 0.0) continue;
        position += weight * morph(vertex_index, i);
    }
    VOut o;
    float4 world = mul(mesh[vertex_in.instance_index].world_from_local, float4(position, 1));
    o.pos = mul(view.clip_from_world, world);
    o.color = float3(0.9, 0.6, 0.3);
    return o;
}
[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return float4(i.color, 1); }
