import covenant;
// Godot Forward+ scene shader, MODE_RENDER_SDF (reduced): geometry is
// voxelised into the SDFGI cascade's storage images, and each fragment
// records that its material was used in the material feedback buffer.
struct SceneDataBlock {
    [Maps("world", "view")] float4x4 view_matrix;
    [Maps("view", "clip")] float4x4 projection_matrix;
};
struct ImplementationData {
    uint4 sdf_offset;          // xyz = cascade offset in cells
    uint4 sdf_size;
};
struct InstanceData {
    [Maps("object", "world")] float4x4 transform;
    uint material_id; uint3 pad;
};
[[vk::binding(0, 1)]] ConstantBuffer<SceneDataBlock> scene_data_block;
[[vk::binding(1, 1)]] ConstantBuffer<ImplementationData> implementation_data;
[[vk::binding(2, 1)]] StructuredBuffer<InstanceData> instances;
[[vk::binding(24, 1)]] [[vk::image_format("r32ui")]] RWTexture2D<uint> geom_facets;
[[vk::binding(38, 1)]] RWStructuredBuffer<uint> material_feedback;   // MaterialFeedbackBuffer

struct VIn { float3 pos : POSITION; };
struct VOut { float4 pos : SV_Position; nointerpolation uint inst : INSTANCE; };
[shader("vertex")]
VOut vsMain(VIn v, uint iid : SV_InstanceID) {
    VOut o;
    float4 w = mul(instances[iid].transform, float4(v.pos, 1));
    o.pos = mul(scene_data_block.projection_matrix, mul(scene_data_block.view_matrix, w));
    o.inst = iid; return o;
}
[shader("fragment")]
void fsMain(VOut i) {
    int2 cell = int2(i.pos.xy) + int2(implementation_data.sdf_offset.xy);
    if (all(cell >= 0) && all(cell < int2(implementation_data.sdf_size.xy)))
        InterlockedOr(geom_facets[cell], 1u << (i.inst & 31));
    material_feedback[instances[i.inst].material_id] = 1u;
}
