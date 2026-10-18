// Distilled from bevyengine/bevy PR #14375 (fixes #14366): mesh vertex
// buffer packing put several meshes into one vertex buffer, and draws now
// start at the mesh's slice (vertex_buffer_slice.range, i.e. firstVertex =
// the mesh's offset in the shared buffer). The vertex index the shader sees
// includes that offset, but morph_vertex() still used it directly to look up
// the mesh's morph target displacements, which are stored per mesh starting
// at vertex 0: every mesh that was not first in the buffer read another
// vertex's (or another mesh's) displacements. The fix adds
// first_vertex_index to MeshUniform / MeshInputUniform (plus padding) and
// subtracts it in morph_vertex(): vertex_index = vertex.index - first_vertex.
// (Morph targets in a storage buffer instead of bevy's texture_3d; one view
// uniform and two meshes.)
#include "../../common/app.hpp"
using namespace bench;

struct MeshUniform {
  float world_from_local[16];
  uint32_t flags;
#ifdef FIXED
  uint32_t first_vertex_index;
  uint32_t pad_a, pad_b;
#else
  uint32_t pad_a, pad_b, pad_c;
#endif
};

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  // the shared (packed) vertex buffer: mesh 0 "ground" (6 vertices), mesh 1 "morph_face" (6 vertices)
  const uint32_t ground_first = 0, ground_count = 6, face_first = 6, face_count = 6;
  float verts[12 * 3] = {-1, 0.8f, 0, 1, 0.8f, 0, -1, 1, 0, 1, 0.8f, 0, 1, 1, 0, -1, 1, 0,
                         -0.5f, -0.5f, 0, 0.5f, -0.5f, 0, -0.5f, 0.5f, 0, 0.5f, -0.5f, 0, 0.5f, 0.5f, 0, -0.5f, 0.5f, 0};
  Buffer vb = make_buffer(a, sizeof verts, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "mesh_allocator_vertex_slab");
  memcpy(vb.map, verts, sizeof verts);

  // per-instance MeshUniforms
  Buffer meshes = make_buffer(a, 2 * sizeof(MeshUniform), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, "mesh_uniforms");
  MeshUniform mu[2]{};
  for (int i = 0; i < 2; i++) for (int k = 0; k < 4; k++) mu[i].world_from_local[k * 5] = 1.0f;
#ifdef FIXED
  mu[0].first_vertex_index = ground_first; mu[1].first_vertex_index = face_first;
#endif
  memcpy(meshes.map, mu, sizeof mu);
  // morph_face: 2 targets x 6 vertices x xyz, indexed from the mesh's own vertex 0
  const uint32_t layers = 2;
  Buffer targets = make_buffer(a, layers * face_count * 3 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, "morph_face.morph_targets");
  float* t = (float*)targets.map;
  for (uint32_t l = 0; l < layers; l++) for (uint32_t v = 0; v < face_count; v++) {
    t[(l * face_count + v) * 3 + 0] = 0; t[(l * face_count + v) * 3 + 1] = l == 0 ? 0.2f * (v % 3) : 0; t[(l * face_count + v) * 3 + 2] = 0;
  }
  struct { float weights[16]; uint32_t layer_count, vertex_count, pad[2]; } w = {{0.75f, 0.25f}, layers, face_count, {0, 0}};
  Buffer weights = make_buffer(a, sizeof w, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "morph_face.morph_weights");
  memcpy(weights.map, &w, sizeof w);
  Buffer viewb = make_buffer(a, 64, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "view_uniforms");
  float id[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0.5f, 1};
  memcpy(viewb.map, id, sizeof id);

  auto viewL = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}});
  auto meshL = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT},
                                   {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT},
                                   {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}});
  auto viewSet = alloc_set(a, viewL, "mesh_view_bind_group");
  write_buffer(a, viewSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, viewb);
  auto meshSet = alloc_set(a, meshL, "mesh_bind_group (morphed)");
  write_buffer(a, meshSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, meshes);
  write_buffer(a, meshSet, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, weights);
  write_buffer(a, meshSet, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, targets);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {viewL, meshL}; gd.colors = {a.swap_format};
  gd.vertex_stride = 12; gd.attributes = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}};
  Pipeline pbr = make_graphics(a, gd, "mesh_pipeline (MORPH_TARGETS)");

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pbr.pipe);
    VkDescriptorSet sets[2] = {viewSet, meshSet};
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pbr.layout, 0, 2, sets, 0, nullptr);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &off);
    // DrawMesh: draw(vertex_buffer_slice.range, batch_range)
    (void)ground_first; (void)ground_count;
    vkCmdDraw(c, face_count, 1, face_first, 1);          // morph_face: instance 1, vertices 6..11
    vkCmdEndRendering(c);
  });
}
