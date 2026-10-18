// Distilled from processing/processing #2904: vertex attributes were sent
// in view space while the shader treats them as model space, so lighting
// moved with the camera.
#include "../../common/app.hpp"
#include "mesh.hpp"
using namespace bench;

float3 object_normal(int i) { return {0, 0, 1}; }
float3 view_normal(int i) { return {0, 0.2f, 0.98f}; }
float3 object_position(int i) { return {float(i & 1), float(i >> 1), 0}; }

std::vector<Vertex> build_mesh() {
  std::vector<Vertex> out;
  for (int i = 0; i < 3; i++) {
    Vertex v;
    v.position = object_position(i);
#ifdef FIXED
    v.normal = object_normal(i);
#else
    v.normal = view_normal(i);                        // BUG: view-space normals
#endif
    out.push_back(v);
  }
  return out;
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto mesh = build_mesh();
  Buffer vb = make_buffer(a, sizeof(Vertex) * mesh.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "mesh_vertices");
  memcpy(vb.map, mesh.data(), sizeof(Vertex) * mesh.size());
  Buffer cam = make_buffer(a, 128, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "camera");
  Buffer sun = make_buffer(a, 32, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "sun");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "scene_set");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, cam);
  write_buffer(a, set, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, sun);
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  gd.vertex_stride = sizeof(Vertex);
  gd.attributes = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12}};
  Pipeline pipe = make_graphics(a, gd, "lit");
  run_frames(a, frames_from_env(), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout, 0, 1, &set, 0, nullptr);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &off);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
