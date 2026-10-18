// Distilled from godotengine/godot PR #123656 ("Fix SDFGI render pass
// material feedback binding", merged 2026-09-21; issue #123626). The
// material feedback buffer added to the Forward+ scene shader is declared
// as layout(set = 1, binding = 38, std430) buffer MaterialFeedbackBuffer.
// RenderForwardClustered::_setup_sdfgi_render_pass_uniform_set builds the
// set-1 uniforms for the SDFGI voxelisation pass by hand and provided the
// feedback buffer with u.binding = 37. Enabling SDFGI then failed with
// "Binding (38), set (1) was not provided" when the uniform set was
// created against the MODE_RENDER_SDF shader, and the SDF pass was not
// drawn. The fix sets u.binding = 38.
// Port: the RenderingDevice uniform set becomes a descriptor set layout +
// set built from the same list; the four SDF storage images (24-27) are
// reduced to one R32_UINT facet image at 24; 128x128 cascade slice.
#include "../../common/app.hpp"
using namespace bench;

struct Uniform { uint32_t binding; VkDescriptorType type; Buffer* buf; Image* img; };

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");

  Buffer scene = make_buffer(a, 128, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "scene_data_block");
  float ident[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  memcpy(scene.map, ident, 64); memcpy((uint8_t*)scene.map + 64, ident, 64);
  Buffer impl = make_buffer(a, 32, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "sdfgi_implementation_data");
  uint32_t idata[8] = {0, 0, 0, 0, 128, 128, 128, 0}; memcpy(impl.map, idata, sizeof idata);
  Buffer inst = make_buffer(a, 2 * 80, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, "instance_data");
  for (uint32_t i = 0; i < 2; i++) {
    float t[16] = {0.5f, 0, 0, 0, 0, 0.5f, 0, 0, 0, 0, 1, 0, i ? 0.4f : -0.4f, 0, 0.5f, 1};
    memcpy((uint8_t*)inst.map + i * 80, t, 64);
    uint32_t mid = 3 + i; memcpy((uint8_t*)inst.map + i * 80 + 64, &mid, 4);
  }
  Buffer feedback = make_buffer(a, 256 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true,
                                "material_feedback_buffer");
  memset(feedback.map, 0, 256 * sizeof(uint32_t));
  Image facets = make_image(a, 128, 128, VK_FORMAT_R32_UINT, VK_IMAGE_USAGE_STORAGE_BIT, "sdfgi_geom_facets");
  Buffer vb = make_buffer(a, 3 * 3 * sizeof(float), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "mesh_vb");
  float tri[9] = {-1, -1, 0, 1, -1, 0, 0, 1, 0}; memcpy(vb.map, tri, sizeof tri);

  // _setup_sdfgi_render_pass_uniform_set
  std::vector<Uniform> uniforms;
  uniforms.push_back({0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &scene, nullptr});
  uniforms.push_back({1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &impl, nullptr});
  uniforms.push_back({2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &inst, nullptr});
  uniforms.push_back({24, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, nullptr, &facets});
  {
    Uniform u{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &feedback, nullptr};
#ifdef FIXED
    u.binding = 38;
#else
    u.binding = 37;
#endif
    uniforms.push_back(u);
  }
  std::vector<Binding> bl;
  for (auto& u : uniforms) bl.push_back({u.binding, u.type, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT});
  auto set1l = make_set_layout(a, bl);
  auto set0l = make_set_layout(a, {});
  auto set1 = alloc_set(a, set1l, "sdfgi_render_pass_uniform_set");
  for (auto& u : uniforms) {
    if (u.buf) write_buffer(a, set1, u.binding, u.type, *u.buf);
    else write_image(a, set1, u.binding, u.type, u.img->view, VK_IMAGE_LAYOUT_GENERAL);
  }

  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {set0l, set1l};
  gd.vertex_stride = 12; gd.attributes = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}};
  Pipeline sdf = make_graphics(a, gd, "scene_shader_render_sdf");

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    image_barrier(c, facets.img, f == 0 ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    buffer_barrier(c, feedback.buf, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {128, 128}}; ri.layerCount = 1;
    vkCmdBeginRendering(c, &ri);
    VkViewport vp{0, 0, 128, 128, 0, 1}; VkRect2D sc{{0, 0}, {128, 128}};
    vkCmdSetViewport(c, 0, 1, &vp); vkCmdSetScissor(c, 0, 1, &sc);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, sdf.pipe);
    VkDeviceSize zero = 0;
    vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &zero);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, sdf.layout, 1, 1, &set1, 0, nullptr);
    vkCmdDraw(c, 3, 2, 0, 0);
    vkCmdEndRendering(c);
    begin_rendering(a, c, {a.swap_views[idx]});        // main view (cleared only here)
    vkCmdEndRendering(c);
  });
  return 0;
}
