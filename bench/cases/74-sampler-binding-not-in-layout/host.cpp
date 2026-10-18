// Distilled from timi-liuliang/echo #1149 (reported; no fix linked). Creating
// the graphics pipeline for a textured material fails
// VUID-VkGraphicsPipelineCreateInfo-layout-00756: "Shader uses descriptor
// slot 0.1 (expected VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) but not
// declared in pipeline layout". The issue gives only the validation message;
// the pipeline layout's set 0 lacks the combined image sampler at binding 1
// that the shader statically uses. The fixed variant declares (and writes)
// binding 1 in set 0, as the VUID requires. The material's uniform block at
// binding 0 and the texture itself are ours.
#include "../../common/app.hpp"
using namespace bench;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
#ifdef FIXED
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
#else
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT}});
#endif
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format}; gd.vertex_stride = 20;
  gd.attributes = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32_SFLOAT, 12}};
  Pipeline p = make_graphics(a, gd, "echo_material_pipeline");

  Image tex = make_image(a, 1, 1, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, "base_color_texture");
  VkCommandBuffer up = alloc_cmd(a, a.gfx_pool);
  begin(up);
  image_barrier(up, tex.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
  VkClearColorValue white{{1, 1, 1, 1}}; VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdClearColorImage(up, tex.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &white, 1, &rng);
  image_barrier(up, tex.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
  VK(vkEndCommandBuffer(up));
  submit(a.gfx, up, {}, {}, VK_NULL_HANDLE);
  VK(vkQueueWaitIdle(a.gfx));

  Buffer ubo = make_buffer(a, 80, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "material_uniforms");
  float u[20] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  memcpy(ubo.map, u, sizeof u);
  auto set = alloc_set(a, setl, "material_descriptor_set");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo);
#ifdef FIXED
  write_image(a, set, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, tex.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
#endif
  float verts[] = {-0.5f, 0.5f, 0, 0, 1, 0.5f, 0.5f, 0, 1, 1, 0, -0.5f, 0, 0.5f, 0};
  Buffer vb = make_buffer(a, sizeof verts, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "mesh_vertices");
  memcpy(vb.map, verts, sizeof verts);

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, p.layout, 0, 1, &set, 0, nullptr);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &off);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
