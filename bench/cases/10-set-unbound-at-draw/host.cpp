// Distilled from themuffinator/FnQ3 PR #19: a draw path for surfaces without
// a lightmap skipped binding descriptor set 1, but the pipeline still reads
// it ("descriptor set unbound at draw").
#include "../../common/app.hpp"
using namespace bench;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  Buffer cam = make_buffer(a, 64, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "camera");
  Image lm = make_image(a, 64, 64, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT, "lightmap");
  Image white = make_image(a, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT, "white_lightmap");
  auto l0 = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}});
  auto l1 = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto s0 = alloc_set(a, l0, "view_set");
  write_buffer(a, s0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, cam);
  auto s1 = alloc_set(a, l1, "lightmap_set");
  write_image(a, s1, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, lm.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  auto sw = alloc_set(a, l1, "white_lightmap_set");
  write_image(a, sw, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, white.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {l0, l1}; gd.colors = {VK_FORMAT_B8G8R8A8_SRGB};
  Pipeline pipe = make_graphics(a, gd, "world_surface");
  struct Surface { bool has_lightmap; } surfaces[] = {{false}, {true}};
  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    if (f == 0) {
      for (VkImage im : {lm.img, white.img})
        image_barrier(c, im, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout, 0, 1, &s0, 0, nullptr);
    for (auto& s : surfaces) {
      if (s.has_lightmap)
        vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout, 1, 1, &s1, 0, nullptr);
#ifdef FIXED
      else
        vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout, 1, 1, &sw, 0, nullptr);
#endif
      vkCmdDraw(c, 3, 1, 0, 0);
    }
    vkCmdEndRendering(c);
  });
}
