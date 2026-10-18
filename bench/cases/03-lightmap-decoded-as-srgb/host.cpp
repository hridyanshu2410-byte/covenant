// Distilled from godotengine/godot #53388 (lightmaps washed out): baked
// lighting stored in an 8-bit target was later read through an sRGB view, so
// linear data was "decoded" a second time.
#include "../../common/app.hpp"
using namespace bench;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
#ifdef FIXED
  VkFormat sample_view = VK_FORMAT_R8G8B8A8_UNORM;
#else
  VkFormat sample_view = VK_FORMAT_R8G8B8A8_SRGB;     // BUG: data written as linear
#endif
  Image lm = make_image(a, 256, 256, VK_FORMAT_R8G8B8A8_UNORM,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, "lightmap");
  // second view of the same image for sampling (mutable format)
  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vi.image = lm.img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = sample_view;
  vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkImageView lm_sample; VK(vkCreateImageView(a.dev, &vi, nullptr, &lm_sample));
  a.name(lm_sample, VK_OBJECT_TYPE_IMAGE_VIEW, "lightmap");
  Image alb = make_image(a, 256, 256, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_USAGE_SAMPLED_BIT, "albedo");
  Buffer sun = make_buffer(a, 32, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "sun");
  float s[8] = {0, -1, 0, 0, 1, 1, 0.9f, 0};
  memcpy(sun.map, s, sizeof s);

  auto bake_l = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto shade_l = make_set_layout(a, {{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
                                     {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto bake_s = alloc_set(a, bake_l, "bake_set");
  write_buffer(a, bake_s, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, sun);
  auto shade_s = alloc_set(a, shade_l, "shade_set");
  write_image(a, shade_s, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, lm_sample, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  write_image(a, shade_s, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, alb.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  auto sh = load_shader(a, "shaders.spv");
  GfxDesc bd; bd.vs = bd.fs = sh; bd.fs_entry = "bakeMain"; bd.sets = {bake_l}; bd.colors = {VK_FORMAT_R8G8B8A8_UNORM};
  Pipeline bake = make_graphics(a, bd, "bake");
  GfxDesc sd; sd.vs = sd.fs = sh; sd.fs_entry = "shadeMain"; sd.sets = {shade_l}; sd.colors = {a.swap_format};
  Pipeline shade = make_graphics(a, sd, "shade");

  run_frames(a, frames_from_env(), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    if (f == 0) {
      image_barrier(c, alb.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
      // bake once
      image_barrier(c, lm.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
      begin_rendering(a, c, {lm.view});
      vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, bake.pipe);
      vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, bake.layout, 0, 1, &bake_s, 0, nullptr);
      vkCmdDraw(c, 3, 1, 0, 0);
      vkCmdEndRendering(c);
      image_barrier(c, lm.img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, shade.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, shade.layout, 0, 1, &shade_s, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
