// Distilled from drsnuggles8/OloEngineBase #1512 (fix PR #1515): the
// transparent pass drew with the bindless material heap never bound.
#include "../../common/app.hpp"
using namespace bench;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  Buffer info = make_buffer(a, 16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "draw_info");
  std::vector<Image> tex;
  for (int i = 0; i < 4; i++)
    tex.push_back(make_image(a, 64, 64, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_USAGE_SAMPLED_BIT, "material_tex[" + std::to_string(i) + "]"));
  auto l0 = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto l1 = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 4}});
  auto s0 = alloc_set(a, l0, "draw_set");
  write_buffer(a, s0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, info);
  auto heap = alloc_set(a, l1, "material_heap");
  for (int i = 0; i < 4; i++) {
    VkDescriptorImageInfo ii{a.sampler, tex[i].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = heap; w.dstBinding = 0; w.dstArrayElement = i; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &ii;
    vkUpdateDescriptorSets(a.dev, 1, &w, 0, nullptr);
  }
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {l0, l1}; gd.colors = {a.swap_format};
  Pipeline opaque = make_graphics(a, gd, "opaque");
  Pipeline transparent = make_graphics(a, gd, "transparent");
  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    if (f == 0)
      for (auto& t : tex)
        image_barrier(c, t.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    begin_rendering(a, c, {a.swap_views[idx]});
    // transparent pass first in this engine's sort order
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, transparent.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, transparent.layout, 0, 1, &s0, 0, nullptr);
#ifdef FIXED
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, transparent.layout, 1, 1, &heap, 0, nullptr);
#endif
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, opaque.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, opaque.layout, 0, 1, &s0, 0, nullptr);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, opaque.layout, 1, 1, &heap, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
