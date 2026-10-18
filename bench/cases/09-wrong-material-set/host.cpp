// Distilled from etib-corp/evan #110 (fix PR #121): with materials
// interleaved across draws, the renderer bound the descriptor set of the
// previous material. Every set is valid and layout-compatible, so this is
// an expected miss for COVENANT: nothing says which set belongs to which draw.
#include "../../common/app.hpp"
using namespace bench;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  std::vector<Image> tex; std::vector<VkDescriptorSet> sets;
  for (int i = 0; i < 2; i++) {
    tex.push_back(make_image(a, 64, 64, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_USAGE_SAMPLED_BIT, "material" + std::to_string(i)));
    sets.push_back(alloc_set(a, setl, "material_set[" + std::to_string(i) + "]"));
    write_image(a, sets[i], 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, tex[i].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline pipe = make_graphics(a, gd, "material");
  int draw_material[] = {0, 1, 0, 1};
  run_frames(a, frames_from_env(2), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    if (f == 0)
      for (auto& t : tex)
        image_barrier(c, t.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipe);
    int prev = 0;
    for (int d = 0; d < 4; d++) {
#ifdef FIXED
      int m = draw_material[d];
#else
      int m = d == 0 ? draw_material[0] : prev;     // BUG: stale material index
#endif
      vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout, 0, 1, &sets[m], 0, nullptr);
      vkCmdDraw(c, 3, 1, 0, 0);
      prev = draw_material[d];
    }
    vkCmdEndRendering(c);
  });
}
