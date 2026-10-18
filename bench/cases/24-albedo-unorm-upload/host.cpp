// Distilled from TommyRadan/AlphaEngine #194 (fix PR #229): every 8-bit colour
// image was allocated rgba8_unorm, so sRGB-encoded albedo bytes entered the
// lighting math as if linear while the output was gamma-encoded:
// systematically over-bright midtones. The fix adds rgba8_srgb
// (VK_FORMAT_R8G8B8A8_SRGB) and picks the format by colour space: albedo,
// diffuse, emissive and sprites are sRGB; normal, metalness and roughness
// maps stay linear (UNORM).
#include "../../common/app.hpp"
using namespace bench;

enum class color_space { srgb, linear };
static VkFormat rgba8_format(color_space cs) {
#ifdef FIXED
  return cs == color_space::srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
#else
  (void)cs;
  return VK_FORMAT_R8G8B8A8_UNORM;                   // BUG: one format for every 8-bit image
#endif
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);                        // swapchain B8G8R8A8_SRGB
  const VkImageUsageFlags tex_usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  Image albedo = make_image(a, 256, 256, rgba8_format(color_space::srgb), tex_usage, "albedo");
  Image normal = make_image(a, 256, 256, rgba8_format(color_space::linear), tex_usage, "normal_map");
  Buffer light = make_buffer(a, 32, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "light");
  float l[8] = {0, -1, -1, 0, 1, 0.95f, 0.9f, 0};
  memcpy(light.map, l, sizeof l);
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "material_set");
  write_image(a, set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, albedo.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  write_image(a, set, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, normal.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  write_buffer(a, set, 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, light);
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline pipe = make_graphics(a, gd, "lit_textured");
  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    if (f == 0) {
      for (VkImage img : {albedo.img, normal.img})
        image_barrier(c, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
