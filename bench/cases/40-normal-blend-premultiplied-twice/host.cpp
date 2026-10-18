// Distilled from kevinraymond/fosfora #94 (fix PR #194): layer passes write
// premultiplied colour, and the compositor blends each layer onto the
// accumulated result. Its Normal blend mode mixed the layer's RGB by both
// the layer opacity and the layer's alpha, fg.rgb * opacity * fg.a, so the
// premultiplication was applied twice and antialiased edges darkened (a
// half-covered white fringe over white gave 0.75 instead of 1.0). The fix:
// a = clamp(fg.a) * opacity, rgb = fg.rgb * opacity + bg.rgb * (1 - a), and
// alpha accumulates as a + bg.a * (1 - a) instead of a max-union.
#include "../../common/app.hpp"
using namespace bench;

enum BlendMode : uint32_t { BLEND_NORMAL = 0, BLEND_ADD = 1, BLEND_SCREEN = 2 };
struct CompositeParams { float opacity; uint32_t blend_mode; float pad[2]; };

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto rt = [&](const char* nm) {
    return make_image(a, 640, 480, VK_FORMAT_R16G16B16A16_SFLOAT,
                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, nm);
  };
  Image layer = rt("layer_output"), accum = rt("composite_accum");
  Buffer params = make_buffer(a, sizeof(CompositeParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "composite_params");
  CompositeParams p{0.8f, BLEND_NORMAL, {0, 0}};
  memcpy(params.map, &p, sizeof p);

  auto sh = load_shader(a, "shaders.spv");
  GfxDesc ld; ld.vs = ld.fs = sh; ld.vs_entry = "vsFullscreen"; ld.fs_entry = "fsLayer";
  ld.colors = {VK_FORMAT_R16G16B16A16_SFLOAT};
  Pipeline layer_pass = make_graphics(a, ld, "layer_particles");
  ld.fs_entry = "fsBackground";
  Pipeline bg_pass = make_graphics(a, ld, "background");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "compositor_set");
  write_image(a, set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, layer.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  write_image(a, set, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, accum.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  write_buffer(a, set, 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, params);
  GfxDesc cd; cd.vs = cd.fs = sh; cd.vs_entry = "vsFullscreen"; cd.fs_entry = "fsComposite"; cd.sets = {setl};
  cd.colors = {a.swap_format};
  Pipeline compositor = make_graphics(a, cd, "compositor");

  auto offscreen = [&](VkCommandBuffer c, Image& im, Pipeline& pl) {
    image_barrier(c, im.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    begin_rendering(a, c, {im.view});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pl.pipe);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
    image_barrier(c, im.img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  };
  // run_frames moves the swapchain image to COLOR_ATTACHMENT first; offscreen passes are independent of it
  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    offscreen(c, accum, bg_pass);
    offscreen(c, layer, layer_pass);
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, compositor.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, compositor.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
