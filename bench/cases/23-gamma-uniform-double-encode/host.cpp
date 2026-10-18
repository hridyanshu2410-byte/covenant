// Distilled from MatthewLacerda2/rusty #415 (fix PR #534): the final composite
// pass applied pow(c, 1/gamma) with gamma = 2.2 from the visual-correction
// component, and the window surface is an sRGB format that encodes again on
// write: double gamma in the window (headless screenshots used Rgba8Unorm and
// encoded once). The fix lets the target format own the encode: gamma becomes
// a neutral display-gamma tweak (default 1.0; old scenes load gamma / 2.2) and
// the headless target also becomes sRGB. This reproduction is the window path.
#include "../../common/app.hpp"
using namespace bench;

struct PostFx { float gamma; float exposure; float pad[2]; };

struct VisualCorrection {
#ifdef FIXED
  float display_gamma = 1.0f;     // display tweak only; the sRGB surface encodes
#else
  float gamma = 2.2f;             // BUG: encodes in the shader, then the sRGB surface encodes again
#endif
  float exposure = 1.0f;
};

int main(int argc, char** argv) {
  App a = make_app(argc, argv);                        // window surface: B8G8R8A8_SRGB
  Image scene = make_image(a, 640, 480, VK_FORMAT_R16G16B16A16_SFLOAT,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, "hdr_scene");
  Buffer ubo = make_buffer(a, sizeof(PostFx), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "postfx_uniforms");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "composite_set");
  write_image(a, set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, scene.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  write_buffer(a, set, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo);
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline composite = make_graphics(a, gd, "fs_composite");

  VisualCorrection vc;
  PostFx p{};
#ifdef FIXED
  p.gamma = vc.display_gamma;
#else
  p.gamma = vc.gamma;
#endif
  p.exposure = vc.exposure;
  memcpy(ubo.map, &p, sizeof p);
  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    if (f == 0)
      image_barrier(c, scene.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, composite.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, composite.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
