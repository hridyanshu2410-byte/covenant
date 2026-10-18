// Distilled from water-rs/waterui #239 (fixed in shaderloom 0.1.2, PR #249):
// waterui precompiles its WGSL shaders to SPIR-V with shaderloom, which
// used naga's default writer flags, including ADJUST_COORDINATE_SPACE: the
// SPIR-V negates the y of BuiltIn::Position. wgpu-hal's Vulkan backend does
// not use that flag; it maps WebGPU clip space (+y up) to Vulkan's (+y down)
// by setting a negative-height viewport (wgpu-hal/src/vulkan/command.rs).
// Fed precompiled SPIR-V, wgpu cannot know the shader already flipped, so
// both flips apply and everything renders upside down on Vulkan. The fix
// clears ADJUST_COORDINATE_SPACE in shaderloom so the SPIR-V matches
// wgpu-hal's handling. This host code plays wgpu-hal's part.
#include "../../common/app.hpp"
using namespace bench;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");     // shaderloom output
  GfxDesc gd; gd.vs = gd.fs = sh; gd.vs_entry = "vsRect"; gd.fs_entry = "fsRect";
  gd.colors = {a.swap_format}; gd.push_size = 32;
  Pipeline rect = make_graphics(a, gd, "waterui_rect");

  struct { float bounds[4]; float color[4]; } title_bar = {{-1.0f, 0.8f, 1.0f, 1.0f}, {0.2f, 0.4f, 0.9f, 1.0f}};   // top strip
  struct { float bounds[4]; float color[4]; } body = {{-0.9f, -0.9f, 0.9f, 0.7f}, {0.95f, 0.95f, 0.95f, 1.0f}};

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    // wgpu-hal set_viewport: flip y with a negative height (VK_KHR_maintenance1)
    VkViewport vp{0, (float)a.extent.height, (float)a.extent.width, -(float)a.extent.height, 0, 1};
    vkCmdSetViewport(c, 0, 1, &vp);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, rect.pipe);
    vkCmdPushConstants(c, rect.layout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, 32, &body);
    vkCmdDraw(c, 6, 1, 0, 0);
    vkCmdPushConstants(c, rect.layout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, 32, &title_bar);
    vkCmdDraw(c, 6, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
