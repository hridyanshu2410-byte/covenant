// Distilled from KhronosGroup/Vulkan-Samples #1334 (dynamic_rendering
// sample): each frame's command buffer moves the shared depth image
// UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL with vkb::image_layout_transition's
// defaults for an UNDEFINED source, srcStage TOP_OF_PIPE and srcAccess 0.
// The previous frame's vkCmdEndRendering wrote that depth image, and with a
// TOP_OF_PIPE / no-access first scope the transition is not ordered after
// those writes: WRITE_AFTER_WRITE at LATE_FRAGMENT_TESTS. The issue gives
// the fix: srcStage LATE_FRAGMENT_TESTS, srcAccess DEPTH_STENCIL_ATTACHMENT_WRITE.
#include "../../common/app.hpp"
using namespace bench;

struct UBOVS { float projection[16]; float modelview[16]; float skybox_modelview[16]; };

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  Image depth_stencil = make_image(a, 640, 480, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                                   "depth_stencil");
  Buffer ubo = make_buffer(a, sizeof(UBOVS), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "ubo_vs");
  UBOVS u{};
  for (int i = 0; i < 4; i++) u.projection[i * 5] = u.modelview[i * 5] = u.skybox_modelview[i * 5] = 1;
  u.modelview[14] = 0.5f;
  memcpy(ubo.map, &u, sizeof u);
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}});
  auto set = alloc_set(a, setl, "descriptor_set");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo);
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  gd.depth = VK_FORMAT_D32_SFLOAT; gd.depth_compare = VK_COMPARE_OP_LESS_OR_EQUAL;
  gd.vs_entry = "vsSkybox"; gd.fs_entry = "fsSkybox";
  Pipeline skybox = make_graphics(a, gd, "skybox_pipeline");
  gd.vs_entry = "vsModel"; gd.fs_entry = "fsModel";
  Pipeline model = make_graphics(a, gd, "model_pipeline");

  run_frames(a, frames_from_env(4), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
#ifdef FIXED
    depth_barrier(c, depth_stencil.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
#else
    // image_layout_transition(cmd, image, UNDEFINED, DEPTH_ATTACHMENT_OPTIMAL, range): masks from the layouts
    depth_barrier(c, depth_stencil.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
#endif
    begin_rendering(a, c, {a.swap_views[idx]}, depth_stencil.view, 1.0f);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, skybox.layout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, skybox.pipe);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, model.pipe);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
