// Distilled from briar-systems/boom #162 (fix PR #173): the renderer uses
// reversed depth (GREATER test, far = 0). The first window pass of a frame
// that requested no color clear loaded depth with a fixed 1.0, the forward-Z
// far value, so every fragment failed GREATER against 1.0 and nothing drew.
// The fix starts depth at the far value of the pass's depth convention.
#include "../../common/app.hpp"
using namespace bench;

struct DepthConvention { VkCompareOp compare; float far_value; };
static const DepthConvention REVERSED{VK_COMPARE_OP_GREATER, 0.0f};

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  Image depth = make_image(a, 640, 480, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, "window_depth");
  Buffer cam = make_buffer(a, 64, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "camera");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}});
  auto set = alloc_set(a, setl, "camera_set");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, cam);
  auto sh = load_shader(a, "shaders.spv");
  const DepthConvention& conv = REVERSED;
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  gd.depth = VK_FORMAT_D32_SFLOAT; gd.depth_compare = conv.compare;
  Pipeline pipe = make_graphics(a, gd, "window_pass");
  float vp[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0.1f, 0};   // infinite reversed-Z projection
  memcpy(cam.map, vp, sizeof vp);
  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    depth_barrier(c, depth.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
#ifdef FIXED
    float depth_start = conv.far_value;           // far value of the pass's convention
#else
    float depth_start = 1.0f;                     // BUG: forward-Z far value, whatever the convention
#endif
    begin_rendering(a, c, {a.swap_views[idx]}, depth.view, depth_start);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
