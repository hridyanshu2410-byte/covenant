// Distilled from hrydgard/ppsspp PR #12224 (issue #12215): a game renders to
// framebuffer A, switches to framebuffer B, then returns to A and draws more.
// The second render pass on A had no barrier after the first, so newer ARM
// Mali drivers could overlap or reorder the two passes: flickering in Star
// Wars: The Force Unleashed. The fix adds a barrier between the passes that
// target the same framebuffer.
#include "../../common/app.hpp"
using namespace bench;

static void begin_load(App& a, VkCommandBuffer c, VkImageView v) {   // continue a framebuffer (loadOp LOAD)
  VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  ca.imageView = v; ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  ca.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = {{0, 0}, a.extent}; ri.layerCount = 1; ri.colorAttachmentCount = 1; ri.pColorAttachments = &ca;
  vkCmdBeginRendering(c, &ri);
  VkViewport vp{0, 0, (float)a.extent.width, (float)a.extent.height, 0, 1};
  VkRect2D sc{{0, 0}, a.extent};
  vkCmdSetViewport(c, 0, 1, &vp);
  vkCmdSetScissor(c, 0, 1, &sc);
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  const VkFormat fb_fmt = VK_FORMAT_R8G8B8A8_UNORM;
  Image fbA = make_image(a, 640, 480, fb_fmt, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, "framebuffer_A");
  Image fbB = make_image(a, 640, 480, fb_fmt, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, "framebuffer_B");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "display_set");
  write_image(a, set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, fbA.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.fs_entry = "gameFs"; gd.colors = {fb_fmt};
  Pipeline game = make_graphics(a, gd, "game_draw");
  GfxDesc pd; pd.vs = pd.fs = sh; pd.fs_entry = "presentFs"; pd.sets = {setl}; pd.colors = {a.swap_format};
  Pipeline out = make_graphics(a, pd, "present_display");

  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    image_barrier(c, fbA.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    image_barrier(c, fbB.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    // pass 1: framebuffer A
    begin_rendering(a, c, {fbA.view});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, game.pipe);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
    // pass 2: the game switches to framebuffer B
    begin_rendering(a, c, {fbB.view});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, game.pipe);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
#ifdef FIXED
    // sequential dependency between the two passes on framebuffer A
    image_barrier(c, fbA.img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
#endif
    // pass 3: back to framebuffer A, drawing on top (BUG: no barrier after pass 1)
    begin_load(a, c, fbA.view);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, game.pipe);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
    // display framebuffer A
    image_barrier(c, fbA.img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, out.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, out.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
