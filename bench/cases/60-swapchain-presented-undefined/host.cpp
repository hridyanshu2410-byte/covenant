// Distilled from Pikachuxxxx/Razix #83 ("Vulkan swapchain and RT layout
// transition validation error", open): the renderer draws into the acquired
// swapchain image and presents it without ever moving the image out of
// VK_IMAGE_LAYOUT_UNDEFINED, so the present fails
// VUID-VkPresentInfoKHR-pImageIndices-01296 ("images passed to present must
// be in layout VK_IMAGE_LAYOUT_PRESENT_SRC_KHR or
// VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR but is in VK_IMAGE_LAYOUT_UNDEFINED").
// No fix is linked; the fixed variant does what the issue proposes: a
// pipeline barrier moves the image to COLOR_ATTACHMENT_OPTIMAL before
// rendering and another to PRESENT_SRC_KHR before presenting.
#include "../../common/app.hpp"
using namespace bench;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.colors = {a.swap_format}; gd.push_size = 4;
  Pipeline p = make_graphics(a, gd, "forward_test");

  const int MAX_FRAMES_IN_FLIGHT = 2;
  std::vector<VkCommandBuffer> cmd(MAX_FRAMES_IN_FLIGHT);
  std::vector<VkSemaphore> image_available(MAX_FRAMES_IN_FLIGHT);
  std::vector<VkFence> in_flight(MAX_FRAMES_IN_FLIGHT);
  std::vector<VkSemaphore> render_finished(a.swap_images.size());
  for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    cmd[i] = alloc_cmd(a, a.gfx_pool);
    image_available[i] = make_semaphore(a, "imageAvailable[" + std::to_string(i) + "]");
    in_flight[i] = make_fence(a, true, "inFlight[" + std::to_string(i) + "]");
  }
  for (size_t i = 0; i < render_finished.size(); i++)
    render_finished[i] = make_semaphore(a, "renderingDone[" + std::to_string(i) + "]");

  int frames = frames_from_env(6);
  for (int f = 0; f < frames; f++) {
    int k = f % MAX_FRAMES_IN_FLIGHT;
    VK(vkWaitForFences(a.dev, 1, &in_flight[k], VK_TRUE, UINT64_MAX));
    VK(vkResetFences(a.dev, 1, &in_flight[k]));
    uint32_t idx;
    VK(vkAcquireNextImageKHR(a.dev, a.swapchain, UINT64_MAX, image_available[k], VK_NULL_HANDLE, &idx));
    VkCommandBuffer c = cmd[k];
    begin(c);
#ifdef FIXED
    image_barrier(c, a.swap_images[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
#endif
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipe);
    float t = f * 0.016f;
    vkCmdPushConstants(c, p.layout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, 4, &t);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
#ifdef FIXED
    image_barrier(c, a.swap_images[idx], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
#endif
    VK(vkEndCommandBuffer(c));
    submit(a.gfx, c, {{image_available[k], VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT}}, {render_finished[idx]}, in_flight[k]);
    present(a, a.gfx, idx, render_finished[idx]);
  }
  vkDeviceWaitIdle(a.dev);
}
