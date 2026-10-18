// Distilled from Overv/VulkanTutorial #407 (the same pattern is in bgfx #4017
// and sokol #1420): the render-finished semaphore that vkQueuePresentKHR waits
// on is one per frame in flight. The frame fence says the submit finished, but
// nothing says the presentation engine has consumed the semaphore, so the
// next frame using that slot may signal it while it is still in use
// (VUID-vkQueueSubmit-pSignalSemaphores-00067). The fix: one render-finished
// semaphore per swapchain image, chosen by the acquired image index.
#include "../../common/app.hpp"
using namespace bench;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);                        // 3 swapchain images
  const int MAX_FRAMES_IN_FLIGHT = 2, FRAMES = frames_from_env(8);
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.colors = {a.swap_format};
  Pipeline pipe = make_graphics(a, gd, "triangle");

  std::vector<VkCommandBuffer> cmd(MAX_FRAMES_IN_FLIGHT);
  std::vector<VkSemaphore> image_available(MAX_FRAMES_IN_FLIGHT);
  std::vector<VkFence> in_flight(MAX_FRAMES_IN_FLIGHT);
  for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    cmd[i] = alloc_cmd(a, a.gfx_pool);
    image_available[i] = make_semaphore(a, "image_available[" + std::to_string(i) + "]");
    in_flight[i] = make_fence(a, true, "in_flight[" + std::to_string(i) + "]");
  }
#ifdef FIXED
  const size_t N_FINISHED = a.swap_images.size();     // one per swapchain image
#else
  const size_t N_FINISHED = MAX_FRAMES_IN_FLIGHT;     // BUG: one per frame in flight
#endif
  std::vector<VkSemaphore> render_finished(N_FINISHED);
  for (size_t i = 0; i < N_FINISHED; i++)
    render_finished[i] = make_semaphore(a, "render_finished[" + std::to_string(i) + "]");

  for (int f = 0; f < FRAMES; f++) {
    int cur = f % MAX_FRAMES_IN_FLIGHT;
    VK(vkWaitForFences(a.dev, 1, &in_flight[cur], VK_TRUE, UINT64_MAX));
    VK(vkResetFences(a.dev, 1, &in_flight[cur]));
    uint32_t idx;
    VK(vkAcquireNextImageKHR(a.dev, a.swapchain, UINT64_MAX, image_available[cur], VK_NULL_HANDLE, &idx));
    VkCommandBuffer c = cmd[cur];
    begin(c);
    image_barrier(c, a.swap_images[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipe);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
    image_barrier(c, a.swap_images[idx], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
    VK(vkEndCommandBuffer(c));
#ifdef FIXED
    VkSemaphore finished = render_finished[idx];
#else
    VkSemaphore finished = render_finished[cur];
#endif
    submit(a.gfx, c, {{image_available[cur], VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT}}, {finished}, in_flight[cur]);
    present(a, a.gfx, idx, finished);
  }
  vkDeviceWaitIdle(a.dev);
  return 0;
}
