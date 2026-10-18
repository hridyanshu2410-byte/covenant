// Distilled from bkaradzic/bgfx #3302 (open; workaround kick(true)): the
// Vulkan renderer keeps two frames in flight. FrameBufferVK::acquire() calls
// vkAcquireNextImageKHR with the frame slot's image-acquired semaphore, and
// only afterwards does the frame wait for the slot's completion fence
// (m_completedFence) before recording; CommandQueueVK::kick() then submits
// waiting on that semaphore and signalling the fence. Acquiring for frame
// N + 2 therefore signals the semaphore that frame N's submission may still
// be waiting on (VUID-vkAcquireNextImageKHR-semaphore-01779: "Semaphore must
// not have any pending operations"). No upstream fix is merged; the fixed
// variant follows the issue's analysis: wait for the slot's fence before
// acquiring with its semaphore. (The workaround, kick(true), waits for the
// queue to go idle after every submit.)
#include "../../common/app.hpp"
using namespace bench;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.colors = {a.swap_format}; gd.push_size = 4;
  Pipeline p = make_graphics(a, gd, "bgfx_program");

  const int kMaxFramesInFlight = 2;
  std::vector<VkCommandBuffer> cmd(kMaxFramesInFlight);
  std::vector<VkSemaphore> image_acquired(kMaxFramesInFlight);
  std::vector<VkFence> completed(kMaxFramesInFlight);
  std::vector<VkSemaphore> render_done(a.swap_images.size());
  for (int i = 0; i < kMaxFramesInFlight; i++) {
    cmd[i] = alloc_cmd(a, a.gfx_pool);
    image_acquired[i] = make_semaphore(a, "m_presentDoneSemaphore[" + std::to_string(i) + "]");
    completed[i] = make_fence(a, true, "m_completedFence[" + std::to_string(i) + "]");
  }
  for (size_t i = 0; i < render_done.size(); i++)
    render_done[i] = make_semaphore(a, "m_renderDoneSemaphore[" + std::to_string(i) + "]");

  int frames = frames_from_env(6);
  for (int f = 0; f < frames; f++) {
    int k = f % kMaxFramesInFlight;
    uint32_t idx;
#ifdef FIXED
    VK(vkWaitForFences(a.dev, 1, &completed[k], VK_TRUE, UINT64_MAX));
    VK(vkAcquireNextImageKHR(a.dev, a.swapchain, UINT64_MAX, image_acquired[k], VK_NULL_HANDLE, &idx));
#else
    // FrameBufferVK::acquire()
    VK(vkAcquireNextImageKHR(a.dev, a.swapchain, UINT64_MAX, image_acquired[k], VK_NULL_HANDLE, &idx));
    // CommandQueueVK::alloc(): wait for this slot's previous submission
    VK(vkWaitForFences(a.dev, 1, &completed[k], VK_TRUE, UINT64_MAX));
#endif
    VK(vkResetFences(a.dev, 1, &completed[k]));
    VkCommandBuffer c = cmd[k];
    begin(c);
    image_barrier(c, a.swap_images[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipe);
    float t = f * 0.016f;
    vkCmdPushConstants(c, p.layout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, 4, &t);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
    image_barrier(c, a.swap_images[idx], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
    VK(vkEndCommandBuffer(c));
    // CommandQueueVK::kick(false)
    submit(a.gfx, c, {{image_acquired[k], VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT}}, {render_done[idx]}, completed[k]);
    present(a, a.gfx, idx, render_done[idx]);
  }
  vkDeviceWaitIdle(a.dev);
}
