// Distilled from JasonPhone/vrtr commit c459d88 ("fix: temp for a strange VLL
// semaphore error, not triggered on my laptop", refs vulkan-guide#149 and
// Vulkan-ValidationLayers#9802). GPU::draw() keeps one FrameData per
// swapchain image (m_frames.resize(m_swapchain_images.size())) and cycles
// m_frame_number = (m_frame_number + 1) % m_frames.size(). Each frame waits
// its render_fence, acquires with its image_presented_semaphore, and the
// submit signalled m_frames[m_frame_number].present_ready_semaphore, which
// vkQueuePresentKHR then waited on. The fence only says the submit is done,
// not that the present that waits on the semaphore has consumed it, so the
// next use of the same frame slot may signal the semaphore again while the
// earlier present still holds it; and the frame number need not equal the
// acquired image index. The fix indexes present_ready_semaphore by the
// acquired swapchain image (m_frames[swapchain_img_idx]) for both the submit
// signal and the present wait. (vrtr records into a VkRenderPass; here the
// swapchain image is drawn with dynamic rendering.)
#include "../../common/app.hpp"
using namespace bench;

struct FrameData {
  VkCommandBuffer cmd_buffer_main;
  VkSemaphore image_presented_semaphore, present_ready_semaphore;
  VkFence render_fence;
};

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.colors = {a.swap_format};
  Pipeline tri = make_graphics(a, gd, "triangle");

  std::vector<FrameData> m_frames(a.swap_images.size());        // initFrameBuffers / initSyncStructures
  for (size_t i = 0; i < m_frames.size(); i++) {
    m_frames[i].cmd_buffer_main = alloc_cmd(a, a.gfx_pool);
    m_frames[i].image_presented_semaphore = make_semaphore(a, "m_frames[" + std::to_string(i) + "].image_presented_semaphore");
    m_frames[i].present_ready_semaphore = make_semaphore(a, "m_frames[" + std::to_string(i) + "].present_ready_semaphore");
    m_frames[i].render_fence = make_fence(a, true, "m_frames[" + std::to_string(i) + "].render_fence");
  }
  const uint64_t VK_ONE_SEC = 1000000000ull;
  uint32_t m_frame_number = 0;

  int frames = frames_from_env(8);
  for (int f = 0; f < frames; f++) {                             // GPU::draw()
    FrameData& fr = m_frames[m_frame_number];
    VK(vkWaitForFences(a.dev, 1, &fr.render_fence, VK_TRUE, VK_ONE_SEC));
    VK(vkResetFences(a.dev, 1, &fr.render_fence));
    uint32_t swapchain_img_idx = 0;
    VK(vkAcquireNextImageKHR(a.dev, a.swapchain, VK_ONE_SEC, fr.image_presented_semaphore, VK_NULL_HANDLE,
                             &swapchain_img_idx));
    VkCommandBuffer cmd = fr.cmd_buffer_main;
    begin(cmd);
    image_barrier(cmd, a.swap_images[swapchain_img_idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    begin_rendering(a, cmd, {a.swap_views[swapchain_img_idx]});
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tri.pipe);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);
    image_barrier(cmd, a.swap_images[swapchain_img_idx], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
    VK(vkEndCommandBuffer(cmd));

#ifdef FIXED
    VkSemaphore present_ready = m_frames[swapchain_img_idx].present_ready_semaphore;
#else
    VkSemaphore present_ready = m_frames[m_frame_number].present_ready_semaphore;
#endif
    VkCommandBufferSubmitInfo cmd_submit_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmd_submit_info.commandBuffer = cmd;
    VkSemaphoreSubmitInfo wait_info{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    wait_info.semaphore = fr.image_presented_semaphore; wait_info.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSemaphoreSubmitInfo signal_info{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signal_info.semaphore = present_ready; signal_info.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
    VkSubmitInfo2 submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit_info.waitSemaphoreInfoCount = 1; submit_info.pWaitSemaphoreInfos = &wait_info;
    submit_info.commandBufferInfoCount = 1; submit_info.pCommandBufferInfos = &cmd_submit_info;
    submit_info.signalSemaphoreInfoCount = 1; submit_info.pSignalSemaphoreInfos = &signal_info;
    VK(vkQueueSubmit2(a.gfx, 1, &submit_info, fr.render_fence));

    VkPresentInfoKHR present_info{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present_info.pSwapchains = &a.swapchain; present_info.swapchainCount = 1;
    present_info.pWaitSemaphores = &present_ready; present_info.waitSemaphoreCount = 1;
    present_info.pImageIndices = &swapchain_img_idx;
    VK(vkQueuePresentKHR(a.gfx, &present_info));
    m_frame_number = (m_frame_number + 1) % m_frames.size();
  }
  vkDeviceWaitIdle(a.dev);
}
