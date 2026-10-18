// Distilled from tatolab/streamlib #2506 (fix PR #2509): every frame the
// window's present target gets a first barrier UNDEFINED ->
// COLOR_ATTACHMENT_OPTIMAL whose source stage is NONE. The image-available
// semaphore is waited at COLOR_ATTACHMENT_OUTPUT, and a barrier only chains
// off that wait if its first scope includes that stage, so the layout
// transition was unordered against the presentation engine's read of the
// image: SYNC-HAZARD-WRITE-AFTER-READ against vkAcquireNextImageKHR every
// frame. The fix makes the barrier's source stage the semaphore wait stage
// (one constant, IMAGE_AVAILABLE_SEMAPHORE_WAIT_STAGE, used by both) and
// passes the four masks as one SwapchainImageBarrierScopes struct.
#include "../../common/app.hpp"
using namespace bench;

static const VkPipelineStageFlags2 IMAGE_AVAILABLE_SEMAPHORE_WAIT_STAGE = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

struct SwapchainImageBarrierScopes {
  VkPipelineStageFlags2 src_stage; VkAccessFlags2 src_access;
  VkPipelineStageFlags2 dst_stage; VkAccessFlags2 dst_access;
};

#ifdef FIXED
static const SwapchainImageBarrierScopes SWAPCHAIN_ACQUIRE_BARRIER_SCOPES = {
    IMAGE_AVAILABLE_SEMAPHORE_WAIT_STAGE, VK_ACCESS_2_NONE,
    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT};
#endif
static const SwapchainImageBarrierScopes SWAPCHAIN_PRESENT_BARRIER_SCOPES = {
    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
    VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE};

static void record_swapchain_image_barrier(VkCommandBuffer c, VkImage img, VkImageLayout from, VkImageLayout to,
                                           const SwapchainImageBarrierScopes& s) {
  image_barrier(c, img, from, to, s.src_stage, s.src_access, s.dst_stage, s.dst_access);
}

static void submit2(VkQueue q, VkCommandBuffer c, VkSemaphore wait, VkPipelineStageFlags2 wait_stage,
                    VkSemaphore signal, VkFence fence) {
  VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO}; cbi.commandBuffer = c;
  VkSemaphoreSubmitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO}; wi.semaphore = wait; wi.stageMask = wait_stage;
  VkSemaphoreSubmitInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO}; si.semaphore = signal;
  si.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  VkSubmitInfo2 s{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  s.waitSemaphoreInfoCount = 1; s.pWaitSemaphoreInfos = &wi;
  s.commandBufferInfoCount = 1; s.pCommandBufferInfos = &cbi;
  s.signalSemaphoreInfoCount = 1; s.pSignalSemaphoreInfos = &si;
  VK(vkQueueSubmit2(q, 1, &s, fence));
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  // the display pipeline blits a camera frame into the window
  Image frame = make_image(a, 1280, 720, VK_FORMAT_R8G8B8A8_SRGB,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, "camera_frame");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "display_set");
  write_image(a, set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, frame.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline display = make_graphics(a, gd, "display");

  const int FRAMES_IN_FLIGHT = 2;
  std::vector<VkCommandBuffer> cmd(FRAMES_IN_FLIGHT);
  std::vector<VkSemaphore> image_available(FRAMES_IN_FLIGHT);
  std::vector<VkFence> frame_fence(FRAMES_IN_FLIGHT);
  for (int i = 0; i < FRAMES_IN_FLIGHT; i++) {
    cmd[i] = alloc_cmd(a, a.gfx_pool);
    image_available[i] = make_semaphore(a, "image_available[" + std::to_string(i) + "]");
    frame_fence[i] = make_fence(a, true, "frame_fence[" + std::to_string(i) + "]");
  }
  std::vector<VkSemaphore> render_finished;                 // one per swapchain image
  for (size_t i = 0; i < a.swap_images.size(); i++)
    render_finished.push_back(make_semaphore(a, "render_finished[" + std::to_string(i) + "]"));

  for (int f = 0; f < frames_from_env(6); f++) {
    int k = f % FRAMES_IN_FLIGHT;
    VK(vkWaitForFences(a.dev, 1, &frame_fence[k], VK_TRUE, UINT64_MAX));
    VK(vkResetFences(a.dev, 1, &frame_fence[k]));
    uint32_t idx;
    VK(vkAcquireNextImageKHR(a.dev, a.swapchain, UINT64_MAX, image_available[k], VK_NULL_HANDLE, &idx));
    VkCommandBuffer c = cmd[k];
    begin(c);
    if (f == 0)
      image_barrier(c, frame.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
#ifdef FIXED
    record_swapchain_image_barrier(c, a.swap_images[idx], VK_IMAGE_LAYOUT_UNDEFINED,
                                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, SWAPCHAIN_ACQUIRE_BARRIER_SCOPES);
#else
    record_swapchain_image_barrier(c, a.swap_images[idx], VK_IMAGE_LAYOUT_UNDEFINED,
                                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                   {VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT});
#endif
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, display.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, display.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
    record_swapchain_image_barrier(c, a.swap_images[idx], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                   VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, SWAPCHAIN_PRESENT_BARRIER_SCOPES);
    VK(vkEndCommandBuffer(c));
#ifdef FIXED
    VkPipelineStageFlags2 wait_stage = IMAGE_AVAILABLE_SEMAPHORE_WAIT_STAGE;
#else
    VkPipelineStageFlags2 wait_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
#endif
    submit2(a.gfx, c, image_available[k], wait_stage, render_finished[idx], frame_fence[k]);
    present(a, a.gfx, idx, render_finished[idx]);
  }
  vkDeviceWaitIdle(a.dev);
  return 0;
}
