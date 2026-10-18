// Observed running iveevi/rcgp-samples on a desktop NVIDIA GPU: about 3
// runs in 100 froze in vkAcquireNextImageKHR, and validation reported that
// the app acquires more swapchain images than it may hold with an infinite
// timeout. Distilled: a frame that is skipped (here: every third, standing
// in for a minimized or out-of-date window) acquires an image and returns
// without presenting it, so acquired images pile up.
#include "../../common/app.hpp"
using namespace bench;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);                // 3 images; the surface allows holding 2
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}});
  Buffer cam = make_buffer(a, 64, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "camera");
  auto set = alloc_set(a, setl, "camera_set");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, cam);
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline pipe = make_graphics(a, gd, "forward");
  VkCommandBuffer cmd = alloc_cmd(a, a.gfx_pool);
  VkSemaphore acquired[8];
  std::vector<VkSemaphore> done(a.swap_images.size());  // one per swapchain image
  for (size_t i = 0; i < done.size(); i++) done[i] = make_semaphore(a, "render_done[" + std::to_string(i) + "]");
  for (int i = 0; i < 8; i++) acquired[i] = make_semaphore(a, "acquired[" + std::to_string(i) + "]");
  VkFence fence = make_fence(a, true, "frame_fence");
  const int FRAMES = frames_from_env(7);
  for (int f = 0; f < FRAMES; f++) {
    VK(vkWaitForFences(a.dev, 1, &fence, VK_TRUE, UINT64_MAX));
    uint32_t idx;
    VK(vkAcquireNextImageKHR(a.dev, a.swapchain, UINT64_MAX, acquired[f % 8], VK_NULL_HANDLE, &idx));
    bool skip = f % 3 == 1;
    if (skip) {
#ifdef FIXED
      // give the image back: present it unchanged after waiting for the acquire
      VkCommandBuffer c2 = cmd;
      VK(vkResetFences(a.dev, 1, &fence));
      begin(c2);
      image_barrier(c2, a.swap_images[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
      VK(vkEndCommandBuffer(c2));
      submit(a.gfx, c2, {{acquired[f % 8], VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT}}, {done[idx]}, fence);
      present(a, a.gfx, idx, done[idx]);
#endif
      continue;                                // BUG: acquired image never presented
    }
    VK(vkResetFences(a.dev, 1, &fence));
    begin(cmd);
    image_barrier(cmd, a.swap_images[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    begin_rendering(a, cmd, {a.swap_views[idx]});
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);
    image_barrier(cmd, a.swap_images[idx], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
    VK(vkEndCommandBuffer(cmd));
    submit(a.gfx, cmd, {{acquired[f % 8], VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT}}, {done[idx]}, fence);
    present(a, a.gfx, idx, done[idx]);
  }
  vkDeviceWaitIdle(a.dev);
}
