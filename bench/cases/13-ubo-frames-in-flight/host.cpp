// Distilled from KingoBoiii/DingoEngine PR #63: the uniform buffer was not
// versioned per frame in flight, so the CPU rewrote it while the GPU was
// still reading it for the previous frame.
#include "../../common/app.hpp"
using namespace bench;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  const int IN_FLIGHT = 2, FRAMES = frames_from_env();
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}});
  auto vs = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = vs; gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline pipe = make_graphics(a, gd, "forward");

#ifdef FIXED
  const int COPIES = IN_FLIGHT;                 // one camera buffer per frame in flight
#else
  const int COPIES = 1;                         // BUG: one buffer shared by every frame
#endif
  std::vector<Buffer> cam; std::vector<VkDescriptorSet> sets;
  for (int i = 0; i < COPIES; i++) {
    cam.push_back(make_buffer(a, 64, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "camera[" + std::to_string(i) + "]"));
    sets.push_back(alloc_set(a, setl, "camera_set[" + std::to_string(i) + "]"));
    write_buffer(a, sets[i], 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, cam[i]);
  }
  std::vector<VkSemaphore> done(a.swap_images.size());  // one per swapchain image
  for (size_t i = 0; i < done.size(); i++) done[i] = make_semaphore(a, "render_done[" + std::to_string(i) + "]");
  VkCommandBuffer cmd[IN_FLIGHT]; VkSemaphore acquired[IN_FLIGHT]; VkFence fence[IN_FLIGHT];
  for (int i = 0; i < IN_FLIGHT; i++) {
    cmd[i] = alloc_cmd(a, a.gfx_pool);
    acquired[i] = make_semaphore(a, "acquired[" + std::to_string(i) + "]");
    fence[i] = make_fence(a, true, "frame_fence[" + std::to_string(i) + "]");
  }
  for (int f = 0; f < FRAMES; f++) {
    int k = f % IN_FLIGHT;
    VK(vkWaitForFences(a.dev, 1, &fence[k], VK_TRUE, UINT64_MAX));
    VK(vkResetFences(a.dev, 1, &fence[k]));
    uint32_t idx;
    VK(vkAcquireNextImageKHR(a.dev, a.swapchain, UINT64_MAX, acquired[k], VK_NULL_HANDLE, &idx));
    float vp[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    vp[0] = 1.0f + 0.01f * f;
    memcpy(cam[k % COPIES].map, vp, sizeof vp);   // CPU writes this frame's camera

    begin(cmd[k]);
    image_barrier(cmd[k], a.swap_images[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    begin_rendering(a, cmd[k], {a.swap_views[idx]});
    vkCmdBindPipeline(cmd[k], VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipe);
    vkCmdBindDescriptorSets(cmd[k], VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout, 0, 1, &sets[k % COPIES], 0, nullptr);
    vkCmdDraw(cmd[k], 3, 1, 0, 0);
    vkCmdEndRendering(cmd[k]);
    image_barrier(cmd[k], a.swap_images[idx], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
    VK(vkEndCommandBuffer(cmd[k]));
    submit(a.gfx, cmd[k], {{acquired[k], VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT}}, {done[idx]}, fence[k]);
    present(a, a.gfx, idx, done[idx]);
  }
  vkDeviceWaitIdle(a.dev);
  return 0;
}
