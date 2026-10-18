// Distilled from JorriFransen/v10 commit 3652d32: the renderer kept
// MAX_FRAMES_IN_FLIGHT (2) render-finished semaphores and used
// render_finished_semaphores[current_frame] both as the submit's signal and
// as the present's wait, while the swapchain has 3 images. The frame fence
// only says the submit is done, not that the presentation engine has
// consumed the semaphore, so the slot's next submit may signal it while it
// is still in use. The fix allocates one per swapchain image
// (this.images.len) and indexes it with the acquired image_index.
#include "../../common/app.hpp"
using namespace bench;

static const int MAX_FRAMES_IN_FLIGHT = 2;

struct Vertex { float pos[2]; float color[3]; };

struct Renderer {
  App* a;
  std::vector<VkCommandBuffer> command_buffers;
  std::vector<VkSemaphore> image_available_semaphores;
  std::vector<VkSemaphore> render_finished_semaphores;
  std::vector<VkFence> in_flight_fences;
  uint32_t current_frame = 0;

  void init(App& app) {
    a = &app;
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      command_buffers.push_back(alloc_cmd(app, app.gfx_pool));
      image_available_semaphores.push_back(make_semaphore(app, "image_available[" + std::to_string(i) + "]"));
      in_flight_fences.push_back(make_fence(app, true, "in_flight[" + std::to_string(i) + "]"));
    }
#ifdef FIXED
    size_t n = app.swap_images.size();
#else
    size_t n = MAX_FRAMES_IN_FLIGHT;               // BUG: fewer than the swapchain's images
#endif
    for (size_t i = 0; i < n; i++)
      render_finished_semaphores.push_back(make_semaphore(app, "render_finished[" + std::to_string(i) + "]"));
  }

  // returns the acquired image index
  uint32_t begin_frame() {
    VK(vkWaitForFences(a->dev, 1, &in_flight_fences[current_frame], VK_TRUE, UINT64_MAX));
    uint32_t image_index;
    VK(vkAcquireNextImageKHR(a->dev, a->swapchain, UINT64_MAX, image_available_semaphores[current_frame],
                             VK_NULL_HANDLE, &image_index));
    VK(vkResetFences(a->dev, 1, &in_flight_fences[current_frame]));
    begin(command_buffers[current_frame]);
    return image_index;
  }

  void end_frame(uint32_t image_index) {
    VkCommandBuffer c = command_buffers[current_frame];
    VK(vkEndCommandBuffer(c));
#ifdef FIXED
    VkSemaphore finished = render_finished_semaphores[image_index];
#else
    VkSemaphore finished = render_finished_semaphores[current_frame];
#endif
    submit(a->gfx, c, {{image_available_semaphores[current_frame], VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT}},
           {finished}, in_flight_fences[current_frame]);
    present(*a, a->gfx, image_index, finished);
    current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
  }
};

int main(int argc, char** argv) {
  App a = make_app(argc, argv);                       // 3 swapchain images
  const Vertex verts[3] = {{{0.0f, -0.5f}, {1, 0, 0}}, {{0.5f, 0.5f}, {0, 1, 0}}, {{-0.5f, 0.5f}, {0, 0, 1}}};
  Buffer vb = make_buffer(a, sizeof verts, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "triangle_vertices");
  memcpy(vb.map, verts, sizeof verts);
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.colors = {a.swap_format}; gd.vertex_stride = sizeof(Vertex);
  gd.attributes = {{0, 0, VK_FORMAT_R32G32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 8}};
  Pipeline pipe = make_graphics(a, gd, "triangle");

  Renderer r; r.init(a);
  for (int f = 0; f < frames_from_env(8); f++) {
    uint32_t idx = r.begin_frame();
    VkCommandBuffer c = r.command_buffers[r.current_frame];
    image_barrier(c, a.swap_images[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipe);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &off);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
    image_barrier(c, a.swap_images[idx], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
    r.end_frame(idx);
  }
  vkDeviceWaitIdle(a.dev);
  return 0;
}
