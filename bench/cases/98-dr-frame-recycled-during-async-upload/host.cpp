// Distilled from mpv-player/mpv issue #13811 ("GPU-next + Vulkan frame
// corruption under heavy system load"), fixed by commit 7c25b3e
// ("vo_gpu_next: fix upload synchronization", committed 2026-05-03). With direct
// rendering (DR) the software decoder decodes straight into host-visible GPU
// buffers handed out by vo_gpu_next. map_frame() then uploads each plane from
// that buffer with pl_upload_plane(), which only records an asynchronous
// buffer-to-texture copy. For DR buffers no completion callback held a
// reference to the mp_image, so the image (and with it the DR buffer) went
// back to the decoder's frame pool as soon as map_frame() returned; the
// decoder could then write the next frame into the same buffer while the copy
// of the previous one was still pending on the GPU (torn / out-of-order
// frames under load). The fix sets
//   data[n].callback = talloc_free; data[n].priv = mp_image_new_ref(mpi);
// for DR buffers too, so the buffer is returned to the pool only once the GPU
// has finished reading it.
// Port: libplacebo's pl_buf / pl_tex_upload -> a VkBuffer pool and
// vkCmdCopyBufferToImage in the frame's command buffer; the libplacebo
// completion callback runs when the frame's fence has been waited on.
#include "../../common/app.hpp"
#include <deque>
using namespace bench;

constexpr int kInFlight = 2;
constexpr uint32_t W = 320, H = 180;

struct dr_buf { Buffer buf; int id; };
struct mp_image { dr_buf* dr; int pts; };

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");

  // the decoder's frame pool (AVBufferPool backed by vo_gpu_next DR buffers);
  // the most recently released buffer is handed out first
  std::deque<dr_buf*> pool;
  for (int i = 0; i < 3; i++)
    pool.push_back(new dr_buf{make_buffer(a, W * H * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true,
                                          "dr_buf[" + std::to_string(i) + "]"), i});
  auto get_image = [&](int pts) { dr_buf* b = pool.front(); pool.pop_front(); return mp_image{b, pts}; };
  auto talloc_free = [&](mp_image& m) { pool.push_front(m.dr); m.dr = nullptr; };

  Image tex = make_image(a, W, H, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                         "frame_tex[0]");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "main_pass_set");
  write_image(a, set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, tex.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline mainPass = make_graphics(a, gd, "main_pass");

  // pending upload callbacks per frame in flight (libplacebo runs them once
  // the command buffer that read the buffer has completed)
  std::vector<std::vector<mp_image>> pending_callbacks(kInFlight);
  mp_image cur{};

  run_frames(a, frames_from_env(6), kInFlight,
    [&](VkCommandBuffer c, uint32_t imageIndex, int f) {
      // map_frame(): pl_upload_plane() from the DR buffer
      image_barrier(c, tex.img, f == 0 ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
      VkBufferImageCopy region{};
      region.bufferRowLength = W;
      region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      region.imageExtent = {W, H, 1};
      vkCmdCopyBufferToImage(c, cur.dr->buf.buf, tex.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
      image_barrier(c, tex.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
#ifdef FIXED
      pending_callbacks[f % kInFlight].push_back(cur);   // data[n].priv = mp_image_new_ref(mpi)
#endif
      talloc_free(cur);   // map_frame() is done with its reference
#ifdef FIXED
      // the callback still owns the buffer: put it back as "in use"
      pool.pop_front();
#endif
      // pl_render_image(): main pass to the swapchain
      begin_rendering(a, c, {a.swap_views[imageIndex]});
      vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, mainPass.pipe);
      vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, mainPass.layout, 0, 1, &set, 0, nullptr);
      vkCmdDraw(c, 3, 1, 0, 0);
      vkCmdEndRendering(c);
    },
    [&](int f) {
      // this frame slot's fence has signalled: run its completion callbacks
      for (auto& m : pending_callbacks[f % kInFlight]) talloc_free(m);
      pending_callbacks[f % kInFlight].clear();
      // decoder thread: decode the next frame into a pooled DR buffer
      cur = get_image(f);
      uint8_t* px = (uint8_t*)cur.dr->buf.map;
      for (uint32_t y = 0; y < H; y++)
        for (uint32_t x = 0; x < W; x++) {
          uint8_t* p = px + (y * W + x) * 4;
          p[0] = (uint8_t)(x + f * 8); p[1] = (uint8_t)y; p[2] = (uint8_t)(f * 40); p[3] = 255;
        }
    });
  return 0;
}
