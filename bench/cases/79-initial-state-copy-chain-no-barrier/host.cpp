// Distilled from baldurk/renderdoc PR #3896 ("Add missing pipeline barrier
// in Apply_InitialState()", merged 2026-09-08). When replaying a capture,
// WrappedVulkan::Apply_InitialState restores each buffer's initial contents
// by copying from a staging buffer (srcBuf) into the live buffer. srcBuf
// itself was initialised earlier in the same command buffer by a
// vkCmdCopyBuffer() (or vkCmdFillBuffer()), and nothing ordered that write
// before the second copy's read: SYNC-HAZARD-READ-AFTER-WRITE between two
// transfer commands (found with KosmicKrisp, VK_Image_Layouts test). The fix
// records a VkBufferMemoryBarrier (TRANSFER_WRITE -> TRANSFER_READ, whole
// buffer) before the copy out of srcBuf. Buffer roles: initial-contents
// readback -> srcBuf -> the captured application's vertex buffer, which the
// replayed draw then reads.
#include "../../common/app.hpp"
using namespace bench;

static void buffer_barrier1(VkCommandBuffer c, VkBuffer b, VkAccessFlags src, VkAccessFlags dst,
                            VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
  VkBufferMemoryBarrier bm{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  bm.srcAccessMask = src; bm.dstAccessMask = dst;
  bm.srcQueueFamilyIndex = bm.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bm.buffer = b; bm.offset = 0; bm.size = VK_WHOLE_SIZE;
  vkCmdPipelineBarrier(c, srcStage, dstStage, 0, 0, nullptr, 1, &bm, 0, nullptr);
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  const VkDeviceSize size = 6 * 8;
  // the initial contents saved at capture time (host-visible readback buffer)
  Buffer initial = make_buffer(a, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "initial_contents");
  float verts[12] = {-0.5f, -0.5f, 0.5f, -0.5f, 0.0f, 0.5f, -0.9f, 0.9f, -0.7f, 0.9f, -0.8f, 0.7f};
  memcpy(initial.map, verts, sizeof verts);
  Buffer srcBuf = make_buffer(a, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false,
                              "initial_state_src");
  Buffer live = make_buffer(a, size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false,
                            "captured_vertex_buffer");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.colors = {a.swap_format}; gd.vertex_stride = 8;
  gd.attributes = {{0, 0, VK_FORMAT_R32G32_SFLOAT, 0}};
  Pipeline replay = make_graphics(a, gd, "replayed_draw");

  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    // the previous replay's vertex reads of the live buffer happen before it is restored again
    buffer_barrier1(c, live.buf, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT);
    // the previous frame's copy out of srcBuf finished reading before srcBuf is re-initialised
    buffer_barrier1(c, srcBuf.buf, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT);
    // --- Apply_InitialState(live buffer) ---
    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(c, initial.buf, srcBuf.buf, 1, &region);          // srcBuf initialised in this command buffer
#ifdef FIXED
    // srcBuf was initialised with a vkCmdCopyBuffer() or vkCmdFillBuffer() earlier in this command buffer.
    // Add a barrier to make sure that command finishes before this copy.
    buffer_barrier1(c, srcBuf.buf, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
#endif
    vkCmdCopyBuffer(c, srcBuf.buf, live.buf, 1, &region);
    buffer_barrier1(c, live.buf, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
    // --- replay the captured frame ---
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, replay.pipe);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(c, 0, 1, &live.buf, &off);
    vkCmdDraw(c, 6, 1, 0, 0);
    vkCmdEndRendering(c);
  });
  return 0;
}
