// Distilled from mattias800/prosper PR #3250 ("render backend: a readback
// needs an availability operation, not just a fence", fixes #2944, merged
// 2026-09-02). The render backend reads device-written, HOST_VISIBLE
// buffers on the CPU: a vkCmdCopy* writes the buffer, the CPU waits for the
// submission's fence, maps the buffer and reads it (here: the colour
// readback that is written back to guest memory). A fence wait orders
// execution, but the copy's writes are only made available to the host by a
// dependency with dstStageMask = HOST and dstAccessMask = HOST_READ; none of
// the eight host-read sites recorded one. The fix records
// host_read_barrier_for(): a whole-buffer TRANSFER_WRITE -> HOST_READ buffer
// barrier after each copy whose result the CPU reads.
// (Synchronization validation does not report this: it cannot see CPU reads
// through mapped pointers.)
#include "../../common/app.hpp"
using namespace bench;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  const uint32_t W = 256, H = 256;
  Image color = make_image(a, W, H, VK_FORMAT_R8G8B8A8_UNORM,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, "guest_color_target");
  Buffer readback = make_buffer(a, W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, "color_readback");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.colors = {VK_FORMAT_R8G8B8A8_UNORM};
  Pipeline guest = make_graphics(a, gd, "guest_draw");

  VkCommandBuffer c = alloc_cmd(a, a.gfx_pool);
  VkFence fence = make_fence(a, false, "submission_fence");
  std::vector<uint8_t> guest_memory(W * H * 4);
  for (int frame = 0; frame < frames_from_env(3); frame++) {
    begin(c);
    image_barrier(c, color.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COPY_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    ca.imageView = color.view; ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {W, H}}; ri.layerCount = 1; ri.colorAttachmentCount = 1; ri.pColorAttachments = &ca;
    vkCmdBeginRendering(c, &ri);
    VkViewport vp{0, 0, (float)W, (float)H, 0, 1}; VkRect2D sc{{0, 0}, {W, H}};
    vkCmdSetViewport(c, 0, 1, &vp); vkCmdSetScissor(c, 0, 1, &sc);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, guest.pipe);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
    image_barrier(c, color.img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {W, H, 1};
    vkCmdCopyImageToBuffer(c, color.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buf, 1, &r);
#ifdef FIXED
    // record_host_read_barrier(cmd, host_read_barrier_for(readback))
    buffer_barrier(c, readback.buf, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
#endif
    VK(vkEndCommandBuffer(c));
    submit(a.gfx, c, {}, {}, fence);
    VK(vkWaitForFences(a.dev, 1, &fence, VK_TRUE, UINT64_MAX));
    VK(vkResetFences(a.dev, 1, &fence));
    memcpy(guest_memory.data(), readback.map, guest_memory.size());   // CPU read through the mapped pointer
  }
  uint32_t sum = 0;
  for (uint8_t b : guest_memory) sum += b;
  printf("guest memory checksum %u\n", sum);
  vkDeviceWaitIdle(a.dev);
  return 0;
}
