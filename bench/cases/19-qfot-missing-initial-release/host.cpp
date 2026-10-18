// Distilled from KhronosGroup/Vulkan-Samples #3 (compute_nbody): the particle
// storage buffer is VK_SHARING_MODE_EXCLUSIVE and used by a compute queue of
// another family (storage writes) and the graphics queue (vertex input).
// Every frame does release/acquire pairs, and semaphores order the queues.
// But the buffer is first filled on the graphics queue and never released,
// so the compute queue's first acquire has no matching release. The proposed
// fix: a release barrier after the initial upload in prepare_storage_buffers.
#include "../../common/app.hpp"
using namespace bench;

static void qfot_buffer_barrier(VkCommandBuffer c, VkBuffer buf, uint32_t src_family, uint32_t dst_family,
                                VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                                VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
  VkBufferMemoryBarrier2 b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
  b.srcStageMask = src_stage; b.srcAccessMask = src_access; b.dstStageMask = dst_stage; b.dstAccessMask = dst_access;
  b.srcQueueFamilyIndex = src_family; b.dstQueueFamilyIndex = dst_family;
  b.buffer = buf; b.size = VK_WHOLE_SIZE;
  VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  di.bufferMemoryBarrierCount = 1; di.pBufferMemoryBarriers = &b;
  vkCmdPipelineBarrier2(c, &di);
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  const uint32_t N = 1024; const int FRAMES = frames_from_env(4);
  const bool split = a.gfx_family != a.comp_family;
  Buffer staging = make_buffer(a, N * 32, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "particle_staging");
  Buffer storage = make_buffer(a, N * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, "particle_storage");   // EXCLUSIVE
  std::vector<float> init(N * 8, 0.0f);
  memcpy(staging.map, init.data(), N * 32);

  auto sh = load_shader(a, "shaders.spv");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}});
  auto set = alloc_set(a, setl, "compute_set");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, storage);
  Pipeline integrate = make_compute(a, sh, "integrateMain", {setl}, "nbody_integrate");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.colors = {a.swap_format}; gd.vertex_stride = 32;
  gd.attributes = {{0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 16}};
  Pipeline render = make_graphics(a, gd, "nbody_render");

  VkSemaphore graphics_sem = make_semaphore(a, "graphics_semaphore");
  VkSemaphore compute_sem = make_semaphore(a, "compute_semaphore");

  // prepare_storage_buffers: copy on the graphics queue, then let compute start
  VkCommandBuffer up = alloc_cmd(a, a.gfx_pool);
  begin(up);
  VkBufferCopy region{0, 0, N * 32};
  vkCmdCopyBuffer(up, staging.buf, storage.buf, 1, &region);
#ifdef FIXED
  if (split)   // release to the compute family, matching compute's first acquire
    qfot_buffer_barrier(up, storage.buf, a.gfx_family, a.comp_family,
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, 0, 0);
#endif
  VK(vkEndCommandBuffer(up));
  VkFence upf = make_fence(a, false, "upload_fence");
  submit(a.gfx, up, {}, {graphics_sem}, upf);       // signals graphics_sem so compute can run first
  VK(vkWaitForFences(a.dev, 1, &upf, VK_TRUE, UINT64_MAX));

  VkCommandBuffer ccmd = alloc_cmd(a, a.comp_pool), gcmd = alloc_cmd(a, a.gfx_pool);
  VkSemaphore acquired = make_semaphore(a, "acquired");
  std::vector<VkSemaphore> done(a.swap_images.size());  // one per swapchain image
  for (size_t i = 0; i < done.size(); i++) done[i] = make_semaphore(a, "render_done[" + std::to_string(i) + "]");
  VkFence cfence = make_fence(a, true, "compute_fence"), gfence = make_fence(a, true, "graphics_fence");
  for (int f = 0; f < FRAMES; f++) {
    // compute: acquire from graphics, integrate, release to graphics
    VK(vkWaitForFences(a.dev, 1, &cfence, VK_TRUE, UINT64_MAX));
    VK(vkResetFences(a.dev, 1, &cfence));
    begin(ccmd);
    if (split)
      qfot_buffer_barrier(ccmd, storage.buf, a.gfx_family, a.comp_family, 0, 0,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    vkCmdBindPipeline(ccmd, VK_PIPELINE_BIND_POINT_COMPUTE, integrate.pipe);
    vkCmdBindDescriptorSets(ccmd, VK_PIPELINE_BIND_POINT_COMPUTE, integrate.layout, 0, 1, &set, 0, nullptr);
    vkCmdDispatch(ccmd, N / 256, 1, 1);
    if (split)
      qfot_buffer_barrier(ccmd, storage.buf, a.comp_family, a.gfx_family,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, 0, 0);
    VK(vkEndCommandBuffer(ccmd));
    submit(a.comp, ccmd, {{graphics_sem, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT}}, {compute_sem}, cfence);

    // graphics: acquire from compute, draw the particles, release to compute
    VK(vkWaitForFences(a.dev, 1, &gfence, VK_TRUE, UINT64_MAX));
    VK(vkResetFences(a.dev, 1, &gfence));
    uint32_t idx;
    VK(vkAcquireNextImageKHR(a.dev, a.swapchain, UINT64_MAX, acquired, VK_NULL_HANDLE, &idx));
    begin(gcmd);
    if (split)
      qfot_buffer_barrier(gcmd, storage.buf, a.comp_family, a.gfx_family, 0, 0,
                          VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT, VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT);
    image_barrier(gcmd, a.swap_images[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    begin_rendering(a, gcmd, {a.swap_views[idx]});
    vkCmdBindPipeline(gcmd, VK_PIPELINE_BIND_POINT_GRAPHICS, render.pipe);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(gcmd, 0, 1, &storage.buf, &off);
    vkCmdDraw(gcmd, N, 1, 0, 0);
    vkCmdEndRendering(gcmd);
    image_barrier(gcmd, a.swap_images[idx], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
    if (split)
      qfot_buffer_barrier(gcmd, storage.buf, a.gfx_family, a.comp_family,
                          VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT, 0, 0, 0);
    VK(vkEndCommandBuffer(gcmd));
    submit(a.gfx, gcmd, {{compute_sem, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT},
                         {acquired, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT}}, {graphics_sem, done[idx]}, gfence);
    present(a, a.gfx, idx, done[idx]);
  }
  vkDeviceWaitIdle(a.dev);
  return 0;
}
