// Distilled from corporateshark/lightweightvk PR #83 ("Fixed cmdDispatch()/
// cmdDispatchIndirect() buffer barriers to cover compute-to-compute
// hazards", merged 2026-08-23). lvk::CommandBuffer::cmdDispatch() emits one
// bufferBarrier() per buffer listed in Dependencies::buffers before the
// dispatch, with srcStage = VERTEX_SHADER | FRAGMENT_SHADER | TRANSFER and
// dstStage = COMPUTE_SHADER; bufferBarrier() derives the access masks from
// the stages. COMPUTE_SHADER was missing from the source stages, so when a
// dispatch consumes a buffer the previous dispatch wrote (a chained compute
// pipeline: prefix scan, radix sort, ...), the earlier compute write is in
// neither the first synchronization scope nor the first access scope. The
// fix adds VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT to the source stages in
// both dispatch functions. bufferBarrier() below reproduces lvk's
// stage -> access derivation (graphics queue, so no stage stripping).
#include "../../common/app.hpp"
using namespace bench;

// lvk::CommandBuffer::bufferBarrier(handle, srcStage, dstStage)
static void lvk_buffer_barrier(VkCommandBuffer c, const Buffer& b, VkPipelineStageFlags2 srcStage,
                               VkPipelineStageFlags2 dstStage) {
  const VkPipelineStageFlags2 nonShaderStages =
      VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
  VkAccessFlags2 srcAccess = 0, dstAccess = 0;
  if (srcStage & VK_PIPELINE_STAGE_2_TRANSFER_BIT) srcAccess |= VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
  if (srcStage & ~nonShaderStages) srcAccess |= VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
  if (dstStage & VK_PIPELINE_STAGE_2_TRANSFER_BIT) dstAccess |= VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
  if (dstStage & ~nonShaderStages) dstAccess |= VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
  if (dstStage & VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT) dstAccess |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
  buffer_barrier(c, b.buf, srcStage, srcAccess, dstStage, dstAccess);
}

// lvk::CommandBuffer::cmdDispatch(groupCount, deps)
static void cmd_dispatch(VkCommandBuffer c, uint32_t groups, std::vector<const Buffer*> deps) {
  for (const Buffer* b : deps)
    lvk_buffer_barrier(c, *b,
                       VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                           VK_PIPELINE_STAGE_2_TRANSFER_BIT
#ifdef FIXED
                           | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
#endif
                       ,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  vkCmdDispatch(c, groups, 1, 1);
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  const uint32_t N = 64 * 64;
  Buffer data = make_buffer(a, N * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, "scan_data");
  Buffer sums = make_buffer(a, 64 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, "scan_block_sums");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}});
  auto set = alloc_set(a, setl, "scan_bindings");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, data);
  write_buffer(a, set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, sums);
  Pipeline scan = make_compute(a, sh, "scanBlocks", {setl}, "scan_blocks");
  Pipeline add = make_compute(a, sh, "addBlockSums", {setl}, "scan_add_block_sums");

  VkCommandBuffer c = alloc_cmd(a, a.gfx_pool);
  VkFence done = make_fence(a, false, "scan_done");
  for (int iter = 0; iter < frames_from_env(3); iter++) {
    uint32_t* d = (uint32_t*)data.map;
    for (uint32_t i = 0; i < N; i++) d[i] = 1;          // host fills the input before each submit
    begin(c);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, scan.layout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, scan.pipe);
    cmd_dispatch(c, 64, {&data, &sums});                 // writes data and blockSums
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, add.pipe);
    cmd_dispatch(c, 64, {&data, &sums});                 // reads blockSums, read-modify-writes data
    buffer_barrier(c, data.buf, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    VK(vkEndCommandBuffer(c));
    submit(a.gfx, c, {}, {}, done);
    VK(vkWaitForFences(a.dev, 1, &done, VK_TRUE, UINT64_MAX));
    VK(vkResetFences(a.dev, 1, &done));
    printf("scan[%u] = %u\n", N - 1, d[N - 1]);
  }
  vkDeviceWaitIdle(a.dev);
  return 0;
}
