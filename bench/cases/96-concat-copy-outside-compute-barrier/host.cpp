// Distilled from HartsyAI/HartsyInference PR #113 ("Synchronize Vulkan buffer
// copies against the dispatches around them", merged 2026-09-20). Every
// compute dispatch the backend records ends with a global compute->compute
// barrier (ComputeShader/ShaderStorageWrite -> ComputeShader/ShaderStorageRead).
// Concat is recorded as vkCmdCopyBuffer regions from its inputs into the
// output and was followed by the same RecordGlobalComputeBarrier(). That
// leaves two gaps: (1) the copies read buffers the producing dispatches just
// wrote, but a TRANSFER read is outside the dispatch barrier's destination
// scope (ComputeShader/ShaderStorageRead), and (2) the barrier after the
// copies names ComputeShader/ShaderStorageWrite as its source, so the copies'
// TRANSFER writes are not made available to the next dispatch that reads the
// concat output. The fix records RecordComputeToCopyBarrierOn() before the
// copies and RecordCopyToComputeBarrierOn() after them (Copy/TransferWrite ->
// ComputeShader/ShaderStorageRead|Write).
// Port: C# -> C++; two mul_scalar producers, Concat, one rms_norm consumer.
#include "../../common/app.hpp"
using namespace bench;

struct Params { uint32_t count; float scale; float eps; uint32_t pad; };

static void global_barrier(VkCommandBuffer cb, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                           VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
  VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  mb.srcStageMask = srcStage; mb.srcAccessMask = srcAccess; mb.dstStageMask = dstStage; mb.dstAccessMask = dstAccess;
  VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dep.memoryBarrierCount = 1; dep.pMemoryBarriers = &mb;
  vkCmdPipelineBarrier2(cb, &dep);
}

// VulkanCommandStream.RecordGlobalComputeBarrier()
static void record_global_compute_barrier(VkCommandBuffer cb) {
  global_barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
}
#ifdef FIXED
static void record_compute_to_copy_barrier(VkCommandBuffer cb) {
  global_barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
                 VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                     VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT,
                 VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
}
static void record_copy_to_compute_barrier(VkCommandBuffer cb) {
  global_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
                 VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                     VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
}
#endif

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  const uint32_t N = 1024;
  const VkBufferUsageFlags u = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  Buffer x = make_buffer(a, N * 4, u, true, "tensor_x");
  Buffer lhs = make_buffer(a, N * 4, u, false, "tensor_lhs");
  Buffer rhs = make_buffer(a, N * 4, u, false, "tensor_rhs");
  Buffer cat = make_buffer(a, 2 * N * 4, u, false, "tensor_concat");
  Buffer out = make_buffer(a, 2 * N * 4, u, true, "tensor_out");
  for (uint32_t i = 0; i < N; i++) ((float*)x.map)[i] = 0.001f * i;

  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}});
  auto bind_pair = [&](const Buffer& s, const Buffer& d, const char* nm) {
    VkDescriptorSet set = alloc_set(a, setl, nm);
    write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, s);
    write_buffer(a, set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, d);
    return set;
  };
  VkDescriptorSet set_lhs = bind_pair(x, lhs, "mul_scalar_lhs"), set_rhs = bind_pair(x, rhs, "mul_scalar_rhs"),
                  set_norm = bind_pair(cat, out, "rms_norm_concat");

  VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Params)};
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.setLayoutCount = 1; li.pSetLayouts = &setl; li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pr;
  VkPipelineLayout layout; VK(vkCreatePipelineLayout(a.dev, &li, nullptr, &layout));
  auto make_cs = [&](const char* entry, const char* nm) {
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module = sh; ci.stage.pName = entry; ci.layout = layout;
    VkPipeline pl; VK(vkCreateComputePipelines(a.dev, VK_NULL_HANDLE, 1, &ci, nullptr, &pl));
    a.name(pl, VK_OBJECT_TYPE_PIPELINE, nm);
    return pl;
  };
  VkPipeline mul = make_cs("mul_scalar", "mul_scalar"), norm = make_cs("rms_norm", "rms_norm");

  // VulkanBackend: each op = bind + dispatch + trailing compute->compute barrier
  auto dispatch = [&](VkCommandBuffer cb, VkPipeline pl, VkDescriptorSet set, Params prm, uint32_t groups) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof prm, &prm);
    vkCmdDispatch(cb, groups, 1, 1);
    record_global_compute_barrier(cb);
  };

  VkCommandBuffer cb = alloc_cmd(a, a.comp_pool);
  VkFence fence = make_fence(a, false, "stream_fence");
  for (int step = 0; step < frames_from_env(3); step++) {
    begin(cb);
    dispatch(cb, mul, set_lhs, {N, 2.0f, 0, 0}, N / 64);
    dispatch(cb, mul, set_rhs, {N, -0.5f, 0, 0}, N / 64);

    // VulkanBackend.Concat(output, lhs, rhs): one region per input
#ifdef FIXED
    record_compute_to_copy_barrier(cb);
#endif
    VkBufferCopy r0{0, 0, N * 4}, r1{0, N * 4, N * 4};
    vkCmdCopyBuffer(cb, lhs.buf, cat.buf, 1, &r0);
    vkCmdCopyBuffer(cb, rhs.buf, cat.buf, 1, &r1);
#ifdef FIXED
    record_copy_to_compute_barrier(cb);
#else
    record_global_compute_barrier(cb);
#endif

    dispatch(cb, norm, set_norm, {2 * N, 1.0f, 1e-6f, 0}, 1);
    // DrainAndFlush(): the result is read back on the host
    buffer_barrier(cb, out.buf, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    VK(vkEndCommandBuffer(cb));
    submit(a.comp, cb, {}, {}, fence);
    VK(vkWaitForFences(a.dev, 1, &fence, VK_TRUE, UINT64_MAX));
    VK(vkResetFences(a.dev, 1, &fence));
  }
  float sum = 0;
  for (uint32_t i = 0; i < 2 * N; i++) sum += ((float*)out.map)[i];
  printf("rms_norm(concat) sum %f\n", sum);
  vkDeviceWaitIdle(a.dev);
  return 0;
}
