// Distilled from ggml-org/llama.cpp PR #25229 ("vulkan: when using transfer
// queue for async copies, sync on event_wait to avoid race", merged
// 2026-07-16). With async_use_transfer_queue, ggml_backend_vk_cpy_tensor_async
// and set_tensor_async record their copies into a separate transfer context
// that is submitted on the transfer queue; the next compute context waits on
// the transfer semaphore. ggml_backend_vk_event_wait (used by the scheduler
// before a backend consumes a tensor another backend produced) only made the
// compute context wait for the event. The transfer context did not, so the
// async copy of the producer's output could start on the transfer queue while
// the producer's graph was still writing that tensor on the compute queue.
// The fix also pushes the event's semaphore onto the transfer context:
//   transfer_ctx->s->wait_semaphores.push_back(vkev->tl_semaphore);
// Port: one device, the producer and consumer graphs are one dispatch each on
// the compute (graphics-family) queue, the transfer context runs on the bench's
// second queue; the event's timeline semaphore becomes a binary semaphore
// (timeline semaphores are not enabled in the bench device), and the compute
// context's vkCmdWaitEvents becomes the equivalent same-queue barrier.
#include "../../common/app.hpp"
using namespace bench;

struct vk_push { uint32_t ne; float param; };

// ggml_vk_create_buffer: device buffers are shared by the compute and transfer
// queue families when a dedicated transfer queue is used
static Buffer ggml_vk_create_buffer(App& a, VkDeviceSize size, bool host, const std::string& nm) {
  Buffer b{}; b.size = size;
  uint32_t fams[2] = {a.gfx_family, a.comp_family};
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size;
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bi.sharingMode = a.gfx_family != a.comp_family ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
  bi.queueFamilyIndexCount = a.gfx_family != a.comp_family ? 2 : 0; bi.pQueueFamilyIndices = fams;
  VK(vkCreateBuffer(a.dev, &bi, nullptr, &b.buf));
  VkMemoryRequirements r; vkGetBufferMemoryRequirements(a.dev, b.buf, &r);
  VkMemoryAllocateInfo mi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mi.allocationSize = r.size;
  mi.memoryTypeIndex = mem_type(a, r.memoryTypeBits, host ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                                          : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK(vkAllocateMemory(a.dev, &mi, nullptr, &b.mem));
  VK(vkBindBufferMemory(a.dev, b.buf, b.mem, 0));
  if (host) VK(vkMapMemory(a.dev, b.mem, 0, size, 0, &b.map));
  a.name(b.buf, VK_OBJECT_TYPE_BUFFER, nm);
  return b;
}

struct vk_pipeline { Pipeline p; VkDescriptorSet set; };

static void ggml_vk_dispatch_pipeline(VkCommandBuffer c, const vk_pipeline& pl, uint32_t ne, float param) {
  vk_push pc{ne, param};
  vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pl.p.pipe);
  vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, pl.p.layout, 0, 1, &pl.set, 0, nullptr);
  vkCmdPushConstants(c, pl.p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
  vkCmdDispatch(c, (ne + 255) / 256, 1, 1);
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  const uint32_t ne = 4096;
  const VkDeviceSize nb = ne * sizeof(float);

  Buffer inp      = ggml_vk_create_buffer(a, nb, true, "inp_embd");          // backend A input
  Buffer tensor_a = ggml_vk_create_buffer(a, nb, false, "ffn_out-31 (backend A)");
  Buffer tensor_b = ggml_vk_create_buffer(a, nb, false, "ffn_out-31 (copy, backend B)");
  Buffer out      = ggml_vk_create_buffer(a, nb, true, "result_output");
  for (uint32_t i = 0; i < ne; i++) ((float*)inp.map)[i] = (float)(i % 97) * 0.01f;

  auto dsl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
                                 {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}});
  auto make_pipe = [&](const char* entry) {
    Pipeline p{};
    VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(vk_push)};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 1; li.pSetLayouts = &dsl; li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pr;
    VK(vkCreatePipelineLayout(a.dev, &li, nullptr, &p.layout));
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module = sh; ci.stage.pName = entry; ci.layout = p.layout;
    VK(vkCreateComputePipelines(a.dev, VK_NULL_HANDLE, 1, &ci, nullptr, &p.pipe));
    a.name(p.pipe, VK_OBJECT_TYPE_PIPELINE, entry);
    return p;
  };
  vk_pipeline scale{make_pipe("scale_f32"), alloc_set(a, dsl, "scale_f32 set")};
  write_buffer(a, scale.set, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, inp);
  write_buffer(a, scale.set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, tensor_a);
  vk_pipeline add{make_pipe("add_scalar_f32"), alloc_set(a, dsl, "add_scalar_f32 set")};
  write_buffer(a, add.set, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, tensor_b);
  write_buffer(a, add.set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, out);

  VkCommandBuffer compute_a = alloc_cmd(a, a.gfx_pool);   // backend A compute ctx
  VkCommandBuffer compute_b = alloc_cmd(a, a.gfx_pool);   // backend B compute ctx
  VkCommandBuffer transfer_b = alloc_cmd(a, a.comp_pool); // backend B transfer ctx
  VkSemaphore event_sem = make_semaphore(a, "vk_event.tl_semaphore");
  VkSemaphore transfer_sem = make_semaphore(a, "transfer_semaphore");
  VkFence fence = make_fence(a, false, "ctx->fence");

  for (int token = 0; token < frames_from_env(4); token++) {
    // backend A: graph_compute, then event_record (signals the event semaphore)
    begin(compute_a);
    ggml_vk_dispatch_pipeline(compute_a, scale, ne, 1.0f + token);
    VK(vkEndCommandBuffer(compute_a));
    submit(a.gfx, compute_a, {}, {event_sem}, VK_NULL_HANDLE);

    // backend B: event_wait(event) -> compute ctx waits for the event
    begin(compute_b);
    buffer_barrier(compute_b, tensor_a.buf, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                   VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT);   // ggml_vk_wait_events
    std::vector<std::pair<VkSemaphore, VkPipelineStageFlags>> transfer_waits;
#ifdef FIXED
    transfer_waits.push_back({event_sem, VK_PIPELINE_STAGE_TRANSFER_BIT});      // transfer ctx waits too
#endif

    // backend B: cpy_tensor_async(tensor_a -> tensor_b) on the transfer ctx
    begin(transfer_b);
    VkBufferCopy region{0, 0, nb};
    vkCmdCopyBuffer(transfer_b, tensor_a.buf, tensor_b.buf, 1, &region);
    VK(vkEndCommandBuffer(transfer_b));
    submit(a.comp, transfer_b, transfer_waits, {transfer_sem}, VK_NULL_HANDLE);

    // backend B: graph_compute; the compute ctx waits on the transfer semaphore
    ggml_vk_dispatch_pipeline(compute_b, add, ne, 0.5f);
    buffer_barrier(compute_b, out.buf, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    VK(vkEndCommandBuffer(compute_b));
    std::vector<std::pair<VkSemaphore, VkPipelineStageFlags>> compute_waits = {{transfer_sem, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT}};
#ifndef FIXED
    compute_waits.push_back({event_sem, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT});
#endif
    submit(a.gfx, compute_b, compute_waits, {}, fence);
    VK(vkWaitForFences(a.dev, 1, &fence, VK_TRUE, UINT64_MAX));
    VK(vkResetFences(a.dev, 1, &fence));
    float r0 = ((float*)out.map)[1];
    if (token == 0) printf("result[1] = %f\n", r0);
  }
  vkDeviceWaitIdle(a.dev);
  return 0;
}
