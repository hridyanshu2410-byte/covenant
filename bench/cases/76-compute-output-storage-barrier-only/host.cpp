// Distilled from bkaradzic/bgfx PR #3878 ("Fix: add missing memory barriers
// for compute writes", merged 2026-08-10). After a dispatch, the OpenGL
// backend issued glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT) for the
// writable buffers bound to it. That only orders later shader-storage
// accesses; it does not cover the same buffers being consumed next as
// indirect draw arguments (GL_COMMAND_BARRIER_BIT), index data
// (GL_ELEMENT_ARRAY_BARRIER_BIT) or vertex/instance data
// (GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT), which is exactly how GPU-driven
// rendering (example 37-gpudrivenrendering) uses them, so a draw could
// consume stale compute output. The fix adds those bits (and
// GL_TEXTURE_FETCH_BARRIER_BIT for writable images).
// Port to Vulkan: the GL barrier bits become the destination scope of the
// buffer barrier after the dispatch (SHADER_STORAGE -> shader stages with
// SHADER_STORAGE_READ; COMMAND -> DRAW_INDIRECT/INDIRECT_COMMAND_READ;
// VERTEX_ATTRIB_ARRAY -> VERTEX_ATTRIBUTE_INPUT/VERTEX_ATTRIBUTE_READ;
// ELEMENT_ARRAY -> INDEX_INPUT/INDEX_READ). The cull shader is ours.
#include "../../common/app.hpp"
using namespace bench;

// bgfx RendererContextGL::submit, after dispatch: barrier for every writable buffer bound
static void barrier_after_dispatch(VkCommandBuffer c, VkBuffer buf) {
  VkPipelineStageFlags2 dstStage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;               // GL_SHADER_STORAGE_BARRIER_BIT
  VkAccessFlags2 dstAccess = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
#ifdef FIXED
  dstStage |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;            dstAccess |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;   // GL_COMMAND_BARRIER_BIT
  dstStage |= VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT;              dstAccess |= VK_ACCESS_2_INDEX_READ_BIT;              // GL_ELEMENT_ARRAY_BARRIER_BIT
  dstStage |= VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT;   dstAccess |= VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;   // GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT
#endif
  buffer_barrier(c, buf, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, dstStage, dstAccess);
}

// a graphics pipeline with a per-vertex and a per-instance vertex stream
static Pipeline make_instanced(App& a, VkShaderModule sh, const char* nm) {
  Pipeline p{};
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  VK(vkCreatePipelineLayout(a.dev, &li, nullptr, &p.layout));
  VkPipelineShaderStageCreateInfo st[2]{};
  st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[0].stage = VK_SHADER_STAGE_VERTEX_BIT; st[0].module = sh; st[0].pName = "vsMain";
  st[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module = sh; st[1].pName = "fsMain";
  VkVertexInputBindingDescription vb[2] = {{0, 8, VK_VERTEX_INPUT_RATE_VERTEX}, {1, 16, VK_VERTEX_INPUT_RATE_INSTANCE}};
  VkVertexInputAttributeDescription va[2] = {{0, 0, VK_FORMAT_R32G32_SFLOAT, 0}, {1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0}};
  VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vi.vertexBindingDescriptionCount = 2; vi.pVertexBindingDescriptions = vb; vi.vertexAttributeDescriptionCount = 2; vi.pVertexAttributeDescriptions = va;
  VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO}; vp.viewportCount = 1; vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO}; rs.lineWidth = 1;
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  VkPipelineColorBlendAttachmentState cb{}; cb.colorWriteMask = 0xF;
  VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}; cbs.attachmentCount = 1; cbs.pAttachments = &cb;
  VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO}; dy.dynamicStateCount = 2; dy.pDynamicStates = dyn;
  VkPipelineRenderingCreateInfo rinfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rinfo.colorAttachmentCount = 1; rinfo.pColorAttachmentFormats = &a.swap_format;
  VkGraphicsPipelineCreateInfo gi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  gi.pNext = &rinfo; gi.stageCount = 2; gi.pStages = st; gi.pVertexInputState = &vi; gi.pInputAssemblyState = &ia;
  gi.pViewportState = &vp; gi.pRasterizationState = &rs; gi.pMultisampleState = &ms; gi.pDepthStencilState = &ds;
  gi.pColorBlendState = &cbs; gi.pDynamicState = &dy; gi.layout = p.layout;
  VK(vkCreateGraphicsPipelines(a.dev, VK_NULL_HANDLE, 1, &gi, nullptr, &p.pipe));
  a.name(p.pipe, VK_OBJECT_TYPE_PIPELINE, nm);
  return p;
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  const uint32_t kObjects = 256;
  Buffer objects = make_buffer(a, kObjects * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, "object_list");
  float* o = (float*)objects.map;
  for (uint32_t i = 0; i < kObjects; i++) {
    float x = -1.5f + 3.0f * i / kObjects, y = -0.9f + 1.8f * ((i * 37) % kObjects) / kObjects;
    float* r = o + i * 8; r[0] = x; r[1] = y; r[2] = 0; r[3] = 0.02f; r[4] = x; r[5] = y; r[6] = 0.02f; r[7] = 0.02f;
  }
  Buffer instances = make_buffer(a, kObjects * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                 false, "instance_buffer");          // bgfx dynamic vertex buffer, compute write
  Buffer args = make_buffer(a, 20, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, "indirect_buffer");
  Buffer quad = make_buffer(a, 4 * 8, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "quad_vb");
  float qv[8] = {-1, -1, 1, -1, 1, 1, -1, 1}; memcpy(quad.map, qv, sizeof qv);
  Buffer ib = make_buffer(a, 6 * 2, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, true, "quad_ib");
  uint16_t qi[6] = {0, 1, 2, 0, 2, 3}; memcpy(ib.map, qi, sizeof qi);

  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
                                  {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}});
  auto set = alloc_set(a, setl, "cull_bindings");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, objects);
  write_buffer(a, set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, instances);
  write_buffer(a, set, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, args);
  Pipeline cull = make_compute(a, sh, "csCull", {setl}, "gpu_cull");
  Pipeline draw = make_instanced(a, sh, "instanced_draw");

  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    // reset the instance counter (the previous frame's indirect read is done: in_flight = 1 and the fence was waited)
    buffer_barrier(c, args.buf, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                   VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    vkCmdFillBuffer(c, args.buf, 0, 20, 0);
    buffer_barrier(c, args.buf, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    buffer_barrier(c, instances.buf, VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT, VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, cull.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, cull.layout, 0, 1, &set, 0, nullptr);
    vkCmdDispatch(c, kObjects / 64, 1, 1);
    barrier_after_dispatch(c, instances.buf);
    barrier_after_dispatch(c, args.buf);
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.pipe);
    VkBuffer vbs[2] = {quad.buf, instances.buf}; VkDeviceSize offs[2] = {0, 0};
    vkCmdBindVertexBuffers(c, 0, 2, vbs, offs);
    vkCmdBindIndexBuffer(c, ib.buf, 0, VK_INDEX_TYPE_UINT16);
    vkCmdDrawIndexedIndirect(c, args.buf, 0, 1, 20);
    vkCmdEndRendering(c);
  });
  return 0;
}
