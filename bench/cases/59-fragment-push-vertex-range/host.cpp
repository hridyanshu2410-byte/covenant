// Distilled from Thinkofname/rust-quake #1 (reported; no fix in the
// thread): the world pipeline's layout declares one push constant range,
// VERTEX (0x1), offset 0, size 256. The renderer pushes the vertex stage's
// matrix and then a 4-byte value at offset 64 with stageFlags 0x10
// (FRAGMENT). Validation: "stageFlags (0x10), offset (64), and size (4), must
// contain all stages in overlapping VkPushConstantRange stageFlags (0x1)"
// (VUID-vkCmdPushConstants-offset-01796) and "VkPushConstantRange in
// pipeline layout overlapping offset = 64 and size = 4, do not contain
// stageFlags 0x10" (-01795). The fixed variant follows the issue's answer:
// the layout's range includes the fragment stage, and the pushes name every
// stage of the range they overlap.
#include "../../common/app.hpp"
using namespace bench;

#ifdef FIXED
static const VkShaderStageFlags kRangeStages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
static const VkShaderStageFlags kMatrixPushStages = kRangeStages, kFragmentPushStages = kRangeStages;
#else
static const VkShaderStageFlags kRangeStages = VK_SHADER_STAGE_VERTEX_BIT;
static const VkShaderStageFlags kMatrixPushStages = VK_SHADER_STAGE_VERTEX_BIT, kFragmentPushStages = VK_SHADER_STAGE_FRAGMENT_BIT;
#endif

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  Image tex = make_image(a, 64, 64, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         "surface_texture");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "surface_set");
  write_image(a, set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, tex.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  struct Vtx { float pos[3]; float uv[2]; };
  Vtx tri[3] = {{{-0.5f, 0.5f, 0}, {0, 1}}, {{0.5f, 0.5f, 0}, {1, 1}}, {{0, -0.5f, 0}, {0.5f, 0}}};
  Buffer vb = make_buffer(a, sizeof tri, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "bsp_faces");
  memcpy(vb.map, tri, sizeof tri);

  Pipeline p{};
  {
    VkPushConstantRange pr{kRangeStages, 0, 256};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 1; li.pSetLayouts = &setl; li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pr;
    VK(vkCreatePipelineLayout(a.dev, &li, nullptr, &p.layout));
    VkPipelineShaderStageCreateInfo st[2]{};
    st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    st[0].module = sh; st[0].pName = "vsWorld";
    st[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    st[1].module = sh; st[1].pName = "fsWorld";
    VkVertexInputBindingDescription vbd{0, sizeof(Vtx), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription at[2] = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32_SFLOAT, 12}};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vbd;
    vi.vertexAttributeDescriptionCount = 2; vi.pVertexAttributeDescriptions = at;
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1; vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.lineWidth = 1; rs.cullMode = VK_CULL_MODE_NONE;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cb{}; cb.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cbs.attachmentCount = 1; cbs.pAttachments = &cb;
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dy.dynamicStateCount = 2; dy.pDynamicStates = dyn;
    VkPipelineRenderingCreateInfo rinfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rinfo.colorAttachmentCount = 1; rinfo.pColorAttachmentFormats = &a.swap_format;
    VkGraphicsPipelineCreateInfo gi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gi.pNext = &rinfo; gi.stageCount = 2; gi.pStages = st; gi.pVertexInputState = &vi; gi.pInputAssemblyState = &ia;
    gi.pViewportState = &vp; gi.pRasterizationState = &rs; gi.pMultisampleState = &ms; gi.pDepthStencilState = &ds;
    gi.pColorBlendState = &cbs; gi.pDynamicState = &dy; gi.layout = p.layout;
    VK(vkCreateGraphicsPipelines(a.dev, VK_NULL_HANDLE, 1, &gi, nullptr, &p.pipe));
    a.name(p.pipe, VK_OBJECT_TYPE_PIPELINE, "world");
  }

  VkCommandBuffer up = alloc_cmd(a, a.gfx_pool); begin(up);
  image_barrier(up, tex.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
  VK(vkEndCommandBuffer(up)); submit(a.gfx, up, {}, {}, VK_NULL_HANDLE); VK(vkQueueWaitIdle(a.gfx));

  float view_proj[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, p.layout, 0, 1, &set, 0, nullptr);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &off);
    vkCmdPushConstants(c, p.layout, kMatrixPushStages, 0, 64, view_proj);
    float fragment_value = 0.75f;
    vkCmdPushConstants(c, p.layout, kFragmentPushStages, 64, 4, &fragment_value);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
