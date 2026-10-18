// Distilled from RPCS3/rpcs3 PR #13573: the overlay renderer's pipeline layout
// declares one push constant range per stage (vertex: bytes 0..79, fragment:
// bytes 80..111), but it updated them with a single vkCmdPushConstants using
// VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT over the whole
// block (VUID-vkCmdPushConstants-offset-01795: every byte of the update must
// be in a range that includes every stage in stageFlags). The fix issues one
// call per stage, each over its own range.
#include "../../common/app.hpp"
using namespace bench;

struct OverlayPush { float transform[16]; float viewport[4]; float color[4]; float time, blur, pad[2]; };
static const uint32_t VS_SIZE = 80, FS_OFFSET = 80, FS_SIZE = 32;

static Pipeline make_overlay_pipeline(App& a, VkShaderModule sh) {
  Pipeline p{};
  VkPushConstantRange ranges[2] = {{VK_SHADER_STAGE_VERTEX_BIT, 0, VS_SIZE},
                                   {VK_SHADER_STAGE_FRAGMENT_BIT, FS_OFFSET, FS_SIZE}};
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.pushConstantRangeCount = 2; li.pPushConstantRanges = ranges;
  VK(vkCreatePipelineLayout(a.dev, &li, nullptr, &p.layout));
  VkPipelineShaderStageCreateInfo st[2]{};
  st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  st[0].module = sh; st[0].pName = "vsMain";
  st[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  st[1].module = sh; st[1].pName = "fsMain";
  VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vp.viewportCount = 1; vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.lineWidth = 1; rs.cullMode = VK_CULL_MODE_NONE;
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  VkPipelineColorBlendAttachmentState cb{};
  cb.blendEnable = VK_TRUE; cb.colorWriteMask = 0xF;
  cb.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA; cb.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  cb.colorBlendOp = VK_BLEND_OP_ADD; cb.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  cb.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; cb.alphaBlendOp = VK_BLEND_OP_ADD;
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
  a.name(p.pipe, VK_OBJECT_TYPE_PIPELINE, "overlay");
  return p;
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  Pipeline overlay = make_overlay_pipeline(a, sh);
  OverlayPush push{};
  for (int i = 0; i < 4; i++) push.transform[i * 5] = 1;
  push.viewport[2] = push.viewport[3] = 1; push.color[0] = push.color[3] = 1; push.blur = 0.25f;
  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    push.time = f * 0.016f;
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, overlay.pipe);
#ifdef FIXED
    vkCmdPushConstants(c, overlay.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, VS_SIZE, &push);
    vkCmdPushConstants(c, overlay.layout, VK_SHADER_STAGE_FRAGMENT_BIT, FS_OFFSET, FS_SIZE,
                       reinterpret_cast<const char*>(&push) + FS_OFFSET);
#else
    vkCmdPushConstants(c, overlay.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof push, &push);                     // BUG: no range covers both stages
#endif
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
