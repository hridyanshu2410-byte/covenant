// Distilled from google/filament PR #10165 ("Fix srgb conversion for FL0
// with transparent blending mode", merged 2026-07-07). At feature level 0
// (GLES 2.0 class devices) Filament emulates an sRGB swapchain: the
// framebuffer is UNORM and surface_main.fs applies
// pow(fragColor.rgb, 0.4545) itself when frameUniforms.rec709 is set. For
// BLEND_MODE_TRANSPARENT (and FADE) fragColor is premultiplied by alpha at
// that point, so the encode computes pow(c * a, 0.4545) =
// pow(c, 0.4545) * pow(a, 0.4545): with a = 0.047 the colour that reaches
// the ONE / ONE_MINUS_SRC_ALPHA blend is about 5x too strong (pow(0.047,
// 0.4545) ~ 0.25). The fix unpremultiplies, encodes, then premultiplies
// again. GLES2 -> Vulkan: a B8G8R8A8_UNORM swapchain and premultiplied
// blend state; the material is a faint transparent overlay over a grey
// background.
#include "../../common/app.hpp"
using namespace bench;

struct Blend { VkBlendFactor src_c, dst_c, src_a, dst_a; };
static Pipeline make_blended(App& a, VkShaderModule sh, VkFormat color, std::vector<VkDescriptorSetLayout> sets,
                             bool blend, Blend bl, const char* nm) {
  Pipeline p{};
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.setLayoutCount = (uint32_t)sets.size(); li.pSetLayouts = sets.data();
  VK(vkCreatePipelineLayout(a.dev, &li, nullptr, &p.layout));
  VkPipelineShaderStageCreateInfo st[2]{};
  st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[0].stage = VK_SHADER_STAGE_VERTEX_BIT; st[0].module = sh; st[0].pName = "vsMain";
  st[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module = sh; st[1].pName = "fsMain";
  VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO}; vp.viewportCount = 1; vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO}; rs.lineWidth = 1;
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  VkPipelineColorBlendAttachmentState cb{};
  cb.blendEnable = blend; cb.colorWriteMask = 0xF;
  cb.srcColorBlendFactor = bl.src_c; cb.dstColorBlendFactor = bl.dst_c; cb.colorBlendOp = VK_BLEND_OP_ADD;
  cb.srcAlphaBlendFactor = bl.src_a; cb.dstAlphaBlendFactor = bl.dst_a; cb.alphaBlendOp = VK_BLEND_OP_ADD;
  VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}; cbs.attachmentCount = 1; cbs.pAttachments = &cb;
  VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO}; dy.dynamicStateCount = 2; dy.pDynamicStates = dyn;
  VkPipelineRenderingCreateInfo rinfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rinfo.colorAttachmentCount = 1; rinfo.pColorAttachmentFormats = &color;
  VkGraphicsPipelineCreateInfo gi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  gi.pNext = &rinfo; gi.stageCount = 2; gi.pStages = st; gi.pVertexInputState = &vi; gi.pInputAssemblyState = &ia;
  gi.pViewportState = &vp; gi.pRasterizationState = &rs; gi.pMultisampleState = &ms; gi.pDepthStencilState = &ds;
  gi.pColorBlendState = &cbs; gi.pDynamicState = &dy; gi.layout = p.layout;
  VK(vkCreateGraphicsPipelines(a.dev, VK_NULL_HANDLE, 1, &gi, nullptr, &p.pipe));
  a.name(p.pipe, VK_OBJECT_TYPE_PIPELINE, nm);
  return p;
}

int main(int argc, char** argv) {
  AppDesc d; d.swap_format = VK_FORMAT_B8G8R8A8_UNORM;   // FL0: no sRGB swapchain
  App a = make_app(argc, argv, d);
  auto sh = load_shader(a, "shaders.spv");
  Buffer frame = make_buffer(a, 16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "frame_uniforms");
  int32_t fu[4] = {1, 0, 0, 0};                           // rec709: emulate the sRGB swapchain
  memcpy(frame.map, fu, sizeof fu);
  Buffer mat = make_buffer(a, 16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "overlay_material");
  float base[4] = {0.8f, 0.9f, 1.0f, 0.047f};             // faint transparent overlay (linear colour, alpha 0.047)
  memcpy(mat.map, base, sizeof base);
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "overlay_bindings");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, frame);
  write_buffer(a, set, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, mat);
  const VkBlendFactor ONE = VK_BLEND_FACTOR_ONE, ISA = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  Pipeline transparent = make_blended(a, sh, a.swap_format, {setl}, true, {ONE, ISA, ONE, ISA}, "material_transparent");

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    ca.imageView = a.swap_views[idx]; ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ca.clearValue.color = {{0.5f, 0.5f, 0.5f, 1.0f}};    // scene behind the overlay (sRGB-encoded grey)
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, a.extent}; ri.layerCount = 1; ri.colorAttachmentCount = 1; ri.pColorAttachments = &ca;
    vkCmdBeginRendering(c, &ri);
    VkViewport vp{0, 0, (float)a.extent.width, (float)a.extent.height, 0, 1}; VkRect2D sc{{0, 0}, a.extent};
    vkCmdSetViewport(c, 0, 1, &vp); vkCmdSetScissor(c, 0, 1, &sc);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, transparent.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, transparent.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
  return 0;
}
