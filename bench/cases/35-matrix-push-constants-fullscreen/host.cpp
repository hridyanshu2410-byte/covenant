// Distilled from vsg-dev/VulkanSceneGraph #1223: State/MatrixStack::record()
// pushes the projection and modelview mat4s (bytes 0..63 and 64..127) with
// the stage flags of the currently bound pipeline layout whenever a
// pipeline is bound. That is right for the built-in phong/pbr/flat shader
// sets, which declare a 128-byte vertex range, but a fullscreen pass whose
// layout declares only its own fragment push constants also gets the
// matrices: VUID-vkCmdPushConstants-offset-01795, and its own block is
// overwritten. The check `if (pipeline == 0 || stageFlags == 0)` does not
// catch it. The issue proposes a trait on vsg::PipelineLayout saying the
// layout takes the matrix push constants, consulted by MatrixStack::record.
#include "../../common/app.hpp"
using namespace bench;

struct PipelineLayout {
  VkPipelineLayout handle;
  std::vector<VkPushConstantRange> push_constant_ranges;
#ifdef FIXED
  bool matrix_push_constants = false;    // layout reserves bytes 0..127 for projection/modelview
#endif
};

struct GraphicsPipeline { VkPipeline pipeline; PipelineLayout* layout; };

static PipelineLayout* make_layout(App& a, std::vector<VkDescriptorSetLayout> sets, std::vector<VkPushConstantRange> ranges) {
  auto* l = new PipelineLayout{};
  l->push_constant_ranges = ranges;
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.setLayoutCount = (uint32_t)sets.size(); li.pSetLayouts = sets.data();
  li.pushConstantRangeCount = (uint32_t)ranges.size(); li.pPushConstantRanges = ranges.data();
  VK(vkCreatePipelineLayout(a.dev, &li, nullptr, &l->handle));
  return l;
}

static GraphicsPipeline make_pipeline(App& a, VkShaderModule sh, const char* vs, const char* fs, PipelineLayout* layout,
                                      const std::string& nm) {
  VkPipelineShaderStageCreateInfo st[2]{};
  st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  st[0].module = sh; st[0].pName = vs;
  st[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  st[1].module = sh; st[1].pName = fs;
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
  gi.pColorBlendState = &cbs; gi.pDynamicState = &dy; gi.layout = layout->handle;
  GraphicsPipeline p{VK_NULL_HANDLE, layout};
  VK(vkCreateGraphicsPipelines(a.dev, VK_NULL_HANDLE, 1, &gi, nullptr, &p.pipeline));
  a.name(p.pipeline, VK_OBJECT_TYPE_PIPELINE, nm);
  return p;
}

// vsg::State with its projection and modelview matrix stacks
struct State {
  float projection[16]{}, modelview[16]{};
  PipelineLayout* current_layout = nullptr;
  bool dirty = true;

  void bind(VkCommandBuffer c, const GraphicsPipeline& p) {
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipeline);
    current_layout = p.layout;
    dirty = true;
  }
  // MatrixStack::record for both stacks
  void record(VkCommandBuffer c) {
    if (!dirty) return;
    dirty = false;
    VkPipelineLayout pipeline = current_layout ? current_layout->handle : VK_NULL_HANDLE;
    VkShaderStageFlags stageFlags = current_layout && !current_layout->push_constant_ranges.empty()
                                        ? current_layout->push_constant_ranges[0].stageFlags : 0;
    if (pipeline == 0 || stageFlags == 0) return;
#ifdef FIXED
    if (!current_layout->matrix_push_constants) return;
#endif
    vkCmdPushConstants(c, pipeline, stageFlags, 0, 64, projection);
    vkCmdPushConstants(c, pipeline, stageFlags, 64, 64, modelview);
  }
};

struct TonemapParams { float exposure; float gamma; float vignette; float time; };

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  // phong shader set: the standard 128-byte vertex range for the matrices
  PipelineLayout* phong_layout = make_layout(a, {}, {{VK_SHADER_STAGE_VERTEX_BIT, 0, 128}});
#ifdef FIXED
  phong_layout->matrix_push_constants = true;
#endif
  // fullscreen tonemap: its own fragment push constants only
  PipelineLayout* fullscreen_layout = make_layout(a, {}, {{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(TonemapParams)}});
  GraphicsPipeline phong = make_pipeline(a, sh, "vsPhong", "fsPhong", phong_layout, "phong");
  GraphicsPipeline tonemap = make_pipeline(a, sh, "vsFullscreen", "fsTonemap", fullscreen_layout, "fullscreen_tonemap");

  State state;
  for (int i = 0; i < 4; i++) state.projection[i * 5] = state.modelview[i * 5] = 1;
  state.modelview[14] = -3;
  TonemapParams tp{1.0f, 1.0f, 0.3f, 0};

  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    state.bind(c, phong);
    state.record(c);
    vkCmdDraw(c, 3, 1, 0, 0);
    // fullscreen pass on top
    state.bind(c, tonemap);
    tp.time = f * 0.016f;
    vkCmdPushConstants(c, fullscreen_layout->handle, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof tp, &tp);
    state.record(c);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
