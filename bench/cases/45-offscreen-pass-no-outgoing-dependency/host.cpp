// Distilled from derekmwright/glyphengine #109 (fix PR #123): the engine's
// offscreen passes (sky, bloom downsample, ...) are VkRenderPasses whose
// colour attachment ends in SHADER_READ_ONLY_OPTIMAL and is sampled by the
// next pass. Every pass declared only an incoming dependency (EXTERNAL ->
// subpass 0); the final-layout transition therefore ran under the implicit
// outgoing dependency, which ends at BOTTOM_OF_PIPE with no access, so the
// next pass's sampling was not ordered after the attachment writes and the
// transition (SYNC-HAZARD-READ-AFTER-WRITE at vkCmdDraw). The incoming
// dependency also covered only the previous consumers' reads, not attachment
// writes, so the DONT_CARE load (a write) raced the initial-layout transition
// (SYNC-HAZARD-WRITE-AFTER-WRITE at vkCmdBeginRenderPass). The fix derives
// an entry/exit dependency pair for every graphics node: the incoming one
// includes attachment writes, and an outgoing subpass 0 -> EXTERNAL
// dependency makes the attachment writes visible to the consumers.
#include "../../common/app.hpp"
using namespace bench;

static const VkFormat HDR = VK_FORMAT_R16G16B16A16_SFLOAT;

// One colour attachment, DONT_CARE load, final layout SHADER_READ_ONLY.
static VkRenderPass make_offscreen_pass(App& a, const char* nm) {
  VkAttachmentDescription att{};
  att.format = HDR; att.samples = VK_SAMPLE_COUNT_1_BIT;
  att.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; att.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkSubpassDescription sp{};
  sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; sp.colorAttachmentCount = 1; sp.pColorAttachments = &ref;
  VkSubpassDependency deps[2]{};
  // sceneEntryDependency(): incoming
  deps[0].srcSubpass = VK_SUBPASS_EXTERNAL; deps[0].dstSubpass = 0;
#ifdef FIXED
  deps[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
#else
  deps[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
#endif
  deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  // exit dependency to every declared consumer (sampled in a fragment shader)
  deps[1].srcSubpass = 0; deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
  deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  ci.attachmentCount = 1; ci.pAttachments = &att; ci.subpassCount = 1; ci.pSubpasses = &sp;
#ifdef FIXED
  ci.dependencyCount = 2;
#else
  ci.dependencyCount = 1;          // incoming only: the final-layout transition is left to the implicit dependency
#endif
  ci.pDependencies = deps;
  VkRenderPass rp; VK(vkCreateRenderPass(a.dev, &ci, nullptr, &rp));
  a.name(rp, VK_OBJECT_TYPE_RENDER_PASS, nm);
  return rp;
}

static Pipeline make_pass_pipeline(App& a, VkShaderModule sh, const char* fs, VkRenderPass rp,
                                   std::vector<VkDescriptorSetLayout> sets, const char* nm) {
  Pipeline p{};
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.setLayoutCount = (uint32_t)sets.size(); li.pSetLayouts = sets.data();
  VK(vkCreatePipelineLayout(a.dev, &li, nullptr, &p.layout));
  VkPipelineShaderStageCreateInfo st[2]{};
  st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  st[0].module = sh; st[0].pName = "vsFullscreen";
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
  VkGraphicsPipelineCreateInfo gi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  gi.stageCount = 2; gi.pStages = st; gi.pVertexInputState = &vi; gi.pInputAssemblyState = &ia;
  gi.pViewportState = &vp; gi.pRasterizationState = &rs; gi.pMultisampleState = &ms; gi.pDepthStencilState = &ds;
  gi.pColorBlendState = &cbs; gi.pDynamicState = &dy; gi.layout = p.layout; gi.renderPass = rp; gi.subpass = 0;
  VK(vkCreateGraphicsPipelines(a.dev, VK_NULL_HANDLE, 1, &gi, nullptr, &p.pipe));
  a.name(p.pipe, VK_OBJECT_TYPE_PIPELINE, nm);
  return p;
}

static VkFramebuffer make_fb(App& a, VkRenderPass rp, const Image& im) {
  VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  fi.renderPass = rp; fi.attachmentCount = 1; fi.pAttachments = &im.view;
  fi.width = im.w; fi.height = im.h; fi.layers = 1;
  VkFramebuffer fb; VK(vkCreateFramebuffer(a.dev, &fi, nullptr, &fb));
  return fb;
}

static void run_pass(App& a, VkCommandBuffer c, VkRenderPass rp, VkFramebuffer fb, const Image& im,
                     const Pipeline& p, VkDescriptorSet set) {
  VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  bi.renderPass = rp; bi.framebuffer = fb; bi.renderArea = {{0, 0}, {im.w, im.h}};
  vkCmdBeginRenderPass(c, &bi, VK_SUBPASS_CONTENTS_INLINE);
  VkViewport vp{0, 0, (float)im.w, (float)im.h, 0, 1};
  VkRect2D sc{{0, 0}, {im.w, im.h}};
  vkCmdSetViewport(c, 0, 1, &vp);
  vkCmdSetScissor(c, 0, 1, &sc);
  vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipe);
  if (set) vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, p.layout, 0, 1, &set, 0, nullptr);
  vkCmdDraw(c, 3, 1, 0, 0);
  vkCmdEndRenderPass(c);
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  const VkImageUsageFlags RT = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  Image sky = make_image(a, 640, 480, HDR, RT, "sky_color");
  Image bloom = make_image(a, 320, 240, HDR, RT, "bloom_downsample");

  VkRenderPass sky_rp = make_offscreen_pass(a, "sky_pass");
  VkRenderPass bloom_rp = make_offscreen_pass(a, "bloom_pass");
  VkFramebuffer sky_fb = make_fb(a, sky_rp, sky), bloom_fb = make_fb(a, bloom_rp, bloom);

  auto one = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto two = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
                                 {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto bloom_set = alloc_set(a, one, "bloom_input");
  write_image(a, bloom_set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sky.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  auto tonemap_set = alloc_set(a, two, "tonemap_input");
  write_image(a, tonemap_set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sky.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  write_image(a, tonemap_set, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, bloom.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  Pipeline sky_p = make_pass_pipeline(a, sh, "fsSky", sky_rp, {}, "sky");
  Pipeline bloom_p = make_pass_pipeline(a, sh, "fsBloomDownsample", bloom_rp, {one}, "bloom_downsample");
  GfxDesc td; td.vs = td.fs = sh; td.vs_entry = "vsFullscreen"; td.fs_entry = "fsTonemap"; td.sets = {two};
  td.colors = {a.swap_format};
  Pipeline tonemap = make_graphics(a, td, "tonemap");

  run_frames(a, frames_from_env(4), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    run_pass(a, c, sky_rp, sky_fb, sky, sky_p, VK_NULL_HANDLE);
    run_pass(a, c, bloom_rp, bloom_fb, bloom, bloom_p, bloom_set);
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, tonemap.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, tonemap.layout, 0, 1, &tonemap_set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
