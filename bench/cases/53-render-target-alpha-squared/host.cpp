// Distilled from Forge-Game-Engine/Forge PR #685: sprites are drawn into a
// transparent render target, and a present pass then layers that target
// over the screen. Both passes used blendFunc(SRC_ALPHA,
// ONE_MINUS_SRC_ALPHA) for colour and alpha. Into a target cleared to
// transparent, that stores premultiplied colour (rgb * a) but alpha
// squared (a * a); the present pass then multiplied the colour by that
// alpha again. A 50% sprite reached the screen at about 25% or less. The fix
// uses blendFuncSeparate(SRC_ALPHA, ONE_MINUS_SRC_ALPHA, ONE,
// ONE_MINUS_SRC_ALPHA) for the sprite pass, so the target holds correctly
// premultiplied colour and alpha, and blendFunc(ONE, ONE_MINUS_SRC_ALPHA) to
// layer the premultiplied target in the present pass (clear colours are
// premultiplied too). WebGL in the original; blend state as pipeline state here.
#include "../../common/app.hpp"
using namespace bench;

struct Blend { VkBlendFactor src_c, dst_c, src_a, dst_a; };

static Pipeline make_blended(App& a, VkShaderModule sh, const char* vs, const char* fs, VkFormat color,
                             std::vector<VkDescriptorSetLayout> sets, Blend bl, const char* nm) {
  Pipeline p{};
  VkPushConstantRange pr{VK_SHADER_STAGE_ALL_GRAPHICS, 0, 32};
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.setLayoutCount = (uint32_t)sets.size(); li.pSetLayouts = sets.data();
  li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pr;
  VK(vkCreatePipelineLayout(a.dev, &li, nullptr, &p.layout));
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
  VkPipelineColorBlendAttachmentState cb{};
  cb.blendEnable = VK_TRUE; cb.colorWriteMask = 0xF;
  cb.srcColorBlendFactor = bl.src_c; cb.dstColorBlendFactor = bl.dst_c; cb.colorBlendOp = VK_BLEND_OP_ADD;
  cb.srcAlphaBlendFactor = bl.src_a; cb.dstAlphaBlendFactor = bl.dst_a; cb.alphaBlendOp = VK_BLEND_OP_ADD;
  VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cbs.attachmentCount = 1; cbs.pAttachments = &cb;
  VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dy.dynamicStateCount = 2; dy.pDynamicStates = dyn;
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

static void begin_target(App& a, VkCommandBuffer c, VkImageView v, VkClearColorValue clear) {
  VkRenderingAttachmentInfo r{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  r.imageView = v; r.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  r.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; r.storeOp = VK_ATTACHMENT_STORE_OP_STORE; r.clearValue.color = clear;
  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = {{0, 0}, a.extent}; ri.layerCount = 1; ri.colorAttachmentCount = 1; ri.pColorAttachments = &r;
  vkCmdBeginRendering(c, &ri);
  VkViewport vp{0, 0, (float)a.extent.width, (float)a.extent.height, 0, 1};
  VkRect2D sc{{0, 0}, a.extent};
  vkCmdSetViewport(c, 0, 1, &vp);
  vkCmdSetScissor(c, 0, 1, &sc);
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  Image sprite_tex = make_image(a, 64, 64, VK_FORMAT_R8G8B8A8_SRGB,
                                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, "sprite_texture");
  Image layer_rt = make_image(a, 640, 480, VK_FORMAT_R8G8B8A8_UNORM,
                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, "sprite_layer_target");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto sprite_set = alloc_set(a, setl, "sprite_material");
  write_image(a, sprite_set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sprite_tex.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  auto layer_set = alloc_set(a, setl, "present_layer");
  write_image(a, layer_set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, layer_rt.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  const VkBlendFactor SA = VK_BLEND_FACTOR_SRC_ALPHA, ISA = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, ONE = VK_BLEND_FACTOR_ONE;
#ifdef FIXED
  Blend sprite_blend{SA, ISA, ONE, ISA};        // blendFuncSeparate(SRC_ALPHA, ONE_MINUS_SRC_ALPHA, ONE, ONE_MINUS_SRC_ALPHA)
  Blend present_blend{ONE, ISA, ONE, ISA};      // blendFunc(ONE, ONE_MINUS_SRC_ALPHA)
#else
  Blend sprite_blend{SA, ISA, SA, ISA};         // blendFunc(SRC_ALPHA, ONE_MINUS_SRC_ALPHA)
  Blend present_blend{SA, ISA, SA, ISA};
#endif
  Pipeline sprite = make_blended(a, sh, "vsSprite", "fsSprite", VK_FORMAT_R8G8B8A8_UNORM, {setl}, sprite_blend, "sprite_batch");
  Pipeline present_layer = make_blended(a, sh, "vsFullscreen", "fsLayer", a.swap_format, {setl}, present_blend, "present_layer");

  VkCommandBuffer up = alloc_cmd(a, a.gfx_pool); begin(up);
  image_barrier(up, sprite_tex.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
  VK(vkEndCommandBuffer(up)); submit(a.gfx, up, {}, {}, VK_NULL_HANDLE); VK(vkQueueWaitIdle(a.gfx));

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    image_barrier(c, layer_rt.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    begin_target(a, c, layer_rt.view, {{0, 0, 0, 0}});          // transparent target
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, sprite.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, sprite.layout, 0, 1, &sprite_set, 0, nullptr);
    float tint[8] = {1, 1, 1, 0.5f, 0.25f, 0.25f, 0.5f, 0.5f};     // 50% opacity sprite, rect
    vkCmdPushConstants(c, sprite.layout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, sizeof tint, tint);
    vkCmdDraw(c, 6, 1, 0, 0);
    vkCmdEndRendering(c);
    image_barrier(c, layer_rt.img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    begin_target(a, c, a.swap_views[idx], {{0.2f, 0.3f, 0.4f, 1.0f}});   // opaque scene background
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, present_layer.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, present_layer.layout, 0, 1, &layer_set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
