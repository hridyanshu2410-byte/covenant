// Distilled from slint-ui/slint #7850 (fixed by correcting the typo): Slint's
// Skia Vulkan surface wraps each swapchain image as a Skia render target
// built from the window size, and the size code read
//     let width: i32 = size.width ...;  let height: i32 = width ...;
// so the render target, and the VkFramebuffer Skia creates for it, were
// square. On an 875x700 window: "pAttachments[0] mip level 0 has height
// (700) smaller than the corresponding framebuffer height (875)"
// (VUID-VkFramebufferCreateInfo-flags-04534), followed by QueueSubmit
// failing with VK_ERROR_DEVICE_LOST (-4) on Wayland + KDE + NVIDIA. The fix
// takes the height from size.height.
#include "../../common/app.hpp"
using namespace bench;

struct PhysicalWindowSize { uint32_t width, height; };

int main(int argc, char** argv) {
  AppDesc d; d.width = 875; d.height = 700; d.swap_format = VK_FORMAT_B8G8R8A8_UNORM;
  App a = make_app(argc, argv, d);
  auto sh = load_shader(a, "shaders.spv");
  PhysicalWindowSize size{a.extent.width, a.extent.height};

  // Skia's render pass for a wrapped swapchain image (layouts handled around it)
  VkAttachmentDescription att{};
  att.format = a.swap_format; att.samples = VK_SAMPLE_COUNT_1_BIT;
  att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  att.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; att.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkSubpassDescription sp{};
  sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; sp.colorAttachmentCount = 1; sp.pColorAttachments = &ref;
  VkSubpassDependency dep{};
  dep.srcSubpass = VK_SUBPASS_EXTERNAL; dep.dstSubpass = 0;
  dep.srcStageMask = dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dep.srcAccessMask = dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  rpi.attachmentCount = 1; rpi.pAttachments = &att; rpi.subpassCount = 1; rpi.pSubpasses = &sp;
  rpi.dependencyCount = 1; rpi.pDependencies = &dep;
  VkRenderPass rp; VK(vkCreateRenderPass(a.dev, &rpi, nullptr, &rp));
  a.name(rp, VK_OBJECT_TYPE_RENDER_PASS, "skia_render_pass");

  // skia_vulkan_surface.rs: render target dimensions for the wrapped swapchain image
  int32_t width = (int32_t)size.width;
#ifdef FIXED
  int32_t height = (int32_t)size.height;
#else
  int32_t height = width;
#endif
  std::vector<VkFramebuffer> fbs;
  for (size_t i = 0; i < a.swap_views.size(); i++) {
    VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fi.renderPass = rp; fi.attachmentCount = 1; fi.pAttachments = &a.swap_views[i];
    fi.width = (uint32_t)width; fi.height = (uint32_t)height; fi.layers = 1;
    VkFramebuffer fb; VK(vkCreateFramebuffer(a.dev, &fi, nullptr, &fb));
    a.name(fb, VK_OBJECT_TYPE_FRAMEBUFFER, "skia_framebuffer[" + std::to_string(i) + "]");
    fbs.push_back(fb);
  }

  // Skia's fill pipeline for the window background and a button
  Pipeline p{};
  {
    VkPushConstantRange pr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pr;
    VK(vkCreatePipelineLayout(a.dev, &li, nullptr, &p.layout));
    VkPipelineShaderStageCreateInfo st[2]{};
    st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    st[0].module = sh; st[0].pName = "vsRect";
    st[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    st[1].module = sh; st[1].pName = "fsRect";
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
    gi.pColorBlendState = &cbs; gi.pDynamicState = &dy; gi.layout = p.layout; gi.renderPass = rp;
    VK(vkCreateGraphicsPipelines(a.dev, VK_NULL_HANDLE, 1, &gi, nullptr, &p.pipe));
    a.name(p.pipe, VK_OBJECT_TYPE_PIPELINE, "skia_fill_rect");
  }

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    VkClearValue clear{}; clear.color = {{0.95f, 0.95f, 0.95f, 1.0f}};
    VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    bi.renderPass = rp; bi.framebuffer = fbs[idx];
    bi.renderArea = {{0, 0}, {(uint32_t)width, (uint32_t)height}};
    bi.clearValueCount = 1; bi.pClearValues = &clear;
    vkCmdBeginRenderPass(c, &bi, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport vp{0, 0, (float)width, (float)height, 0, 1};
    VkRect2D sc{{0, 0}, {(uint32_t)width, (uint32_t)height}};
    vkCmdSetViewport(c, 0, 1, &vp);
    vkCmdSetScissor(c, 0, 1, &sc);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipe);
    float button[8] = {0.4f, 0.45f, 0.6f, 0.55f, 0.2f, 0.45f, 0.85f, 1.0f};   // rect (ndc), colour
    vkCmdPushConstants(c, p.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof button, button);
    vkCmdDraw(c, 6, 1, 0, 0);
    vkCmdEndRenderPass(c);
  });
}
