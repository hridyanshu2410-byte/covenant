// Distilled from SaschaWillems/Vulkan #1291 (open, fix identified in the
// issue): the multisampling example renders into a 4x multisampled colour
// attachment (loadOp CLEAR, storeOp DONT_CARE) that is resolved into the
// swapchain image, with a 4x depth attachment. The single multisampled
// colour image is shared by every frame. Its render pass declares
// dependencies[1] (EXTERNAL -> 0, colour) with srcStageMask
// COLOR_ATTACHMENT_OUTPUT but srcAccessMask 0, so the next frame's
// vkCmdBeginRenderPass clear (a write) is not ordered against the previous
// frame's attachment writes and storeOp: SYNC-HAZARD-WRITE-AFTER-WRITE
// "vkCmdBeginRenderPass writes to resource, which was previously written at
// the end of subpass 0 by the attachment storeOp". The fix identified in the
// issue: dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT.
#include "../../common/app.hpp"
using namespace bench;

static const VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_4_BIT;
static const VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

static Image make_msaa(App& a, VkFormat fmt, VkImageUsageFlags usage, VkImageAspectFlags aspect, const char* nm) {
  Image im{}; im.format = fmt; im.w = a.extent.width; im.h = a.extent.height; im.mips = 1;
  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.imageType = VK_IMAGE_TYPE_2D; ii.format = fmt; ii.extent = {im.w, im.h, 1}; ii.mipLevels = 1; ii.arrayLayers = 1;
  ii.samples = sampleCount; ii.tiling = VK_IMAGE_TILING_OPTIMAL; ii.usage = usage;
  VK(vkCreateImage(a.dev, &ii, nullptr, &im.img));
  VkMemoryRequirements r; vkGetImageMemoryRequirements(a.dev, im.img, &r);
  VkMemoryAllocateInfo mi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mi.allocationSize = r.size; mi.memoryTypeIndex = mem_type(a, r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK(vkAllocateMemory(a.dev, &mi, nullptr, &im.mem));
  VK(vkBindImageMemory(a.dev, im.img, im.mem, 0));
  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vi.image = im.img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt; vi.subresourceRange = {aspect, 0, 1, 0, 1};
  VK(vkCreateImageView(a.dev, &vi, nullptr, &im.view));
  a.name(im.img, VK_OBJECT_TYPE_IMAGE, nm); a.name(im.view, VK_OBJECT_TYPE_IMAGE_VIEW, nm);
  return im;
}

// VulkanExample::setupRenderPass()
static VkRenderPass setup_render_pass(App& a) {
  VkAttachmentDescription attachments[3]{};
  // Multisampled attachment that we render to
  attachments[0].format = a.swap_format; attachments[0].samples = sampleCount;
  attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  // This is the frame buffer attachment to where the multisampled image will be resolved to
  attachments[1].format = a.swap_format; attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
  attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; attachments[1].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  // Multisampled depth attachment we render to
  attachments[2].format = depthFormat; attachments[2].samples = sampleCount;
  attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; attachments[2].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  VkAttachmentReference colorReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkAttachmentReference resolveReference{1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkAttachmentReference depthReference{2, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1; subpass.pColorAttachments = &colorReference;
  subpass.pResolveAttachments = &resolveReference; subpass.pDepthStencilAttachment = &depthReference;

  VkSubpassDependency dependencies[2]{};
  // Depth attachment
  dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL; dependencies[0].dstSubpass = 0;
  dependencies[0].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  dependencies[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
  // Color attachment
  dependencies[1].srcSubpass = VK_SUBPASS_EXTERNAL; dependencies[1].dstSubpass = 0;
  dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependencies[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
#ifdef FIXED
  dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
#else
  dependencies[1].srcAccessMask = 0;
#endif
  dependencies[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;

  VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  ci.attachmentCount = 3; ci.pAttachments = attachments; ci.subpassCount = 1; ci.pSubpasses = &subpass;
  ci.dependencyCount = 2; ci.pDependencies = dependencies;
  VkRenderPass rp; VK(vkCreateRenderPass(a.dev, &ci, nullptr, &rp));
  a.name(rp, VK_OBJECT_TYPE_RENDER_PASS, "renderPass");
  return rp;
}

static Pipeline make_msaa_pipeline(App& a, VkShaderModule sh, VkRenderPass rp, VkDescriptorSetLayout setl) {
  Pipeline p{};
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.setLayoutCount = 1; li.pSetLayouts = &setl;
  VK(vkCreatePipelineLayout(a.dev, &li, nullptr, &p.layout));
  VkPipelineShaderStageCreateInfo st[2]{};
  st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[0].stage = VK_SHADER_STAGE_VERTEX_BIT; st[0].module = sh; st[0].pName = "vsMain";
  st[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module = sh; st[1].pName = "fsMain";
  VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vp.viewportCount = 1; vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.lineWidth = 1; rs.cullMode = VK_CULL_MODE_NONE;
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = sampleCount;
  VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  ds.depthTestEnable = ds.depthWriteEnable = VK_TRUE; ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
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
  a.name(p.pipe, VK_OBJECT_TYPE_PIPELINE, "pipelines.MSAA");
  return p;
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  // setupMultisampleTarget(): one colour and one depth target shared by all frames
  Image msColor = make_msaa(a, a.swap_format, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                            VK_IMAGE_ASPECT_COLOR_BIT, "multisampleTarget.color");
  Image msDepth = make_msaa(a, depthFormat, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                            VK_IMAGE_ASPECT_DEPTH_BIT, "multisampleTarget.depth");
  VkRenderPass renderPass = setup_render_pass(a);
  std::vector<VkFramebuffer> frameBuffers(a.swap_images.size());
  for (size_t i = 0; i < frameBuffers.size(); i++) {          // setupFrameBuffer()
    VkImageView att[3] = {msColor.view, a.swap_views[i], msDepth.view};
    VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fi.renderPass = renderPass; fi.attachmentCount = 3; fi.pAttachments = att;
    fi.width = a.extent.width; fi.height = a.extent.height; fi.layers = 1;
    VK(vkCreateFramebuffer(a.dev, &fi, nullptr, &frameBuffers[i]));
  }
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}});
  const int MAX_CONCURRENT_FRAMES = 2;
  std::vector<Buffer> ubo; std::vector<VkDescriptorSet> sets;
  for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    ubo.push_back(make_buffer(a, 80, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "uniformBuffers[" + std::to_string(i) + "]"));
    sets.push_back(alloc_set(a, setl, "descriptorSets[" + std::to_string(i) + "]"));
    write_buffer(a, sets[i], 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo[i]);
  }
  Pipeline msaa = make_msaa_pipeline(a, sh, renderPass, setl);

  std::vector<VkCommandBuffer> drawCmdBuffers(MAX_CONCURRENT_FRAMES);
  std::vector<VkSemaphore> presentComplete(MAX_CONCURRENT_FRAMES), renderComplete(a.swap_images.size());
  std::vector<VkFence> waitFences(MAX_CONCURRENT_FRAMES);
  for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    drawCmdBuffers[i] = alloc_cmd(a, a.gfx_pool);
    presentComplete[i] = make_semaphore(a, "presentCompleteSemaphores[" + std::to_string(i) + "]");
    waitFences[i] = make_fence(a, true, "waitFences[" + std::to_string(i) + "]");
  }
  for (size_t i = 0; i < renderComplete.size(); i++)
    renderComplete[i] = make_semaphore(a, "renderCompleteSemaphores[" + std::to_string(i) + "]");

  int frames = frames_from_env(6);
  for (int f = 0; f < frames; f++) {
    int k = f % MAX_CONCURRENT_FRAMES;
    VK(vkWaitForFences(a.dev, 1, &waitFences[k], VK_TRUE, UINT64_MAX));
    VK(vkResetFences(a.dev, 1, &waitFences[k]));
    uint32_t idx;
    VK(vkAcquireNextImageKHR(a.dev, a.swapchain, UINT64_MAX, presentComplete[k], VK_NULL_HANDLE, &idx));
    float u[20] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0.3f, 0.5f, -1.0f, 0};
    memcpy(ubo[k].map, u, sizeof u);
    VkCommandBuffer c = drawCmdBuffers[k];
    begin(c);
    VkClearValue clear[3]{}; clear[0].color = {{0.25f, 0.25f, 0.25f, 1.0f}}; clear[2].depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    bi.renderPass = renderPass; bi.framebuffer = frameBuffers[idx]; bi.renderArea = {{0, 0}, a.extent};
    bi.clearValueCount = 3; bi.pClearValues = clear;
    vkCmdBeginRenderPass(c, &bi, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport vp{0, 0, (float)a.extent.width, (float)a.extent.height, 0, 1};
    VkRect2D sc{{0, 0}, a.extent};
    vkCmdSetViewport(c, 0, 1, &vp); vkCmdSetScissor(c, 0, 1, &sc);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, msaa.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, msaa.layout, 0, 1, &sets[k], 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRenderPass(c);
    VK(vkEndCommandBuffer(c));
    submit(a.gfx, c, {{presentComplete[k], VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT}}, {renderComplete[idx]}, waitFences[k]);
    present(a, a.gfx, idx, renderComplete[idx]);
  }
  vkDeviceWaitIdle(a.dev);
}
