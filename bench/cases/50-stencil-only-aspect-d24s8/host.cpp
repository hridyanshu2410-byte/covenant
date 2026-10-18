// Distilled from jpbruyere/vkvg #121 (fix committed by the maintainer):
// a vkvg surface fills paths with the stencil-then-cover technique on a
// VK_FORMAT_D24_UNORM_S8_UINT attachment. The surface's layout transition of
// that image used a subresource range with aspectMask
// VK_IMAGE_ASPECT_STENCIL_BIT only (0x4): "image of format
// VK_FORMAT_D24_UNORM_S8_UINT ... must have the depth and stencil aspects
// set, but its aspectMask is 0x4" (VUID-VkImageMemoryBarrier-image-03320;
// separateDepthStencilLayouts is not enabled). The depth aspect is never
// transitioned, so the image is used as a depth/stencil attachment while
// part of it is still UNDEFINED. The fix sets both aspects.
#include "../../common/app.hpp"
using namespace bench;

static const VkFormat kStencilFormat = VK_FORMAT_D24_UNORM_S8_UINT;

static Pipeline make_fill_pipeline(App& a, VkShaderModule sh, const char* fs, bool cover, const char* nm) {
  Pipeline p{};
  VkPushConstantRange pr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32};
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pr;
  VK(vkCreatePipelineLayout(a.dev, &li, nullptr, &p.layout));
  VkPipelineShaderStageCreateInfo st[2]{};
  st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  st[0].module = sh; st[0].pName = "vsPath";
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
  ds.stencilTestEnable = VK_TRUE;
  VkStencilOpState s{};
  if (cover) {   // draw where the winding count is odd, then reset it
    s.compareOp = VK_COMPARE_OP_NOT_EQUAL; s.failOp = VK_STENCIL_OP_KEEP; s.passOp = VK_STENCIL_OP_ZERO;
  } else {       // even-odd: invert per covering triangle, no colour
    s.compareOp = VK_COMPARE_OP_ALWAYS; s.failOp = VK_STENCIL_OP_KEEP; s.passOp = VK_STENCIL_OP_INVERT;
  }
  s.depthFailOp = VK_STENCIL_OP_KEEP; s.compareMask = 0x1; s.writeMask = 0x1; s.reference = 0;
  ds.front = ds.back = s;
  VkPipelineColorBlendAttachmentState cb{};
  cb.colorWriteMask = cover ? 0xF : 0;
  VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cbs.attachmentCount = 1; cbs.pAttachments = &cb;
  VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dy.dynamicStateCount = 2; dy.pDynamicStates = dyn;
  VkPipelineRenderingCreateInfo rinfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  VkFormat color = VK_FORMAT_B8G8R8A8_UNORM;
  rinfo.colorAttachmentCount = 1; rinfo.pColorAttachmentFormats = &color; rinfo.stencilAttachmentFormat = kStencilFormat;
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
  const uint32_t W = 512, H = 512;
  // vkvg_surface: colour image + stencil image
  Image color = make_image(a, W, H, VK_FORMAT_B8G8R8A8_UNORM,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, "surface_color");
  Image stencil{}; stencil.format = kStencilFormat; stencil.w = W; stencil.h = H; stencil.mips = 1;
  {
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D; ii.format = kStencilFormat; ii.extent = {W, H, 1}; ii.mipLevels = 1; ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VK(vkCreateImage(a.dev, &ii, nullptr, &stencil.img));
    VkMemoryRequirements r; vkGetImageMemoryRequirements(a.dev, stencil.img, &r);
    VkMemoryAllocateInfo mi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mi.allocationSize = r.size; mi.memoryTypeIndex = mem_type(a, r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK(vkAllocateMemory(a.dev, &mi, nullptr, &stencil.mem));
    VK(vkBindImageMemory(a.dev, stencil.img, stencil.mem, 0));
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = stencil.img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = kStencilFormat;
    vi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1};
    VK(vkCreateImageView(a.dev, &vi, nullptr, &stencil.view));
    a.name(stencil.img, VK_OBJECT_TYPE_IMAGE, "surface_stencil");
    a.name(stencil.view, VK_OBJECT_TYPE_IMAGE_VIEW, "surface_stencil");
  }
  Pipeline fill = make_fill_pipeline(a, sh, "fsStencil", false, "fill_stencil");
  Pipeline cover = make_fill_pipeline(a, sh, "fsCover", true, "fill_cover");

  VkCommandBuffer c = alloc_cmd(a, a.gfx_pool);
  VkFence done = make_fence(a, false, "surface_flush");
  for (int f = 0; f < frames_from_env(2); f++) {
    begin(c);
    image_barrier(c, color.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    // _surface_clear / stencil image transition
    {
      VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      b.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      b.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.image = stencil.img;
#ifdef FIXED
      b.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1};
#else
      b.subresourceRange = {VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1};
#endif
      vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                           0, 0, nullptr, 0, nullptr, 1, &b);
    }
    VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    ca.imageView = color.view; ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingAttachmentInfo sa{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    sa.imageView = stencil.view; sa.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    sa.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; sa.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {W, H}}; ri.layerCount = 1; ri.colorAttachmentCount = 1; ri.pColorAttachments = &ca;
    ri.pStencilAttachment = &sa;
    vkCmdBeginRendering(c, &ri);
    VkViewport vp{0, 0, (float)W, (float)H, 0, 1};
    VkRect2D sc{{0, 0}, {W, H}};
    vkCmdSetViewport(c, 0, 1, &vp);
    vkCmdSetScissor(c, 0, 1, &sc);
    float push[8] = {0.2f, 0.4f, 0.8f, 1.0f, 0, 0, 0, 0};
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, fill.pipe);
    vkCmdPushConstants(c, fill.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof push, push);
    vkCmdDraw(c, 15, 1, 0, 0);                  // star polygon as a triangle fan from vertex 0
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, cover.pipe);
    vkCmdPushConstants(c, cover.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof push, push);
    vkCmdDraw(c, 15, 1, 0, 0);
    vkCmdEndRendering(c);
    VK(vkEndCommandBuffer(c));
    submit(a.gfx, c, {}, {}, done);
    VK(vkWaitForFences(a.dev, 1, &done, VK_TRUE, UINT64_MAX));
    VK(vkResetFences(a.dev, 1, &done));
  }
  vkDeviceWaitIdle(a.dev);
}
