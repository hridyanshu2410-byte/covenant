// Fixture for the model driver / trace loader side of the fixed-function
// meaning rules: an HDR10 swapchain, a pipeline with explicit blend factors
// and a static negative-height viewport, a transparent clear, and a draw with
// a dynamic viewport. argv[1]: directory holding yflip.spv.
#include "app.hpp"
using namespace bench;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  sci.surface = a.surface; sci.minImageCount = 3; sci.imageFormat = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
  sci.imageColorSpace = VK_COLOR_SPACE_HDR10_ST2084_EXT; sci.imageExtent = a.extent; sci.imageArrayLayers = 1;
  sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; sci.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  sci.clipped = VK_TRUE; sci.oldSwapchain = a.swapchain;
  VkSwapchainKHR hdr; VK(vkCreateSwapchainKHR(a.dev, &sci, nullptr, &hdr));
  a.name(hdr, VK_OBJECT_TYPE_SWAPCHAIN_KHR, "swapchain_hdr10");

  auto sh = load_shader(a, "yflip.spv");
  Image rt = make_image(a, 640, 480, VK_FORMAT_R8G8B8A8_UNORM,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, "layer_rt");
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  VkPipelineLayout layout; VK(vkCreatePipelineLayout(a.dev, &li, nullptr, &layout));
  VkPipelineShaderStageCreateInfo st{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  st.stage = VK_SHADER_STAGE_VERTEX_BIT; st.module = sh; st.pName = "vsPlain";
  VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkViewport flip{0, 480, 640, -480, 0, 1};
  VkRect2D sc{{0, 0}, a.extent};
  VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vp.viewportCount = 1; vp.pViewports = &flip; vp.scissorCount = 1; vp.pScissors = &sc;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.lineWidth = 1;
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineColorBlendAttachmentState cb{};
  cb.blendEnable = VK_TRUE; cb.colorWriteMask = 0xF;
  cb.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA; cb.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  cb.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; cb.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cbs.attachmentCount = 1; cbs.pAttachments = &cb;
  VkPipelineRenderingCreateInfo rinfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rinfo.colorAttachmentCount = 1; rinfo.pColorAttachmentFormats = &rt.format;
  VkGraphicsPipelineCreateInfo gi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  gi.pNext = &rinfo; gi.stageCount = 1; gi.pStages = &st; gi.pVertexInputState = &vi; gi.pInputAssemblyState = &ia;
  gi.pViewportState = &vp; gi.pRasterizationState = &rs; gi.pMultisampleState = &ms;
  gi.pColorBlendState = &cbs; gi.layout = layout;
  VkPipeline pipe; VK(vkCreateGraphicsPipelines(a.dev, VK_NULL_HANDLE, 1, &gi, nullptr, &pipe));
  a.name(pipe, VK_OBJECT_TYPE_PIPELINE, "static_flip");

  VkCommandBuffer c = alloc_cmd(a, a.gfx_pool); begin(c);
  image_barrier(c, rt.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
  begin_rendering(a, c, {rt.view});                  // cleared to (0, 0, 0, 0), dynamic viewport set
  vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
  vkCmdDraw(c, 3, 1, 0, 0);                          // the static viewport wins
  vkCmdEndRendering(c);
  VK(vkEndCommandBuffer(c)); submit(a.gfx, c, {}, {}, VK_NULL_HANDLE); VK(vkQueueWaitIdle(a.gfx));
}
