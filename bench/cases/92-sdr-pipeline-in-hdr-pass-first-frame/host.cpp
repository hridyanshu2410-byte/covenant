// Distilled from libretro/RetroArch PR #18743 ("Vulkan: create the HDR
// offscreen buffer at init", merged 2026-02-17). With HDR enabled, the
// Vulkan driver renders the frame and menu into an SDR offscreen buffer and
// an HDR pass converts that into the A2B10G10R10 / HDR10 swapchain. The
// offscreen buffer was only created by the end-of-frame resize handling
// (vulkan_check_swapchain -> vulkan_init_render_target), so on the very
// first frame there was none and the menu was drawn straight into the
// swapchain render pass with the menu pipelines, which are built for the
// SDR B8G8R8A8_UNORM offscreen format. Validation reported vkCmdDraw with a
// pipeline whose render pass / attachment formats are incompatible with the
// active one (seen on MoltenVK, where the first frame was also corrupted).
// The fix creates the offscreen buffer in vulkan_init when HDR is enabled,
// so every frame goes menu -> offscreen -> HDR pass.
// Port: dynamic rendering instead of VkRenderPass; the HDR10 swapchain is
// recreated as in case 67.
#include "../../common/app.hpp"
using namespace bench;

static void recreate_swapchain(App& a, VkFormat fmt, VkColorSpaceKHR cs) {
  for (auto v : a.swap_views) vkDestroyImageView(a.dev, v, nullptr);
  a.swap_views.clear();
  VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  sci.surface = a.surface; sci.minImageCount = 3; sci.imageFormat = fmt; sci.imageColorSpace = cs;
  sci.imageExtent = a.extent; sci.imageArrayLayers = 1; sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  sci.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR; sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  sci.presentMode = VK_PRESENT_MODE_FIFO_KHR; sci.clipped = VK_TRUE; sci.oldSwapchain = a.swapchain;
  VkSwapchainKHR sc; VK(vkCreateSwapchainKHR(a.dev, &sci, nullptr, &sc));
  vkDestroySwapchainKHR(a.dev, a.swapchain, nullptr);
  a.swapchain = sc; a.swap_format = fmt;
  a.name(sc, VK_OBJECT_TYPE_SWAPCHAIN_KHR, "hdr10_swapchain");
  uint32_t n = 0; vkGetSwapchainImagesKHR(a.dev, sc, &n, nullptr);
  a.swap_images.resize(n); vkGetSwapchainImagesKHR(a.dev, sc, &n, a.swap_images.data());
  for (auto img : a.swap_images) {
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView v; VK(vkCreateImageView(a.dev, &vi, nullptr, &v)); a.swap_views.push_back(v);
  }
}

struct MenuPush { float rect[4]; float color[4]; };

struct VkData {
  bool hdr_enable = true;
  bool offscreen_valid = false;
  Image offscreen{};
  VkDescriptorSet hdr_set = VK_NULL_HANDLE;
  bool offscreen_first_use = true;
};

constexpr VkFormat kOffscreenFormat = VK_FORMAT_B8G8R8A8_UNORM;

static void vulkan_init_render_target(App& a, VkData& vk, VkDescriptorSetLayout hdr_setl) {
  vk.offscreen = make_image(a, a.extent.width, a.extent.height, kOffscreenFormat,
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, "hdr_offscreen_buffer");
  vk.hdr_set = alloc_set(a, hdr_setl, "hdr_pass_input");
  write_image(a, vk.hdr_set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, vk.offscreen.view,
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  vk.offscreen_valid = true;
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  recreate_swapchain(a, VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_HDR10_ST2084_EXT);
  auto sh = load_shader(a, "shaders.spv");
  VkData vk;

  auto hdr_setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  GfxDesc md; md.vs = md.fs = sh; md.fs_entry = "fsMenu"; md.colors = {kOffscreenFormat}; md.push_size = sizeof(MenuPush);
  Pipeline menu_pipe = make_graphics(a, md, "menu_pipeline_sdr");
  GfxDesc hd; hd.vs = hd.fs = sh; hd.fs_entry = "fsHdr"; hd.sets = {hdr_setl}; hd.colors = {a.swap_format};
  Pipeline hdr_pipe = make_graphics(a, hd, "hdr_pipeline");

#ifdef FIXED
  if (vk.hdr_enable) vulkan_init_render_target(a, vk, hdr_setl);      // vulkan_init
#endif

  MenuPush mp{{0.1f, 0.1f, 0.6f, 0.4f}, {0.85f, 0.85f, 0.85f, 1}};
  run_frames(a, frames_from_env(4), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    if (vk.hdr_enable && vk.offscreen_valid) {
      image_barrier(c, vk.offscreen.img,
                    vk.offscreen_first_use ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
      vk.offscreen_first_use = false;
      begin_rendering(a, c, {vk.offscreen.view});
      vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, menu_pipe.pipe);
      vkCmdPushConstants(c, menu_pipe.layout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, sizeof mp, &mp);
      vkCmdDraw(c, 3, 1, 0, 0);
      vkCmdEndRendering(c);
      image_barrier(c, vk.offscreen.img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
      begin_rendering(a, c, {a.swap_views[idx]});
      vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, hdr_pipe.pipe);
      vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, hdr_pipe.layout, 0, 1, &vk.hdr_set, 0, nullptr);
      vkCmdDraw(c, 3, 1, 0, 0);
      vkCmdEndRendering(c);
    } else {
      // no offscreen buffer yet: the menu goes straight into the swapchain pass
      begin_rendering(a, c, {a.swap_views[idx]});
      vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, menu_pipe.pipe);
      vkCmdPushConstants(c, menu_pipe.layout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, sizeof mp, &mp);
      vkCmdDraw(c, 3, 1, 0, 0);
      vkCmdEndRendering(c);
    }
    // end of frame: vulkan_check_swapchain notices the missing render target
    if (vk.hdr_enable && !vk.offscreen_valid) vulkan_init_render_target(a, vk, hdr_setl);
  });
  return 0;
}
