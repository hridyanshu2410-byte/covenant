// Distilled from ugai/sldshow2 PR #291 ("fix: auto-scale SDR images on
// Rgba16Float swapchain to restore correct brightness", merged 2026-03-03).
// With HDR output enabled the renderer presents an Rgba16Float swapchain in
// the scRGB (extended linear sRGB) colour space, where 1.0 = 80 nits. SDR
// slides were written with their decoded [0, 1] values unchanged, so SDR
// reference white landed at 80 nits instead of ~203 nits (BT.2408): SDR
// images showed at ~40% of their expected brightness. The fix adds
// sdr_scale_a / sdr_scale_b to TransitionUniform (96 -> 112 bytes) and has
// the shader multiply each SDR texture sample by 203 / 80 ~ 2.54 when the
// renderer is HDR and the texture is not HDR content (image_loader's
// is_hdr_content flag); HDR content and SDR swapchains keep 1.0.
// Port: wgpu Rgba16Float + scRGB surface -> VK_FORMAT_R16G16B16A16_SFLOAT +
// VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT; the uniform keeps only the fields
// the reduced shader reads.
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
  a.name(sc, VK_OBJECT_TYPE_SWAPCHAIN_KHR, "hdr_surface_scrgb");
  uint32_t n = 0; vkGetSwapchainImagesKHR(a.dev, sc, &n, nullptr);
  a.swap_images.resize(n); vkGetSwapchainImagesKHR(a.dev, sc, &n, a.swap_images.data());
  for (auto img : a.swap_images) {
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView v; VK(vkCreateImageView(a.dev, &vi, nullptr, &v)); a.swap_views.push_back(v);
  }
}

struct LoadedImage { Image tex; bool is_hdr_content; };

struct TransitionUniform {
  float blend; int32_t mode; float aspect_ratio[2];
  float bg_color[4];
  int32_t display_mode;
#ifdef FIXED
  float sdr_scale_a, sdr_scale_b, pad;
#else
  float pad[3];
#endif
};

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  // renderer: HDR output available -> Rgba16Float swapchain, scRGB
  recreate_swapchain(a, VK_FORMAT_R16G16B16A16_SFLOAT, VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT);
  const bool renderer_is_hdr = true;
  auto sh = load_shader(a, "shaders.spv");

  // two JPEG slides: 8-bit sRGB, uploaded to sRGB textures (decoded to linear when sampled)
  const VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  LoadedImage slide_a{make_image(a, 64, 64, VK_FORMAT_R8G8B8A8_SRGB, usage, "slide_a_jpeg"), false};
  LoadedImage slide_b{make_image(a, 64, 64, VK_FORMAT_R8G8B8A8_SRGB, usage, "slide_b_jpeg"), false};
  Buffer px = make_buffer(a, 64 * 64 * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "decoded_pixels");
  memset(px.map, 235, 64 * 64 * 4);                       // near-white page
  VkCommandBuffer up = alloc_cmd(a, a.gfx_pool);
  begin(up);
  for (Image* im : {&slide_a.tex, &slide_b.tex}) {
    image_barrier(up, im->img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {64, 64, 1};
    vkCmdCopyBufferToImage(up, px.buf, im->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    image_barrier(up, im->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  }
  VK(vkEndCommandBuffer(up));
  submit(a.gfx, up, {}, {}, VK_NULL_HANDLE);
  VK(vkQueueWaitIdle(a.gfx));

  Buffer ubo = make_buffer(a, sizeof(TransitionUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "transition_uniform");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "transition_bind_group");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo);
  write_image(a, set, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, slide_a.tex.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  write_image(a, set, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, slide_b.tex.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.vs_entry = "vs_main"; gd.fs_entry = "fs_main"; gd.sets = {setl};
  gd.colors = {a.swap_format};
  Pipeline transition = make_graphics(a, gd, "transition_pipeline");

  const int frames = frames_from_env(3);
  run_frames(a, frames, 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, transition.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, transition.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  }, [&](int f) {
    // app.rs: per-frame uniform update
    TransitionUniform u{};
    u.blend = frames > 1 ? (float)f / (frames - 1) : 0.0f; u.mode = 0;
    u.aspect_ratio[0] = u.aspect_ratio[1] = 1.0f;
    u.display_mode = renderer_is_hdr ? 1 : 0;
#ifdef FIXED
    const float kSdrScale = 203.0f / 80.0f;                // BT.2408 reference white on scRGB (1.0 = 80 nits)
    u.sdr_scale_a = (renderer_is_hdr && !slide_a.is_hdr_content) ? kSdrScale : 1.0f;
    u.sdr_scale_b = (renderer_is_hdr && !slide_b.is_hdr_content) ? kSdrScale : 1.0f;
#endif
    memcpy(ubo.map, &u, sizeof u);
  });
  return 0;
}
