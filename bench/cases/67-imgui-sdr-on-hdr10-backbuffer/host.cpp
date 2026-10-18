// Distilled from Fexty12573/SharpPluginLoader PR #73 ("Fix ImGui rendering on
// HDR swap chains"): when the game runs in HDR its D3D12 swap chain is
// R10G10B10A2_UNORM in the HDR10 (ST.2084 / BT.2020) colour space, but
// ImGui_ImplDX12_Init was always given R8G8B8A8_UNORM, so the ImGui pipeline
// was built for an 8-bit SDR target and its pixel shader wrote sRGB-encoded
// SDR colours straight into the PQ-encoded back buffer ("extremely bright and
// blue"). The fix reads the back buffer's format, infers the ImGui colour
// space from it (infer_imgui_color_space) and passes both to
// ImGui_ImplDX12_Init; for HDR10 the pixel shader decodes sRGB, converts
// BT.709 -> BT.2020 and PQ-encodes with a 203-nit graphics white.
// Port: DXGI_FORMAT_R10G10B10A2_UNORM + DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020
// -> VK_FORMAT_A2B10G10R10_UNORM_PACK32 + VK_COLOR_SPACE_HDR10_ST2084_EXT;
// the shader variant chosen by IMGUI_COLOR_SPACE is selected at build time
// (FIXED = colour space 2).
#include "../../common/app.hpp"
using namespace bench;

// the game's swap chain, created in HDR mode
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
  a.name(sc, VK_OBJECT_TYPE_SWAPCHAIN_KHR, "game_swap_chain_hdr10");
  uint32_t n = 0; vkGetSwapchainImagesKHR(a.dev, sc, &n, nullptr);
  a.swap_images.resize(n); vkGetSwapchainImagesKHR(a.dev, sc, &n, a.swap_images.data());
  for (auto img : a.swap_images) {
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView v; VK(vkCreateImageView(a.dev, &vi, nullptr, &v)); a.swap_views.push_back(v);
  }
}

struct ImDrawVert { float pos[2]; float uv[2]; uint32_t col; };

int main(int argc, char** argv) {
  AppDesc d; d.swap_format = VK_FORMAT_B8G8R8A8_UNORM;
  App a = make_app(argc, argv, d);
  recreate_swapchain(a, VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_HDR10_ST2084_EXT);
  auto sh = load_shader(a, "shaders.spv");

  // D3DModule: back_buffer_format = swap_chain->GetBuffer(0)->GetDesc().Format
  VkFormat back_buffer_format = a.swap_format;
#ifdef FIXED
  VkFormat rtv_format = back_buffer_format;             // ImGui_ImplDX12_Init(..., back_buffer_format, ..., imgui_color_space)
#else
  VkFormat rtv_format = VK_FORMAT_R8G8B8A8_UNORM;       // ImGui_ImplDX12_Init(..., DXGI_FORMAT_R8G8B8A8_UNORM, ...)
#endif
  (void)back_buffer_format;

  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {rtv_format}; gd.push_size = 64;
  gd.vertex_stride = sizeof(ImDrawVert);
  gd.attributes = {{0, 0, VK_FORMAT_R32G32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32_SFLOAT, 8}, {2, 0, VK_FORMAT_R8G8B8A8_UNORM, 16}};
  Pipeline imgui = make_graphics(a, gd, "imgui_dx12_pipeline");

  Image font = make_image(a, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, "imgui_font_texture");
  VkCommandBuffer up = alloc_cmd(a, a.gfx_pool);
  begin(up);
  image_barrier(up, font.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
  VkClearColorValue white{{1, 1, 1, 1}}; VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdClearColorImage(up, font.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &white, 1, &rng);
  image_barrier(up, font.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
  VK(vkEndCommandBuffer(up));
  submit(a.gfx, up, {}, {}, VK_NULL_HANDLE);
  VK(vkQueueWaitIdle(a.gfx));
  auto font_set = alloc_set(a, setl, "imgui_srv");
  write_image(a, font_set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, font.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  // an ImGui window (World Tuning Tool): grey background
  Buffer vbuf = make_buffer(a, 6 * sizeof(ImDrawVert), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "imgui_vertex_buffer");
  ImDrawVert quad[6] = {{{50, 50}, {0, 0}, 0xf0242424u}, {{450, 50}, {0, 0}, 0xf0242424u}, {{50, 350}, {0, 0}, 0xf0242424u},
                        {{450, 50}, {0, 0}, 0xf0242424u}, {{450, 350}, {0, 0}, 0xf0242424u}, {{50, 350}, {0, 0}, 0xf0242424u}};
  memcpy(vbuf.map, quad, sizeof quad);
  float L = 0, R = (float)a.extent.width, T = 0, B = (float)a.extent.height;
  float mvp[4][4] = {{2.0f / (R - L), 0, 0, 0}, {0, 2.0f / (B - T), 0, 0}, {0, 0, 0.5f, 0},
                     {(R + L) / (L - R), (T + B) / (T - B), 0.5f, 1.0f}};

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    // the game's frame is already in the back buffer; ImGui draws on top
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, imgui.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, imgui.layout, 0, 1, &font_set, 0, nullptr);
    vkCmdPushConstants(c, imgui.layout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, 64, mvp);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(c, 0, 1, &vbuf.buf, &off);
    vkCmdDraw(c, 6, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
