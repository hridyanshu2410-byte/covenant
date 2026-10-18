// Distilled from crosire/reshade PR #441: ReShade draws its ImGui overlay
// on HDR10 swapchains with imgui_hdr.hlsl, which converts the sRGB UI colour
// to linear, applies a BT.709 -> BT.2020 primaries matrix and PQ-encodes.
// The shader is HLSL; for Vulkan it is compiled as GLSL through a
// compatibility header defining mul(a, b) as (a) * (b). The matrix
// coefficients are listed row by row, which GLSL's mat3 constructor reads as
// columns, so the Vulkan path computed M^T * col instead of M * col. White is
// no longer preserved: neutral greys come out tinted green (0.71, 1.34,
// 0.95 times grey). The fix defines mul(a, b) as (b) * (a), turning the
// SPIR-V OpMatrixTimesVector into OpVectorTimesMatrix. D3D and OpenGL paths
// were unaffected. Here the HLSL source is Slang; the buggy variant states the
// product the shim produced.
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
  a.name(sc, VK_OBJECT_TYPE_SWAPCHAIN_KHR, "game_swapchain_hdr10");
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
struct ImGuiPush { float proj[16]; float hdr_white_nits; float pad[3]; };

int main(int argc, char** argv) {
  AppDesc d; d.swap_format = VK_FORMAT_B8G8R8A8_UNORM;
  App a = make_app(argc, argv, d);
  // the game switched its swapchain to HDR10 PQ; ReShade picks imgui_hdr for it
  recreate_swapchain(a, VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_HDR10_ST2084_EXT);
  auto sh = load_shader(a, "shaders.spv");

  Image font = make_image(a, 256, 64, VK_FORMAT_R8G8B8A8_UNORM,
                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, "imgui_font_atlas");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "imgui_font_set");
  write_image(a, set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, font.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  // one window background quad, ImGui's default grey (ImGuiCol_WindowBg-like, opaque)
  Buffer vb = make_buffer(a, 6 * sizeof(ImDrawVert), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "imgui_vertices");
  const uint32_t grey = 0xFF7F7F7Fu;  // ABGR8: (127, 127, 127, 255)
  ImDrawVert q[6] = {{{20, 20}, {0, 0}, grey}, {{420, 20}, {0, 0}, grey}, {{20, 320}, {0, 0}, grey},
                     {{420, 20}, {0, 0}, grey}, {{420, 320}, {0, 0}, grey}, {{20, 320}, {0, 0}, grey}};
  memcpy(vb.map, q, sizeof q);

  GfxDesc gd; gd.vs = gd.fs = sh; gd.vs_entry = "vsImGui"; gd.fs_entry = "fsImGui"; gd.sets = {setl};
  gd.colors = {a.swap_format}; gd.push_size = sizeof(ImGuiPush); gd.vertex_stride = sizeof(ImDrawVert);
  gd.attributes = {{0, 0, VK_FORMAT_R32G32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32_SFLOAT, 8},
                   {2, 0, VK_FORMAT_R8G8B8A8_UNORM, 16}};
  Pipeline imgui = make_graphics(a, gd, "imgui_hdr");
  ImGuiPush push{};
  float L = 0, R = 640, T = 0, B = 480;   // ImGui orthographic projection
  float m[16] = {2 / (R - L), 0, 0, 0, 0, 2 / (B - T), 0, 0, 0, 0, 0.5f, 0, (R + L) / (L - R), (T + B) / (T - B), 0.5f, 1};
  memcpy(push.proj, m, sizeof m);
  push.hdr_white_nits = 203.0f;

  VkCommandBuffer up = alloc_cmd(a, a.gfx_pool); begin(up);
  image_barrier(up, font.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
  VK(vkEndCommandBuffer(up)); submit(a.gfx, up, {}, {}, VK_NULL_HANDLE); VK(vkQueueWaitIdle(a.gfx));

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, imgui.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, imgui.layout, 0, 1, &set, 0, nullptr);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &off);
    vkCmdPushConstants(c, imgui.layout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, sizeof push, &push);
    vkCmdDraw(c, 6, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
