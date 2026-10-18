// Distilled from azahar-emu/azahar PR #2170 ("renderer_vulkan: Fix screenshot
// color channels on RGBA swapchain formats", merged 2026-07-02).
// RendererVulkan::RenderScreenshotWithStagingCopy() renders the screens into a
// Frame created with the present window's surface format, copies the image
// into a host-visible staging buffer and memcpy()s it into
// settings.screenshot_bits -- the bits of a QImage::Format_RGB32, whose bytes
// are B, G, R, A. That is only correct when the swapchain format is
// B8G8R8A8; on surfaces where Azahar picks R8G8B8A8_UNORM (common on Android
// and some Linux drivers) every screenshot had red and blue swapped. The fix
// swaps R and B after the memcpy when the surface format is eR8G8B8A8Unorm
// (in both the staging-copy and the host-memory screenshot paths).
// Port: vulkan-hpp/VMA/scheduler -> C API; the emulated screen texture is a
// generated test card, the QImage is a byte vector written as a 32-bit BMP
// (also B, G, R, A byte order) next to the shaders.
#include "../../common/app.hpp"
using namespace bench;

struct PresentUniformData { float modelview[16]; float i_resolution[4]; float o_resolution[4]; };
struct ScreenRectVertex { float position[2]; float tex_coord[2]; };

static void write_bmp(const std::string& path, const uint8_t* bgra, uint32_t w, uint32_t h) {
  uint8_t hdr[54] = {'B', 'M'};
  uint32_t size = 54 + w * h * 4, off = 54, ih = 40, planes_bpp = 1 | (32 << 16);
  int32_t neg_h = -(int32_t)h;   // top-down rows, as QImage stores them
  memcpy(hdr + 2, &size, 4); memcpy(hdr + 10, &off, 4); memcpy(hdr + 14, &ih, 4);
  memcpy(hdr + 18, &w, 4); memcpy(hdr + 22, &neg_h, 4); memcpy(hdr + 26, &planes_bpp, 4);
  if (FILE* fp = fopen(path.c_str(), "wb")) { fwrite(hdr, 1, 54, fp); fwrite(bgra, 1, (size_t)w * h * 4, fp); fclose(fp); }
}

int main(int argc, char** argv) {
  AppDesc d; d.swap_format = VK_FORMAT_R8G8B8A8_UNORM;    // surface format chosen by the swapchain
  App a = make_app(argc, argv, d);
  auto sh = load_shader(a, "shaders.spv");
  const VkFormat surface_format = a.swap_format;          // main_present_window.GetSurfaceFormat()

  // emulated top screen (400x240), RGBA8: colour bars
  const uint32_t SW = 400, SH = 240;
  Image screen = make_image(a, SW, SH, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                            "top_screen");
  Buffer upload = make_buffer(a, SW * SH * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "screen_upload");
  const uint8_t bars[6][3] = {{255, 0, 0}, {255, 160, 0}, {255, 255, 0}, {0, 200, 0}, {0, 80, 255}, {160, 0, 200}};
  for (uint32_t y = 0; y < SH; y++)
    for (uint32_t x = 0; x < SW; x++) {
      uint8_t* p = (uint8_t*)upload.map + (y * SW + x) * 4; const uint8_t* b = bars[x * 6 / SW];
      p[0] = b[0]; p[1] = b[1]; p[2] = b[2]; p[3] = 255;
    }
  {
    VkCommandBuffer c = alloc_cmd(a, a.gfx_pool); begin(c);
    image_barrier(c, screen.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {SW, SH, 1};
    vkCmdCopyBufferToImage(c, upload.buf, screen.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    image_barrier(c, screen.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    VK(vkEndCommandBuffer(c)); submit(a.gfx, c, {}, {}, VK_NULL_HANDLE); vkQueueWaitIdle(a.gfx);
  }

  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "present_set");
  write_image(a, set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, screen.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.vs_entry = "vertMain"; gd.fs_entry = "fragMain";
  gd.sets = {setl}; gd.colors = {surface_format}; gd.vertex_stride = sizeof(ScreenRectVertex);
  gd.attributes = {{0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(ScreenRectVertex, position)},
                   {1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(ScreenRectVertex, tex_coord)}};
  gd.push_size = sizeof(PresentUniformData);
  Pipeline present = make_graphics(a, gd, "present_pipeline");

  // screen rect for the framebuffer layout (top screen scaled to the output)
  const uint32_t width = a.extent.width, height = a.extent.height;   // settings.screenshot_framebuffer_layout
  const ScreenRectVertex rect[6] = {{{40, 0}, {0, 0}}, {{600, 0}, {1, 0}}, {{40, 336}, {0, 1}},
                                    {{40, 336}, {0, 1}}, {{600, 0}, {1, 0}}, {{600, 336}, {1, 1}}};
  Buffer vb = make_buffer(a, sizeof rect, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "vertex_buffer");
  memcpy(vb.map, rect, sizeof rect);
  PresentUniformData pud{};
  float P[16] = {2.f / width, 0, 0, 0, 0, 2.f / height, 0, 0, 0, 0, 1, 0, -1, -1, 0, 1};   // ortho, y down
  memcpy(pud.modelview, P, sizeof P);
  pud.i_resolution[0] = SW; pud.i_resolution[1] = SH; pud.o_resolution[0] = (float)width; pud.o_resolution[1] = (float)height;

  auto DrawScreens = [&](VkCommandBuffer c, VkImageView target, uint32_t w, uint32_t h) {
    VkRenderingAttachmentInfo att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    att.imageView = target; att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {w, h}}; ri.layerCount = 1; ri.colorAttachmentCount = 1; ri.pColorAttachments = &att;
    vkCmdBeginRendering(c, &ri);
    VkViewport vp{0, 0, (float)w, (float)h, 0, 1}; VkRect2D sc{{0, 0}, {w, h}};
    vkCmdSetViewport(c, 0, 1, &vp); vkCmdSetScissor(c, 0, 1, &sc);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, present.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, present.layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(c, present.layout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, sizeof pud, &pud);
    VkDeviceSize off = 0; vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &off);
    vkCmdDraw(c, 6, 1, 0, 0);
    vkCmdEndRendering(c);
  };

  // normal presentation
  run_frames(a, frames_from_env(2), 1, [&](VkCommandBuffer c, uint32_t idx, int) {
    DrawScreens(c, a.swap_views[idx], width, height);
  });

  // RenderScreenshotWithStagingCopy()
  Buffer staging_buffer = make_buffer(a, (VkDeviceSize)width * height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, "screenshot_staging");
  Image frame = make_image(a, width, height, surface_format,                        // RecreateFrame(&frame, w, h)
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, "screenshot_frame");
  {
    VkCommandBuffer c = alloc_cmd(a, a.gfx_pool); begin(c);
    image_barrier(c, frame.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    DrawScreens(c, frame.view, width, height);
    image_barrier(c, frame.img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferImageCopy image_copy{}; image_copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    image_copy.imageExtent = {width, height, 1};
    vkCmdCopyImageToBuffer(c, frame.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging_buffer.buf, 1, &image_copy);
    buffer_barrier(c, staging_buffer.buf, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    VK(vkEndCommandBuffer(c));
    VkFence fence = make_fence(a, false, "scheduler_fence");
    submit(a.gfx, c, {}, {}, fence);
    VK(vkWaitForFences(a.dev, 1, &fence, VK_TRUE, UINT64_MAX));   // scheduler.Finish()
  }
  // Copy backing image data to the QImage screenshot buffer (QImage::Format_RGB32)
  std::vector<uint8_t> screenshot_bits((size_t)width * height * 4);
  std::memcpy(screenshot_bits.data(), staging_buffer.map, screenshot_bits.size());
#ifdef FIXED
  // QImage::Format_RGB32 expects BGRA byte order. If the swapchain format is RGBA,
  // swap R and B channels so the screenshot colors are correct.
  if (surface_format == VK_FORMAT_R8G8B8A8_UNORM) {
    uint8_t* pixels = screenshot_bits.data();
    for (uint32_t i = 0; i < width * height; i++) std::swap(pixels[i * 4 + 0], pixels[i * 4 + 2]);
  }
#endif
  // screenshot_complete_callback -> QImage::save()
  write_bmp(a.shader_dir + "/screenshot.bmp", screenshot_bits.data(), width, height);
  return 0;
}
