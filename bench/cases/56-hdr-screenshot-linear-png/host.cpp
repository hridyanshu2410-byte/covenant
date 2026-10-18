// Distilled from hi-godot/godot-ai PR #1070: the plugin captures game and
// editor 2D viewports as PNG screenshots. With HDR 2D enabled on the
// Forward+ and Mobile renderers, the 2D viewport is an RGBA16F texture
// holding linear colour (sRGB sprites are decoded when sampled; the window
// gets its sRGB encoding at blit time). The capture read that texture back
// and encoded the linear values straight to an 8-bit PNG, so midtones came
// out too dark. The fix applies a shared linear-to-sRGB conversion before
// PNG encoding (before downscaling); 8-bit quantization and clamping, not
// tone mapping (midgray 126 rather than 127). The Compatibility renderer
// reads back sRGB already and is unchanged. Godot's readback
// (get_texture().get_image()) is a copy to a host buffer here, and the PNG
// encoder is reduced to writing the 8-bit pixels.
#include "../../common/app.hpp"
#include <cmath>
using namespace bench;

static float half_to_float(uint16_t h) {
  uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  float v = e == 0 ? std::ldexp((float)m, -24) : e == 31 ? INFINITY : std::ldexp((float)(m | 0x400), (int)e - 25);
  return s ? -v : v;
}
static float linear_to_srgb(float c) {
  return c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

// Image (FORMAT_RGBAH, linear) -> 8-bit RGBA for save_png_to_buffer()
static std::vector<uint8_t> encode_capture(const uint16_t* rgbah, uint32_t w, uint32_t h) {
  std::vector<uint8_t> out(w * h * 4);
  for (uint32_t i = 0; i < w * h * 4; i++) {
    float v = half_to_float(rgbah[i]);
#ifdef FIXED
    if (i % 4 != 3) v = linear_to_srgb(std::fmin(std::fmax(v, 0.0f), 1.0f));
#endif
    out[i] = (uint8_t)std::lround(std::fmin(std::fmax(v, 0.0f), 1.0f) * 255.0f);
  }
  return out;
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  const uint32_t W = 640, H = 480;
  Image viewport = make_image(a, W, H, VK_FORMAT_R16G16B16A16_SFLOAT,
                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                              VK_IMAGE_USAGE_TRANSFER_SRC_BIT, "viewport_2d_hdr");
  Image sprite = make_image(a, 128, 128, VK_FORMAT_R8G8B8A8_SRGB,
                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, "sprite_texture");
  Buffer readback = make_buffer(a, W * H * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, "capture_readback");

  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto sprite_set = alloc_set(a, setl, "canvas_item_texture");
  write_image(a, sprite_set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sprite.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  auto blit_set = alloc_set(a, setl, "viewport_blit");
  write_image(a, blit_set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, viewport.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  GfxDesc cd; cd.vs = cd.fs = sh; cd.vs_entry = "vsCanvasItem"; cd.fs_entry = "fsCanvasItem"; cd.sets = {setl};
  cd.colors = {VK_FORMAT_R16G16B16A16_SFLOAT};
  Pipeline canvas = make_graphics(a, cd, "canvas_item");
  GfxDesc bd; bd.vs = bd.fs = sh; bd.vs_entry = "vsFullscreen"; bd.fs_entry = "fsBlit"; bd.sets = {setl};
  bd.colors = {a.swap_format};                         // B8G8R8A8_SRGB window: encodes on store
  Pipeline blit = make_graphics(a, bd, "blit_to_window");

  VkCommandBuffer up = alloc_cmd(a, a.gfx_pool); begin(up);
  image_barrier(up, sprite.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
  VK(vkEndCommandBuffer(up)); submit(a.gfx, up, {}, {}, VK_NULL_HANDLE); VK(vkQueueWaitIdle(a.gfx));

  const int frames = frames_from_env(3);
  run_frames(a, frames, 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    image_barrier(c, viewport.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    begin_rendering(a, c, {viewport.view});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, canvas.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, canvas.layout, 0, 1, &sprite_set, 0, nullptr);
    vkCmdDraw(c, 6, 1, 0, 0);
    vkCmdEndRendering(c);
    image_barrier(c, viewport.img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT);
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, blit.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, blit.layout, 0, 1, &blit_set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
    if (f == frames - 1) {    // screenshot request: viewport.get_texture().get_image()
      image_barrier(c, viewport.img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
      VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {W, H, 1};
      vkCmdCopyImageToBuffer(c, viewport.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buf, 1, &r);
      buffer_barrier(c, readback.buf, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
      image_barrier(c, viewport.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0);
    }
  });
  // run_frames ends with vkDeviceWaitIdle: the readback is complete
  std::vector<uint8_t> png_pixels = encode_capture((const uint16_t*)readback.map, W, H);
  std::string path = a.shader_dir + "/screenshot.rgba";
  if (FILE* fp = fopen(path.c_str(), "wb")) { fwrite(png_pixels.data(), 1, png_pixels.size(), fp); fclose(fp); }
  return 0;
}
