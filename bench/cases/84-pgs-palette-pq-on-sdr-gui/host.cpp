// Distilled from xbmc/xbmc PR #28996 ("[Subtitles][Overlay][GLES] Add HDR
// PGS subtitle palette colour handling", merged 2026-09-13). PGS (Blu-ray
// bitmap) subtitles carry a palette but no colour description of their own;
// the decoder converts the palette with the paired video's colour space, so
// with HDR10 video the palette entries are BT.2020 primaries with PQ
// (ST 2084) encoding. On platforms whose GUI layer stays SDR (BT.709 / sRGB:
// Android, webOS) the overlay uploaded those code values unchanged, and the
// subtitles came out dull and hue-shifted. The fix converts the 256-entry
// palette on the CPU before texture upload - PQ -> linear nits, scaled by a
// user white point (default 550 nits, setting range 50-1050), BT.2020 ->
// BT.709, clip, sRGB encode - only on platforms without HDR GUI composition
// (GBM already composites correctly). Here the subtitle is a small bitmap
// whose palette holds PQ-coded white and yellow text plus a transparent
// background; the decoder's YCbCr -> RGB step is folded into the palette.
#include "../../common/app.hpp"
#include <cmath>
using namespace bench;

struct PaletteEntry { float r, g, b, a; };      // [0, 1] code values as produced by the decoder

static float pq_to_nits(float e) {               // ST 2084 EOTF
  const float m1 = 0.1593017578125f, m2 = 78.84375f, c1 = 0.8359375f, c2 = 18.8515625f, c3 = 18.6875f;
  float p = std::pow(std::fmax(e, 0.0f), 1.0f / m2);
  return 10000.0f * std::pow(std::fmax(p - c1, 0.0f) / (c2 - c3 * p), 1.0f / m1);
}
static float srgb_oetf(float l) { return l <= 0.0031308f ? 12.92f * l : 1.055f * std::pow(l, 1.0f / 2.4f) - 0.055f; }

// CDVDOverlayCodecPGS palette -> overlay texture palette
static void prepare_palette(PaletteEntry* pal, int n, bool gui_is_hdr, float white_nits) {
#ifdef FIXED
  if (gui_is_hdr) return;                        // GBM: HDR GUI composition handles PQ / BT.2020
  for (int i = 0; i < n; i++) {
    float r = pq_to_nits(pal[i].r) / white_nits, g = pq_to_nits(pal[i].g) / white_nits, b = pq_to_nits(pal[i].b) / white_nits;
    float r7 = 1.6605f * r - 0.5876f * g - 0.0728f * b;          // BT.2020 -> BT.709
    float g7 = -0.1246f * r + 1.1329f * g - 0.0083f * b;
    float b7 = -0.0182f * r - 0.1006f * g + 1.1187f * b;
    auto clip = [](float v) { return std::fmin(std::fmax(v, 0.0f), 1.0f); };
    pal[i] = {srgb_oetf(clip(r7)), srgb_oetf(clip(g7)), srgb_oetf(clip(b7)), pal[i].a};
  }
#else
  (void)pal; (void)n; (void)gui_is_hdr; (void)white_nits;   // palette used as decoded
#endif
}

int main(int argc, char** argv) {
  AppDesc d; d.swap_format = VK_FORMAT_B8G8R8A8_UNORM;     // Android GUI surface: SDR, sRGB-encoded values
  App a = make_app(argc, argv, d);
  auto sh = load_shader(a, "shaders.spv");
  const bool gui_is_hdr = false;                           // Android / webOS
  const float white_nits = 550.0f;                         // subtitles.hdrwhitepoint default

  // HDR10 movie: the palette was converted with the video's colour space (BT.2020, PQ)
  PaletteEntry palette[3] = {{0, 0, 0, 0},                   // background (transparent)
                             {0.58f, 0.58f, 0.58f, 1},       // white text (~200 nits in PQ)
                             {0.57f, 0.56f, 0.35f, 1}};      // yellow text
  prepare_palette(palette, 3, gui_is_hdr, white_nits);

  // expand the 8-bit indexed bitmap to RGBA8 for upload (COverlayTextureGL)
  const uint32_t W = 32, H = 8;
  Buffer px = make_buffer(a, W * H * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "subtitle_rgba");
  for (uint32_t y = 0; y < H; y++)
    for (uint32_t x = 0; x < W; x++) {
      int idx = (y > 1 && y < 6) ? ((x / 4) % 2 ? 1 : 2) : 0;
      uint8_t* q = (uint8_t*)px.map + (y * W + x) * 4;
      q[0] = (uint8_t)std::lround(palette[idx].r * 255); q[1] = (uint8_t)std::lround(palette[idx].g * 255);
      q[2] = (uint8_t)std::lround(palette[idx].b * 255); q[3] = (uint8_t)std::lround(palette[idx].a * 255);
    }
  Image tex = make_image(a, W, H, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         "pgs_overlay_texture");
  VkCommandBuffer up = alloc_cmd(a, a.gfx_pool);
  begin(up);
  image_barrier(up, tex.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
  VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {W, H, 1};
  vkCmdCopyBufferToImage(up, px.buf, tex.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
  image_barrier(up, tex.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  VK(vkEndCommandBuffer(up));
  submit(a.gfx, up, {}, {}, VK_NULL_HANDLE);
  VK(vkQueueWaitIdle(a.gfx));

  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "overlay_texture");
  write_image(a, set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, tex.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline overlay = make_graphics(a, gd, "overlay_renderer");

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, overlay.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, overlay.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
  return 0;
}
