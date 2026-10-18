// Distilled from zed-industries/zed PR #52641 ("gpui: Fix BGRA conversion for
// SVG rendering", merged 2026-03-29). gpui's sprite atlas stores polychrome
// images as BGRA8 with straight alpha, so every decoder path converts to that
// layout before upload (frames_for_image swaps PNG/JPEG pixels to BGRA, GIF
// frames get pixel.swap(0, 2)). SVGs are rasterised by resvg/tiny-skia into
// premultiplied RGBA; SvgRenderer::render_single_frame(bytes, scale, to_brga)
// only ran swap_rgba_pa_to_bgra (swap R/B and un-premultiply) when to_brga was
// true, and Image::to_image_data() -- used for SVG images created with
// Image::from_bytes(ImageFormat::Svg, ..), e.g. SVGs pasted from the
// clipboard -- passed false. The RGBA bytes were uploaded into the BGRA atlas:
// #38BDF8 (light blue) rendered as #F8BD38 (yellow). The fix removes the flag
// and always converts.
// Port: Rust -> C++; the SVG rasteriser is a stand-in that produces the same
// premultiplied RGBA pixmap for a filled rect; the atlas is a
// B8G8R8A8_UNORM image and the window surface is B8G8R8A8_UNORM.
#include "../../common/app.hpp"
using namespace bench;

struct GlobalParams { float viewport_size[2]; float pad[2]; };
struct PolychromeSprite { float bounds[4]; float tile_uv[4]; float opacity; uint32_t grayscale; float pad[2]; };

struct RenderImage { uint32_t width, height; std::vector<uint8_t> bytes; };   // BGRA, straight alpha

// gpui::color::swap_rgba_pa_to_bgra
static void swap_rgba_pa_to_bgra(uint8_t* color) {
  std::swap(color[0], color[2]);
  if (color[3] > 0) {
    float a = color[3] / 255.f;
    color[0] = (uint8_t)(color[0] / a); color[1] = (uint8_t)(color[1] / a); color[2] = (uint8_t)(color[2] / a);
  }
}

// usvg/resvg stand-in: rasterise <rect fill="#38BDF8" fill-opacity=".9"/> with a
// transparent 2 px margin into a tiny_skia::Pixmap (premultiplied RGBA)
static std::vector<uint8_t> render_pixmap(uint32_t w, uint32_t h) {
  std::vector<uint8_t> px(w * h * 4, 0);
  const uint8_t r = 0x38, g = 0xBD, b = 0xF8, a = 230;
  for (uint32_t y = 2; y < h - 2; y++)
    for (uint32_t x = 2; x < w - 2; x++) {
      uint8_t* p = &px[(y * w + x) * 4];
      p[0] = (uint8_t)(r * a / 255); p[1] = (uint8_t)(g * a / 255); p[2] = (uint8_t)(b * a / 255); p[3] = a;
    }
  return px;
}

// SvgRenderer::render_single_frame
static RenderImage render_single_frame(uint32_t w, uint32_t h, bool to_brga) {
  RenderImage img{w, h, render_pixmap(w, h)};
#ifdef FIXED
  (void)to_brga;
  for (size_t i = 0; i < img.bytes.size(); i += 4) swap_rgba_pa_to_bgra(&img.bytes[i]);
#else
  if (to_brga)
    for (size_t i = 0; i < img.bytes.size(); i += 4) swap_rgba_pa_to_bgra(&img.bytes[i]);
#endif
  return img;
}

// frames_for_image(bytes, ImageFormat::Png): decoded RGBA8 -> BGRA
static RenderImage frames_for_png(uint32_t w, uint32_t h) {
  RenderImage img{w, h, std::vector<uint8_t>(w * h * 4)};
  for (uint32_t i = 0; i < w * h; i++) {
    uint8_t* p = &img.bytes[i * 4];
    p[0] = 0x22; p[1] = 0xC5; p[2] = 0x5E; p[3] = 0xFF;   // decoded #22C55E (RGBA order)
    std::swap(p[0], p[2]);
  }
  return img;
}

int main(int argc, char** argv) {
  AppDesc d; d.swap_format = VK_FORMAT_B8G8R8A8_UNORM;
  App a = make_app(argc, argv, d);
  auto sh = load_shader(a, "shaders.spv");

  // ImageAssetLoader path for an <img> of a PNG asset, and Image::to_image_data
  // for an SVG pasted from the clipboard (Image::from_bytes(ImageFormat::Svg, ..))
  RenderImage png = frames_for_png(32, 32);
  RenderImage svg = render_single_frame(32, 32, /*to_brga=*/false);   // Image::to_image_data

  // polychrome atlas texture (BGRA8), tiles side by side
  const uint32_t AW = 64, AH = 32;
  Image atlas = make_image(a, AW, AH, VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                           "polychrome_atlas_0");
  Buffer staging = make_buffer(a, AW * AH * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "atlas_upload");
  for (uint32_t y = 0; y < 32; y++) {
    memcpy((uint8_t*)staging.map + (y * AW) * 4, &png.bytes[y * 32 * 4], 32 * 4);
    memcpy((uint8_t*)staging.map + (y * AW + 32) * 4, &svg.bytes[y * 32 * 4], 32 * 4);
  }
  {
    VkCommandBuffer c = alloc_cmd(a, a.gfx_pool); begin(c);
    image_barrier(c, atlas.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {AW, AH, 1};
    vkCmdCopyBufferToImage(c, staging.buf, atlas.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    image_barrier(c, atlas.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    VK(vkEndCommandBuffer(c)); submit(a.gfx, c, {}, {}, VK_NULL_HANDLE); vkQueueWaitIdle(a.gfx);
  }

  PolychromeSprite sprites[2] = {
      {{40, 40, 128, 128}, {0.0f, 0, 0.5f, 1}, 1.0f, 0, {0, 0}},
      {{200, 40, 128, 128}, {0.5f, 0, 0.5f, 1}, 1.0f, 0, {0, 0}}};
  Buffer spriteBuf = make_buffer(a, sizeof sprites, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, "poly_sprites");
  memcpy(spriteBuf.map, sprites, sizeof sprites);
  GlobalParams gp{{(float)a.extent.width, (float)a.extent.height}, {0, 0}};
  Buffer globals = make_buffer(a, sizeof gp, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "globals");
  memcpy(globals.map, &gp, sizeof gp);

  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "poly_sprites_set");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, globals);
  write_buffer(a, set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, spriteBuf);
  write_image(a, set, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, atlas.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.vs_entry = "vs_poly_sprite"; gd.fs_entry = "fs_poly_sprite";
  gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline polySprites = make_graphics(a, gd, "poly_sprites");

  run_frames(a, frames_from_env(2), 1, [&](VkCommandBuffer c, uint32_t idx, int) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, polySprites.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, polySprites.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 6, 2, 0, 0);
    vkCmdEndRendering(c);
  });
  return 0;
}
