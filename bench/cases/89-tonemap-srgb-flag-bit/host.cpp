// Distilled from godotengine/godot PR #112272 ("Fix tonemap sRGB conversion
// flag", merged 2025-11-01; issue #112263). The tonemap shader's push
// constant flags were renumbered when FLAG_USE_8_BIT_DEBANDING was split
// out, so tonemap.glsl tests FLAG_CONVERT_TO_SRGB as (1 << 6), but
// TonemapperRD (tone_mapper.h) still defined TONEMAP_FLAG_CONVERT_TO_SRGB as
// (1 << 7). The renderer asked for the conversion when the viewport target
// is not an HDR 2D buffer, the shader never saw the bit, and linear
// tonemapped values were written to the UNORM viewport texture: the 3D view
// came out too dark with crushed shadows. With "HDR 2D" enabled the
// conversion is not requested at all, which hid the bug. The fix sets the
// host constant to (1 << 6).
// Port: the HDR scene colour is an RGBA16F image filled with a linear
// gradient; the viewport target is the B8G8R8A8_UNORM swapchain.
#include "../../common/app.hpp"
using namespace bench;

// servers/rendering/renderer_rd/effects/tone_mapper.h
enum {
  TONEMAP_FLAG_USE_BCS = (1 << 0),
  TONEMAP_FLAG_USE_GLOW = (1 << 1),
  TONEMAP_FLAG_USE_AUTO_EXPOSURE = (1 << 2),
  TONEMAP_FLAG_USE_COLOR_CORRECTION = (1 << 3),
  TONEMAP_FLAG_USE_FXAA = (1 << 4),
  TONEMAP_FLAG_USE_8_BIT_DEBANDING = (1 << 5),
#ifdef FIXED
  TONEMAP_FLAG_CONVERT_TO_SRGB = (1 << 6),
#else
  TONEMAP_FLAG_CONVERT_TO_SRGB = (1 << 7),
#endif
};

struct TonemapPushConstant { float exposure; float white; uint32_t flags; float pad; };

struct TonemapSettings {
  float exposure = 1.0f, white = 4.0f;
  bool convert_to_srgb = false;
  bool use_debanding = false;
};

static TonemapPushConstant tonemapper(const TonemapSettings& s) {
  TonemapPushConstant pc{};
  pc.exposure = s.exposure; pc.white = s.white;
  pc.flags |= s.use_debanding ? TONEMAP_FLAG_USE_8_BIT_DEBANDING : 0;
  pc.flags |= s.convert_to_srgb ? TONEMAP_FLAG_CONVERT_TO_SRGB : 0;
  return pc;
}

int main(int argc, char** argv) {
  AppDesc ad; ad.swap_format = VK_FORMAT_B8G8R8A8_UNORM;     // viewport texture: UNORM, not an sRGB format
  App a = make_app(argc, argv, ad);
  auto sh = load_shader(a, "shaders.spv");

  // the 3D scene's HDR colour buffer (linear)
  Image hdr = make_image(a, 16, 1, VK_FORMAT_R16G16B16A16_SFLOAT,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, "scene_color_hdr");
  Buffer half_px = make_buffer(a, 16 * 4 * 2, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "scene_color_f16");
  auto to_half = [](float f) -> uint16_t {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000; int e = ((x >> 23) & 0xff) - 127 + 15; uint32_t m = (x >> 13) & 0x3ff;
    if (e <= 0) return (uint16_t)s; if (e >= 31) return (uint16_t)(s | 0x7c00);
    return (uint16_t)(s | (e << 10) | m);
  };
  uint16_t* h = (uint16_t*)half_px.map;
  for (int i = 0; i < 16; i++) {
    float v = 0.02f * (float)(i * i);                               // 0 .. 4.5 linear
    h[i * 4 + 0] = to_half(v); h[i * 4 + 1] = to_half(v * 0.8f); h[i * 4 + 2] = to_half(v * 0.6f); h[i * 4 + 3] = to_half(1.0f);
  }
  VkCommandBuffer up = alloc_cmd(a, a.gfx_pool);
  begin(up);
  image_barrier(up, hdr.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
  VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {16, 1, 1};
  vkCmdCopyBufferToImage(up, half_px.buf, hdr.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
  image_barrier(up, hdr.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  VK(vkEndCommandBuffer(up));
  submit(a.gfx, up, {}, {}, VK_NULL_HANDLE);
  VK(vkQueueWaitIdle(a.gfx));

  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "tonemap_source");
  write_image(a, set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, hdr.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format}; gd.push_size = sizeof(TonemapPushConstant);
  Pipeline tp = make_graphics(a, gd, "tonemap");

  // RendererSceneRenderRD::_render_buffers_post_process: the viewport is not
  // an HDR 2D target, so the tonemapper must write sRGB-encoded values
  TonemapSettings ts; ts.convert_to_srgb = true;
  TonemapPushConstant pc = tonemapper(ts);
  printf("tonemap flags = 0x%x\n", pc.flags);

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, tp.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, tp.layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(c, tp.layout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, sizeof pc, &pc);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
  return 0;
}
