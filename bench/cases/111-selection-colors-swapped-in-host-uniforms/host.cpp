// Distilled from ghostty-org/ghostty PR #14460 ("renderer: fix custom shader
// selection color uniform layout", discussion #14459, merged 2026-09-29).
// Custom shaders get a std140 uniform block `Globals` whose last two members
// are
//   vec3 iSelectionForegroundColor;   // offset 4464
//   vec3 iSelectionBackgroundColor;   // offset 4480
// The CPU-side shadertoy.Uniforms extern struct that is copied into that
// buffer declared selection_background_color before
// selection_foreground_color, so every custom shader read the two selection
// colours swapped. The fix reorders the two CPU fields (and adds a test that
// compares SPIRV-Cross reflected offsets with @offsetOf).
// Port: Zig extern struct -> C++ struct with the same alignment; the
// Vulkan custom-shader pass draws over a generated terminal image with a
// selected region.
#include "../../common/app.hpp"
#include <cmath>
using namespace bench;

// shadertoy.Uniforms
struct alignas(16) Uniforms {
  alignas(16) float resolution[3];
  float time;
  float time_delta;
  float frame_rate;
  int32_t frame;
  alignas(16) float channel_time[4][4];
  alignas(16) float channel_resolution[4][4];
  alignas(16) float mouse[4];
  alignas(16) float date[4];
  float sample_rate;
  alignas(16) float current_cursor[4];
  alignas(16) float previous_cursor[4];
  alignas(16) float current_cursor_color[4];
  alignas(16) float previous_cursor_color[4];
  int32_t current_cursor_style;
  int32_t previous_cursor_style;
  int32_t cursor_visible;
  float cursor_change_time;
  float time_focus;
  int32_t focus;
  alignas(16) float palette[256][4];
  alignas(16) float background_color[4];
  alignas(16) float foreground_color[4];
  alignas(16) float cursor_color[4];
  alignas(16) float cursor_text[4];
#ifdef FIXED
  alignas(16) float selection_foreground_color[4];
  alignas(16) float selection_background_color[4];
#else
  alignas(16) float selection_background_color[4];
  alignas(16) float selection_foreground_color[4];
#endif
};

static void rgb(float* dst, uint32_t hex) {
  dst[0] = ((hex >> 16) & 0xff) / 255.f; dst[1] = ((hex >> 8) & 0xff) / 255.f; dst[2] = (hex & 0xff) / 255.f; dst[3] = 1.f;
}

int main(int argc, char** argv) {
  AppDesc d; d.swap_format = VK_FORMAT_B8G8R8A8_UNORM;
  App a = make_app(argc, argv, d);
  auto sh = load_shader(a, "shaders.spv");
  const uint32_t sw = a.extent.width, shh = a.extent.height;

  // config: background = #1e1e2e, foreground = #cdd6f4,
  //         selection-background = #585b70, selection-foreground = #f5e0dc
  const uint32_t bg = 0x1e1e2e, sel_bg = 0x585b70;
  Image term = make_image(a, sw, shh, VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                          "custom_shader_state.back_texture");
  Buffer up = make_buffer(a, (VkDeviceSize)sw * shh * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "back_texture upload");
  for (uint32_t y = 0; y < shh; y++)
    for (uint32_t x = 0; x < sw; x++) {
      uint8_t* p = (uint8_t*)up.map + (y * sw + x) * 4;
      uint32_t c = (y >= 100 && y < 140 && x >= 40 && x < 400) ? sel_bg : bg;   // a selected line
      p[0] = c & 0xff; p[1] = (c >> 8) & 0xff; p[2] = (c >> 16) & 0xff; p[3] = 255;   // BGRA
    }
  {
    VkCommandBuffer c = alloc_cmd(a, a.gfx_pool); begin(c);
    image_barrier(c, term.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {sw, shh, 1};
    vkCmdCopyBufferToImage(c, up.buf, term.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    image_barrier(c, term.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    VK(vkEndCommandBuffer(c)); submit(a.gfx, c, {}, {}, VK_NULL_HANDLE); vkQueueWaitIdle(a.gfx);
  }

  static Uniforms uniforms{};
  uniforms.resolution[0] = (float)sw; uniforms.resolution[1] = (float)shh; uniforms.resolution[2] = 1;
  for (int i = 0; i < 256; i++) rgb(uniforms.palette[i], (uint32_t)(i * 0x010101));
  rgb(uniforms.background_color, bg);
  rgb(uniforms.foreground_color, 0xcdd6f4);
  rgb(uniforms.cursor_color, 0xf5e0dc);
  rgb(uniforms.cursor_text, 0x1e1e2e);
  rgb(uniforms.selection_background_color, sel_bg);
  rgb(uniforms.selection_foreground_color, 0xf5e0dc);
  uniforms.cursor_visible = 1; uniforms.focus = 1;

  Buffer ubo = make_buffer(a, sizeof(Uniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "custom_shader_uniforms");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "custom_shader_set");
  write_image(a, set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, term.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  write_buffer(a, set, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.vs_entry = "full_screen_vertex"; gd.fs_entry = "main"; gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline custom = make_graphics(a, gd, "custom_shader_pipeline");

  run_frames(a, frames_from_env(2), 1, [&](VkCommandBuffer c, uint32_t idx, int) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, custom.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, custom.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  }, [&](int f) {
    uniforms.time = f / 60.0f; uniforms.frame = f;
    memcpy(ubo.map, &uniforms, sizeof uniforms);
  });
  return 0;
}
