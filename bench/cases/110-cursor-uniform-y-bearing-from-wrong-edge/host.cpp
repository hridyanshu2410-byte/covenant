// Distilled from ghostty-org/ghostty PR #8122 ("fix: correct the cursor Y
// position value exposed to shader uniforms", discussion #8113, merged
// 2025-08-31). For custom shaders Ghostty exposes iCurrentCursor /
// iPreviousCursor: x is the cursor's left edge and y its +Y edge in the
// shader's fragCoord space, z/w its size. The renderer computed y as
//   pixel_y = grid_pos.y * cell.height + padding.top;   // top edge of the cell
//   pixel_y += bearings[1];
// but a glyph's Y bearing is measured upward from the bottom of the cell to
// the top of the glyph, so adding it to a y-down top-of-cell coordinate does
// not give the glyph's top (or bottom) edge. Bar and underline cursors were
// reported at the wrong height by cursor-trail shaders. The fix (y-down APIs):
//   pixel_y += cell.height - bearings[1] + glyph_size[1];   // +Y (bottom) edge
// and for y-up APIs flips first, then pixel_y -= cell.height - bearings[1].
// Port: Zig -> C++; Vulkan fragCoord is y-down like Metal's
// (custom_shader_y_is_down = true); the terminal image is a generated grid
// and the cursor is an underline cursor (2 px tall at the cell bottom).
#include "../../common/app.hpp"
using namespace bench;

// std140 mirror of the Globals block (shadertoy_prefix.glsl)
struct Uniforms {
  float resolution[3]; float time;
  float time_delta; float frame_rate; int32_t frame; float _pad0;
  float channel_time[4][4];            // float[4], 16-byte stride
  float channel_resolution[4][4];      // vec3[4]
  float mouse[4]; float date[4];
  float sample_rate; float _pad1[3];
  float current_cursor[4]; float previous_cursor[4];
  float current_cursor_color[4]; float previous_cursor_color[4];
  float time_cursor_change; float _pad2[3];
};
static_assert(offsetof(Uniforms, channel_time) == 32 && offsetof(Uniforms, current_cursor) == 208 &&
              offsetof(Uniforms, time_cursor_change) == 272, "Globals std140");

struct CellSize { uint32_t width, height; };
struct Padding { uint32_t top, left; };
struct CursorGlyph { uint16_t grid_pos[2]; uint32_t glyph_size[2]; int32_t bearings[2]; uint8_t color[4]; };

int main(int argc, char** argv) {
  AppDesc d; d.swap_format = VK_FORMAT_B8G8R8A8_UNORM;
  App a = make_app(argc, argv, d);
  auto sh = load_shader(a, "shaders.spv");
  const CellSize cell{10, 20};
  const Padding padding{4, 4};
  const uint32_t sw = a.extent.width, shh = a.extent.height;

  // the terminal image the custom shader post-processes (iChannel0)
  Image term = make_image(a, sw, shh, VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                          "custom_shader_state.back_texture");
  Buffer up = make_buffer(a, (VkDeviceSize)sw * shh * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "back_texture upload");
  for (uint32_t y = 0; y < shh; y++)
    for (uint32_t x = 0; x < sw; x++) {
      uint8_t* p = (uint8_t*)up.map + (y * sw + x) * 4;
      bool line = ((y - padding.top) % cell.height) == 0 || ((x - padding.left) % cell.width) == 0;
      p[0] = p[1] = p[2] = line ? 0x50 : 0x28; p[3] = 255;
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

  Buffer ubo = make_buffer(a, sizeof(Uniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "custom_shader_uniforms");
  Uniforms uniforms{};
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "custom_shader_set");
  write_image(a, set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, term.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  write_buffer(a, set, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.vs_entry = "full_screen_vertex"; gd.fs_entry = "main"; gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline custom = make_graphics(a, gd, "custom_shader_pipeline");

  // the cursor moves one column per frame; underline cursor: 2 px tall, top 2 px above the cell bottom
  const bool custom_shader_y_is_down = true;
  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, custom.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, custom.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  }, [&](int f) {
    // updateCustomShaderUniforms()
    uniforms.time = f / 60.0f; uniforms.time_delta = 1 / 60.0f; uniforms.frame = f;
    uniforms.resolution[0] = (float)sw; uniforms.resolution[1] = (float)shh; uniforms.resolution[2] = 1;
    CursorGlyph cursor{{(uint16_t)(3 + f), 5}, {cell.width, 2}, {0, 2}, {0xff, 0xcc, 0x00, 0xff}};
    const float cursor_width = (float)cursor.glyph_size[0];
    const float cursor_height = (float)cursor.glyph_size[1];
    // Left edge of the cell the cursor is in.
    float pixel_x = (float)(cursor.grid_pos[0] * cell.width + padding.left);
    // Top edge, relative to the top of the screen, of the cell the cursor is in.
    float pixel_y = (float)(cursor.grid_pos[1] * cell.height + padding.top);
#ifdef FIXED
    if (!custom_shader_y_is_down) pixel_y = (float)shh - pixel_y;
    pixel_x += (float)cursor.bearings[0];
    if (custom_shader_y_is_down) {
      pixel_y += (float)cell.height;
      pixel_y -= (float)cursor.bearings[1];
      pixel_y += (float)cursor.glyph_size[1];
    } else {
      pixel_y -= (float)cell.height;
      pixel_y += (float)cursor.bearings[1];
    }
#else
    pixel_x += (float)cursor.bearings[0];
    pixel_y += (float)cursor.bearings[1];
    // If +Y is up in our shaders, we need to flip the coordinate.
    if (!custom_shader_y_is_down) {
      pixel_y = (float)shh - pixel_y;
      // We need to add the cursor height because we need the +Y
      // edge for the Y coordinate, and flipping means that it's
      // the -Y edge now.
      pixel_y += cursor_height;
    }
#endif
    const float new_cursor[4] = {pixel_x, pixel_y, cursor_width, cursor_height};
    const float cursor_color[4] = {cursor.color[0] / 255.f, cursor.color[1] / 255.f, cursor.color[2] / 255.f, cursor.color[3] / 255.f};
    if (memcmp(new_cursor, uniforms.current_cursor, sizeof new_cursor) != 0) {
      memcpy(uniforms.previous_cursor, uniforms.current_cursor, 16);
      memcpy(uniforms.previous_cursor_color, uniforms.current_cursor_color, 16);
      memcpy(uniforms.current_cursor, new_cursor, 16);
      memcpy(uniforms.current_cursor_color, cursor_color, 16);
      uniforms.time_cursor_change = uniforms.time;
    }
    memcpy(ubo.map, &uniforms, sizeof uniforms);
  });
  return 0;
}
