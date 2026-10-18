// Distilled from BabylonJS/BabylonNative PR #1840 ("Correct gl_FragCoord Y
// orientation on D3D, Metal and Vulkan", merged 2026-09-04). Babylon's
// shaders are written for WebGL, where gl_FragCoord has a bottom-left
// origin. BabylonNative compiles them for bgfx backends that rasterize with
// a top-left origin (D3D, Metal, Vulkan) and never flipped gl_FragCoord.y,
// so any shader that addresses a screen-sized texture by fragment position
// read it vertically mirrored, unlike the same lookup done with the
// full-screen quad's UVs. The fix rewrites every gl_FragCoord read as
// vec4(x, targetHeight - y, z, w) (ShaderCompilerTraversers::FlipFragCoordY)
// with a new bnFragCoordTargetSize uniform that the engine fills per draw
// from the active framebuffer (bgfx's u_viewRect is the viewport, not the
// render target). Here the screen-sized texture is given in WebGL row order
// (row 0 = bottom of the screen, as readPixels / Babylon's texture
// convention produce it): a top half that is red and a bottom half that is
// blue.
#include "../../common/app.hpp"
using namespace bench;

struct Params { float screenSize[2]; float extra[2]; };

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  const uint32_t W = a.extent.width, H = a.extent.height;
  Image screen = make_image(a, W, H, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            "screen_texture");
  Buffer px = make_buffer(a, (VkDeviceSize)W * H * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "screen_pixels_gl_rows");
  for (uint32_t row = 0; row < H; row++) {                 // row 0 = bottom of the screen
    bool top_half = row >= H / 2;
    for (uint32_t x = 0; x < W; x++) {
      uint8_t* q = (uint8_t*)px.map + ((size_t)row * W + x) * 4;
      q[0] = top_half ? 255 : 0; q[1] = 0; q[2] = top_half ? 0 : 255; q[3] = 255;
    }
  }
  VkCommandBuffer up = alloc_cmd(a, a.gfx_pool);
  begin(up);
  image_barrier(up, screen.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
  VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {W, H, 1};
  vkCmdCopyBufferToImage(up, px.buf, screen.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
  image_barrier(up, screen.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  VK(vkEndCommandBuffer(up));
  submit(a.gfx, up, {}, {}, VK_NULL_HANDLE);
  VK(vkQueueWaitIdle(a.gfx));

  Buffer ubo = make_buffer(a, sizeof(Params), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "screen_params");
  Params p{{(float)W, (float)H}, {0, 0}};
#ifdef FIXED
  p.extra[0] = (float)W; p.extra[1] = (float)H;           // NativeEngine: bnFragCoordTargetSize = framebuffer size
#endif
  memcpy(ubo.map, &p, sizeof p);
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "screen_effect_bindings");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo);
  write_image(a, set, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, screen.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline effect = make_graphics(a, gd, "screen_space_effect");

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, effect.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, effect.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
  return 0;
}
