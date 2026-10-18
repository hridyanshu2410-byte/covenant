// Distilled from labelle-toolkit/labelle-bgfx issue #155 / PR #156
// ("fix(video): pick the GPU-YUV matrix from the stream's colour metadata",
// merged 2026-09-26). The GPU video path's fs_yuv had BT.601 limited-range
// coefficients hard-coded, while the decoders report the stream's colour
// space (the CPU path already used it). The FP intro video, 1920x1080
// BT.709 limited range, played on Android with shifted chroma on the GPU
// path. The fix adds yuv_uniform.zig, which turns the decoder's
// ColorSpace / yuv.Matrix into two uniforms (u_yuvOffsetGain, u_yuvCoeffs);
// the player reads decoder.colorSpace() after each frame and rebuilds them
// when the matrix changes. Decoders without colour metadata keep BT.601
// limited (the old behaviour). Planes are 4x4 here.
#include "../../common/app.hpp"
using namespace bench;

enum class Matrix { bt601, bt709 };
struct ColorSpace { Matrix matrix; bool full_range; };

// the hardware decoder: reports the stream's colour metadata
struct Decoder {
  ColorSpace colorSpace() const { return {Matrix::bt709, false}; }     // FP intro: BT.709, limited range
};

struct YuvUniforms { float offsetGain[4]; float coeffs[4]; };
// yuv_uniform.zig: fromColorSpace
static YuvUniforms yuv_uniforms(ColorSpace cs) {
  YuvUniforms u{};
  u.offsetGain[0] = cs.full_range ? 0.0f : 16.0f / 255.0f;
  u.offsetGain[1] = cs.full_range ? 1.0f : 255.0f / 219.0f;
  u.offsetGain[2] = 0.5f;
  const float cscale = cs.full_range ? 1.0f : 255.0f / 224.0f;
  if (cs.matrix == Matrix::bt709) {
    const float Kr = 0.2126f, Kb = 0.0722f, Kg = 1 - Kr - Kb;
    u.coeffs[0] = 2 * (1 - Kr) * cscale; u.coeffs[1] = -2 * (1 - Kb) * Kb / Kg * cscale;
    u.coeffs[2] = -2 * (1 - Kr) * Kr / Kg * cscale; u.coeffs[3] = 2 * (1 - Kb) * cscale;
  } else {
    const float Kr = 0.299f, Kb = 0.114f, Kg = 1 - Kr - Kb;
    u.coeffs[0] = 2 * (1 - Kr) * cscale; u.coeffs[1] = -2 * (1 - Kb) * Kb / Kg * cscale;
    u.coeffs[2] = -2 * (1 - Kr) * Kr / Kg * cscale; u.coeffs[3] = 2 * (1 - Kb) * cscale;
  }
  return u;
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  Decoder decoder;
  const VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  Image planes[3] = {make_image(a, 4, 4, VK_FORMAT_R8_UNORM, usage, "video_plane_y"),
                     make_image(a, 4, 4, VK_FORMAT_R8_UNORM, usage, "video_plane_u"),
                     make_image(a, 4, 4, VK_FORMAT_R8_UNORM, usage, "video_plane_v")};
  Buffer px = make_buffer(a, 48, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "decoded_frame");
  uint8_t* p = (uint8_t*)px.map;
  memset(p, 104, 16); memset(p + 16, 168, 16); memset(p + 32, 82, 16);   // a saturated blue-ish sky (BT.709 coded)
  VkCommandBuffer up = alloc_cmd(a, a.gfx_pool);
  begin(up);
  for (int i = 0; i < 3; i++) {
    image_barrier(up, planes[i].img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkBufferImageCopy r{}; r.bufferOffset = 16 * i; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {4, 4, 1};
    vkCmdCopyBufferToImage(up, px.buf, planes[i].img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    image_barrier(up, planes[i].img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  }
  VK(vkEndCommandBuffer(up));
  submit(a.gfx, up, {}, {}, VK_NULL_HANDLE);
  VK(vkQueueWaitIdle(a.gfx));

  std::vector<Binding> bl = {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
                             {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
                             {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}};
#ifdef FIXED
  bl.push_back({3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT});
  Buffer ubo = make_buffer(a, sizeof(YuvUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "yuv_uniforms");
  YuvUniforms yu = yuv_uniforms(decoder.colorSpace());     // rebuilt when the matrix changes
  memcpy(ubo.map, &yu, sizeof yu);
#else
  (void)decoder; (void)yuv_uniforms;                        // colour metadata unused by the GPU path
#endif
  auto setl = make_set_layout(a, bl);
  auto set = alloc_set(a, setl, "yuv_textures");
  for (int i = 0; i < 3; i++)
    write_image(a, set, i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, planes[i].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
#ifdef FIXED
  write_buffer(a, set, 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo);
#endif
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline yuvp = make_graphics(a, gd, "video_yuv_to_rgb");

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, yuvp.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, yuvp.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
  return 0;
}
