// Distilled from Dawn CL 307215 (https://dawn-review.googlesource.com/c/dawn/+/307215,
// commit c78c5f1 "[YUV AHB] Fix PQ and HLG color space luminance", landed
// 2026-05-07). ComputeExternalTextureParams() builds the uniforms Tint's
// external-texture lowering uses: source EOTF, a gamut matrix from source
// linear RGB to destination linear RGB, and the destination OETF. The PQ EOTF
// returns linear light with 1.0 = 10000 nits and the HLG one with 1.0 = 3.7743
// x 203 nits, while the destination (sRGB) treats 1.0 as SDR reference white
// (203 nits). The gamut matrix was just
//   math::Mul(dstXYZToRGB, srcRGBToXYZ)
// so PQ and HLG content was mapped as if both "1.0"s were the same luminance:
// PQ video came out roughly 50x too dark. The fix multiplies the matrix by
// srcLuminanceOf1 / dstLuminanceOf1 (kPQLuminanceOf1 = 10000,
// kHLGLuminanceOf1 = 3.7743 * 203, kSDRLuminanceOf1 = 203).
// Port: Dawn/Tint -> C++/Slang/Vulkan; the parameter computation is kept
// (Mat3x3 maths inlined), the source is a single-plane RGB HDR10 frame
// (identity YCbCr matrix and range) instead of a YUV AHB, the destination is
// an sRGB canvas on a UNORM swapchain.
#include "../../common/app.hpp"
#include <array>
#include <cmath>
using namespace bench;

struct TransferFunction { float g = 1, a = 1, b = 0, c = 0, d = 0, e = 0, f = 0; };
constexpr TransferFunction kEOTF_PQ = {-2, 2610.0f / 16384.0f, (2523.0f / 4096.0f) * 128.0f, 3424.0f / 4096.0f,
                                       (2413.0f / 4096.0f) * 32.0f, (2392.0f / 4096.0f) * 32.0f, 0.0f};
constexpr TransferFunction kEOTFInverse_sRGB = {1.0f / 2.4f, 1.13711f, 0, 12.92f, 0.0031308f, -0.055f, 0};
#ifdef FIXED
constexpr float kSDRLuminanceOf1 = 203;
constexpr float kPQLuminanceOf1 = 10000;
#endif

using Mat3 = std::array<std::array<float, 3>, 3>;   // [row][col]
constexpr Mat3 kRGBToXYZ_Rec2020 = {{{0.636958f, 0.144617f, 0.168881f}, {0.262700f, 0.677998f, 0.059302f}, {0.0f, 0.028073f, 1.060985f}}};
constexpr Mat3 kXYZToRGB_sRGB = {{{3.240970f, -1.537383f, -0.498611f}, {-0.969244f, 1.875968f, 0.041555f}, {0.055630f, -0.203977f, 1.056972f}}};
static Mat3 Mul(const Mat3& a, const Mat3& b) {
  Mat3 r{};
  for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) for (int k = 0; k < 3; k++) r[i][j] += a[i][k] * b[k][j];
  return r;
}

// std140 mirror of ExternalTextureParams
struct GammaTransferParams { float G, A, B, C, D, E, F, padding; };
struct ExternalTextureParams {
  uint32_t numPlanes; uint32_t doYuvToRgbConversionOnly; uint32_t _pad0[2];
  float yuvToRgbConversionMatrix[3][4];   // 3 columns of float4
  GammaTransferParams gammaDecodeParams;
  GammaTransferParams gammaEncodeParams;
  float gamutConversionMatrix[3][4];      // 3 columns of float3, padded
};

static GammaTransferParams toParams(const TransferFunction& tf) { return {tf.g, tf.a, tf.b, tf.c, tf.d, tf.e, tf.f, 0}; }

// ComputeExternalTextureParams(srcColorSpace = {Rec2020, PQ, Identity, Identity}, dst = SRGB)
static ExternalTextureParams ComputeExternalTextureParams() {
  ExternalTextureParams out{};
  out.numPlanes = 1; out.doYuvToRgbConversionOnly = 0;
  // YCbCr matrix and range Identity: rows of a 3x4 identity
  for (int c = 0; c < 3; c++) for (int r = 0; r < 4; r++) out.yuvToRgbConversionMatrix[c][r] = (r == c) ? 1.f : 0.f;
  Mat3 srcRGBToXYZ = kRGBToXYZ_Rec2020;
  TransferFunction srcEOTF = kEOTF_PQ;
  Mat3 dstXYZToRGB = kXYZToRGB_sRGB;
  TransferFunction dstOETF = kEOTFInverse_sRGB;
  // The gamut matrix is passed column-major.
  Mat3 gamutConversionMatrix = Mul(dstXYZToRGB, srcRGBToXYZ);
#ifdef FIXED
  float luminanceAdjustment = kPQLuminanceOf1 / kSDRLuminanceOf1;
  for (auto& row : gamutConversionMatrix) for (float& v : row) v *= luminanceAdjustment;
#endif
  for (int col = 0; col < 3; col++) for (int row = 0; row < 3; row++) out.gamutConversionMatrix[col][row] = gamutConversionMatrix[row][col];
  out.gammaDecodeParams = toParams(srcEOTF);
  out.gammaEncodeParams = toParams(dstOETF);
  return out;
}

// SMPTE ST 2084 inverse EOTF (to build the test frame): nits -> PQ code value
static float pq_encode(float nits) {
  const float m1 = 2610.f / 16384.f, m2 = 2523.f / 4096.f * 128.f, c1 = 3424.f / 4096.f, c2 = 2413.f / 4096.f * 32.f, c3 = 2392.f / 4096.f * 32.f;
  float y = powf(nits / 10000.f, m1);
  return powf((c1 + c2 * y) / (1 + c3 * y), m2);
}

int main(int argc, char** argv) {
  AppDesc d; d.swap_format = VK_FORMAT_B8G8R8A8_UNORM;   // canvas: 'srgb' predefined colour space, rgba8unorm
  App a = make_app(argc, argv, d);
  auto sh = load_shader(a, "shaders.spv");

  // HDR10 frame (RGBA16F holding PQ code values): grey ramp at 5, 50, 100, 203, 400 nits
  const uint32_t W = 160, H = 32;
  Image frame = make_image(a, W, H, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                           "videoFrame (PQ, BT.2020)");
  Buffer staging = make_buffer(a, W * H * 8, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "videoFrame upload");
  auto half = [](float f) {   // float -> IEEE half (positive, normal range)
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t e = ((x >> 23) & 0xff) - 127 + 15, m = (x >> 13) & 0x3ff;
    return (uint16_t)(f <= 0 ? 0 : ((e << 10) | m));
  };
  const float nits[5] = {5, 50, 100, 203, 400};
  for (uint32_t y = 0; y < H; y++)
    for (uint32_t x = 0; x < W; x++) {
      uint16_t* p = (uint16_t*)staging.map + (y * W + x) * 4;
      uint16_t v = half(pq_encode(nits[x * 5 / W]));
      p[0] = p[1] = p[2] = v; p[3] = half(1.0f);
    }
  {
    VkCommandBuffer c = alloc_cmd(a, a.gfx_pool); begin(c);
    image_barrier(c, frame.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {W, H, 1};
    vkCmdCopyBufferToImage(c, staging.buf, frame.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    image_barrier(c, frame.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    VK(vkEndCommandBuffer(c)); submit(a.gfx, c, {}, {}, VK_NULL_HANDLE); vkQueueWaitIdle(a.gfx);
  }

  ExternalTextureParams params = ComputeExternalTextureParams();
  Buffer paramsBuf = make_buffer(a, sizeof params, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "ExternalTexture params");
  memcpy(paramsBuf.map, &params, sizeof params);

  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "external texture bind group");
  write_image(a, set, 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, frame.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  write_image(a, set, 1, VK_DESCRIPTOR_TYPE_SAMPLER, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED);
  write_buffer(a, set, 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, paramsBuf);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.vs_entry = "vertMain"; gd.fs_entry = "fragMain"; gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline video = make_graphics(a, gd, "video quad");

  run_frames(a, frames_from_env(2), 1, [&](VkCommandBuffer c, uint32_t idx, int) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, video.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, video.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
  return 0;
}
