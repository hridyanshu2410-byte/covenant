// Distilled from moonlight-stream/moonlight-qt issue #1667, fixed by commit
// 50223db ("Fix accuracy issues in YUV to RGB conversion of 10-bit content in
// D3D11VA", 2025-11-01). D3D11VARenderer::bindColorConversion() filled the
// colour-conversion constant buffer from fixed tables written for 8-bit
// video: k_CscMatrix_Bt2020Lim (luma scale 255/219 = 1.1644, chroma scales
// for 255/224) and k_Offsets_Lim = {16/255, 128/255, 128/255}. HDR streams
// are BT.2020 with 10-bit samples, which the shader reads normalized by 1023;
// in those units limited-range black is 64/1023 and the chroma midpoint is
// 512/1023, and the range scales are 1023/876 and 1023/896. Using the 8-bit
// constants shifts black and neutral chroma and slightly mis-scales the
// matrix (raised blacks, tinted greys). The fix computes the offsets and the
// matrix scaling from the sample bit depth:
//   yMin = 16 << (bits - 8), yScale = (range - 1) / (yMax - yMin), ...
//   offsets = {yMin / (range - 1), (range / 2) / (range - 1), ...}
// (and updates the hard-coded bt2020lim shader the same way).
// Port: D3D11 -> Vulkan; the 4:4:4 Y410 path that uses the host-computed
// constant buffer is kept, the Y410 frame is an A2B10G10R10 image filled with
// 10-bit limited-range test bars.
#include "../../common/app.hpp"
#include <cmath>
using namespace bench;

#define CSC_MATRIX_RAW_ELEMENT_COUNT 9
#define CSC_MATRIX_PACKED_ELEMENT_COUNT 12
#define OFFSETS_ELEMENT_COUNT 3
enum { COLORSPACE_REC_601, COLORSPACE_REC_709, COLORSPACE_REC_2020 };

#ifdef FIXED
static const float k_CscMatrix_Bt2020[CSC_MATRIX_RAW_ELEMENT_COUNT] = {
    1.0f, 1.0f, 1.0f,
    0.0f, -0.1646f, 1.8814f,
    1.4746f, -0.5714f, 0.0f,
};
#else
static const float k_CscMatrix_Bt2020Lim[CSC_MATRIX_RAW_ELEMENT_COUNT] = {
    1.1644f, 1.1644f, 1.1644f,
    0.0f, -0.1874f, 2.1418f,
    1.6781f, -0.6505f, 0.0f,
};
static const float k_Offsets_Lim[OFFSETS_ELEMENT_COUNT] = { 16.0f / 255.0f, 128.0f / 255.0f, 128.0f / 255.0f };
#endif

typedef struct _CSC_CONST_BUF {
  // CscMatrix value from above but packed appropriately
  float cscMatrix[CSC_MATRIX_PACKED_ELEMENT_COUNT];
  // YUV offset values from above
  float offsets[OFFSETS_ELEMENT_COUNT];
  // Padding float to be a multiple of 16 bytes
  float padding;
} CSC_CONST_BUF;

struct VERTEX { float x, y; float tu, tv; };

static CSC_CONST_BUF bindColorConversion(int colorspace, bool fullRange) {
  CSC_CONST_BUF constBuf = {};
  int bits = (colorspace == COLORSPACE_REC_2020) ? 10 : 8;
  (void)fullRange; (void)bits;
#ifdef FIXED
  const float* rawCscMatrix = k_CscMatrix_Bt2020;
  int range = (1 << bits);
  double yMin = (fullRange ? 0 : (16 << (bits - 8)));
  double yMax = (fullRange ? (range - 1) : (235 << (bits - 8)));
  double yScale = (range - 1) / (yMax - yMin);
  double uvMin = (fullRange ? 0 : (16 << (bits - 8)));
  double uvMax = (fullRange ? (range - 1) : (240 << (bits - 8)));
  double uvScale = (range - 1) / (uvMax - uvMin);
  // Calculate YUV offsets
  constBuf.offsets[0] = yMin / (double)(range - 1);
  constBuf.offsets[1] = (range / 2) / (double)(range - 1);
  constBuf.offsets[2] = (range / 2) / (double)(range - 1);
  // We need to adjust our raw CSC matrix to be column-major and with float3 vectors
  // padded with a float in between each of them to adhere to HLSL requirements.
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      constBuf.cscMatrix[i * 4 + j] = rawCscMatrix[j * 3 + i];
      // Scale the color matrix according to the color range
      constBuf.cscMatrix[i * 4 + j] *= (j == 0) ? yScale : uvScale;
    }
#else
  const float* rawCscMatrix = k_CscMatrix_Bt2020Lim;   // fullRange ? k_CscMatrix_Bt2020Full : ...
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      constBuf.cscMatrix[i * 4 + j] = rawCscMatrix[j * 3 + i];
  // No adjustments are needed to the float[3] array of offsets, so it can just
  // be copied with memcpy().
  memcpy(constBuf.offsets, k_Offsets_Lim, sizeof(constBuf.offsets));
#endif
  return constBuf;
}

int main(int argc, char** argv) {
  // HDR10 output: the swapchain is 10-bit (DXGI_FORMAT_R10G10B10A2_UNORM, PQ)
  AppDesc d; d.swap_format = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
  App a = make_app(argc, argv, d);
  auto sh = load_shader(a, "shaders.spv");

  // decoded Y410 frame: 10-bit limited range BT.2020 bars (Y in [64, 940], U/V centred on 512)
  const uint32_t W = 256, H = 64;
  Image videoTex = make_image(a, W, H, VK_FORMAT_A2B10G10R10_UNORM_PACK32,
                              VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, "decoder_output (Y410)");
  Buffer staging = make_buffer(a, W * H * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "frame_upload");
  const uint32_t bars[4][3] = {{64, 512, 512}, {502, 512, 512}, {940, 512, 512}, {300, 400, 700}};   // Y, U, V
  for (uint32_t y = 0; y < H; y++)
    for (uint32_t x = 0; x < W; x++) {
      const uint32_t* b = bars[x * 4 / W];
      ((uint32_t*)staging.map)[y * W + x] = b[1] | (b[0] << 10) | (b[2] << 20) | (3u << 30);   // U | Y | V | A
    }
  {
    VkCommandBuffer c = alloc_cmd(a, a.gfx_pool); begin(c);
    image_barrier(c, videoTex.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {W, H, 1};
    vkCmdCopyBufferToImage(c, staging.buf, videoTex.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    image_barrier(c, videoTex.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    VK(vkEndCommandBuffer(c)); submit(a.gfx, c, {}, {}, VK_NULL_HANDLE); vkQueueWaitIdle(a.gfx);
  }

  // frame colorspace from AVFrame: AVCOL_SPC_BT2020_NCL, AVCOL_RANGE_MPEG
  CSC_CONST_BUF constBuf = bindColorConversion(COLORSPACE_REC_2020, /*fullRange=*/false);
  Buffer cscBuf = make_buffer(a, sizeof constBuf, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "m_VideoPixelShaderConstantBuffer");
  memcpy(cscBuf.map, &constBuf, sizeof constBuf);

  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "video_srv_set");
  write_image(a, set, 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, videoTex.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  write_image(a, set, 1, VK_DESCRIPTOR_TYPE_SAMPLER, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED);
  write_buffer(a, set, 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, cscBuf);

  GfxDesc gd; gd.vs = gd.fs = sh; gd.vs_entry = "vertMain"; gd.fs_entry = "fragMain";
  gd.sets = {setl}; gd.colors = {a.swap_format}; gd.vertex_stride = sizeof(VERTEX);
  gd.attributes = {{0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(VERTEX, x)}, {1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(VERTEX, tu)}};
  Pipeline video = make_graphics(a, gd, "video_pipeline (y410)");
  const VERTEX verts[6] = {{-1, -1, 0, 0}, {1, -1, 1, 0}, {-1, 1, 0, 1}, {-1, 1, 0, 1}, {1, -1, 1, 0}, {1, 1, 1, 1}};
  Buffer vb = make_buffer(a, sizeof verts, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "m_VideoVertexBuffer");
  memcpy(vb.map, verts, sizeof verts);

  run_frames(a, frames_from_env(2), 1, [&](VkCommandBuffer c, uint32_t idx, int) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, video.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, video.layout, 0, 1, &set, 0, nullptr);
    VkDeviceSize off = 0; vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &off);
    vkCmdDraw(c, 6, 1, 0, 0);
    vkCmdEndRendering(c);
  });
  return 0;
}
