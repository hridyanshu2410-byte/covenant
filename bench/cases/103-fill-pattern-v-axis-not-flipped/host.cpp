// Distilled from visgl/deck.gl PR #10550 ("fix(fill-style): correct pattern
// orientation by flipping texture y-axis", merged 2026-09-01). The
// FillStyleExtension fills polygons with a pattern taken from an icon atlas.
// The fragment code derives patternUV from the fragment's common-space
// position (fill_uv = geometry.position.xy), in which +y points north, and
// mapped it straight into the atlas frame:
//   vec2 texCoords = fill_patternBounds.xy + fill_patternBounds.zw * patternUV;
// Atlas texture coordinates run with +v downward from the image's top row,
// so every pattern was drawn upside down (an arrow icon pointing north came
// out pointing south). The fix samples with vec2(patternUV.x, 1.0 - patternUV.y).
// Port: GLSL module injection / luma.gl -> one Slang vertex + fragment pair;
// the layer's attributes are per-vertex here, the polygon is pre-projected to
// common space, and the atlas is a 2-icon RGBA8 image uploaded top row first
// (as luma.gl uploads ImageBitmaps).
#include "../../common/app.hpp"
#include <cmath>
using namespace bench;

struct ProjectUniforms { float viewProjectionMatrix[16]; float opacity; float _pad[3]; };
struct FillUniforms { float patternTextureSize[2]; uint32_t patternEnabled; uint32_t patternMask;
                      float uvCoordinateOrigin[2]; float uvCoordinateOrigin64Low[2]; };
struct PolygonVertex {
  float positions[3]; float fillColors[4];
  float fillPatternFrames[4]; float fillPatternScales; float fillPatternOffsets[2]; float fillPatternBackgroundColors[4];
};

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");

  // pattern atlas (fillPatternAtlas): "arrow" at x=[0,32), "hatch" at x=[32,64); 32 px tall
  const uint32_t AW = 64, AH = 32;
  Image atlas = make_image(a, AW, AH, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                           "fillPatternAtlas");
  Buffer staging = make_buffer(a, AW * AH * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "fillPatternAtlas staging");
  uint8_t* px = (uint8_t*)staging.map;
  for (uint32_t y = 0; y < AH; y++)        // row 0 is the top of the image
    for (uint32_t x = 0; x < AW; x++) {
      bool on;
      if (x < 32) { int dx = abs((int)x - 16); on = (y >= 4 && y < 14 && dx <= (int)(y - 4)) || (y >= 14 && y < 28 && dx <= 3); }  // arrow pointing up
      else on = ((x - 32 + y) % 8) < 2;                                                                             // diagonal hatch
      uint8_t* p = px + (y * AW + x) * 4; p[0] = p[1] = p[2] = 255; p[3] = on ? 255 : 0;
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

  // one polygon (a city block) around the coordinate origin, common-space units
  const float s = 0.02f;   // half size in common units (~ a few hundred metres at zoom 12)
  const float ring[4][2] = {{-s, -s}, {s, -s}, {s, s}, {-s, s}};
  const int tri[6] = {0, 1, 2, 2, 3, 0};
  std::vector<PolygonVertex> verts;
  for (int k : tri) {
    PolygonVertex v{};
    v.positions[0] = ring[k][0]; v.positions[1] = ring[k][1]; v.positions[2] = 0;
    v.fillColors[0] = 0.0f; v.fillColors[1] = 0.55f; v.fillColors[2] = 0.85f; v.fillColors[3] = 1.0f;
    float frame[4] = {0, 0, 32, 32};            // getFillPattern: 'arrow' -> atlas mapping
    memcpy(v.fillPatternFrames, frame, sizeof frame);
    v.fillPatternScales = 2.0f;                  // getFillPatternScale
    v.fillPatternOffsets[0] = 0; v.fillPatternOffsets[1] = 0;
    v.fillPatternBackgroundColors[0] = 0.1f; v.fillPatternBackgroundColors[1] = 0.1f;
    v.fillPatternBackgroundColors[2] = 0.15f; v.fillPatternBackgroundColors[3] = 1.0f;
    verts.push_back(v);
  }
  Buffer vb = make_buffer(a, verts.size() * sizeof(PolygonVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "polygon attributes");
  memcpy(vb.map, verts.data(), verts.size() * sizeof(PolygonVertex));

  // project module: orthographic top-down view of common space (+y north up on screen)
  ProjectUniforms pu{};
  float aspect = (float)a.extent.width / a.extent.height, half = 0.03f;
  float P[16] = {1 / (half * aspect), 0, 0, 0, 0, -1 / half, 0, 0, 0, 0, 1, 0, 0, 0, 0.5f, 1};   // Vulkan clip: y down
  memcpy(pu.viewProjectionMatrix, P, sizeof P); pu.opacity = 1.0f;
  Buffer projBuf = make_buffer(a, sizeof pu, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "projectUniforms");
  memcpy(projBuf.map, &pu, sizeof pu);
  FillUniforms fu{{(float)AW, (float)AH}, 1, 1, {0.f, 0.f}, {0.f, 0.f}};   // fillPatternMask: true
  Buffer fillBuf = make_buffer(a, sizeof fu, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "fillUniforms");
  memcpy(fillBuf.map, &fu, sizeof fu);

  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "solid-polygon-layer bindings");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, projBuf);
  write_buffer(a, set, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, fillBuf);
  write_image(a, set, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, atlas.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  GfxDesc gd; gd.vs = gd.fs = sh; gd.vs_entry = "vertMain"; gd.fs_entry = "fragMain";
  gd.sets = {setl}; gd.colors = {a.swap_format}; gd.vertex_stride = sizeof(PolygonVertex);
  gd.attributes = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(PolygonVertex, positions)},
                   {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(PolygonVertex, fillColors)},
                   {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(PolygonVertex, fillPatternFrames)},
                   {3, 0, VK_FORMAT_R32_SFLOAT, offsetof(PolygonVertex, fillPatternScales)},
                   {4, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(PolygonVertex, fillPatternOffsets)},
                   {5, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(PolygonVertex, fillPatternBackgroundColors)}};
  Pipeline layer = make_graphics(a, gd, "solid-polygon-layer-fill");

  run_frames(a, frames_from_env(2), 1, [&](VkCommandBuffer c, uint32_t idx, int) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, layer.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, layer.layout, 0, 1, &set, 0, nullptr);
    VkDeviceSize off = 0; vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &off);
    vkCmdDraw(c, (uint32_t)verts.size(), 1, 0, 0);
    vkCmdEndRendering(c);
  });
  return 0;
}
