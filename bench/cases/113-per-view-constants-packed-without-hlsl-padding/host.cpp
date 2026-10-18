// Distilled from FlaxEngine/FlaxEngine issue #3342, fixed by commit 1087bd2
// ("Fix #3342 to properly place new param in Constant Buffer", 2025-08-14;
// the regression came with PR #3299, which added unscaled time). The per-view
// material constants are declared twice: in HLSL (MaterialCommon.hlsl,
// cbuffer ViewData) and in C++ (GPU_CB_STRUCT MaterialShaderDataPerView,
// which is PACK_BEGIN/PACK_END, i.e. byte-packed). After #3299 both had
//   float3 ViewDir; float UnscaledTimeParam; float ScaledTimeParam; float4 ViewInfo; ...
// The HLSL packing rules put ScaledTimeParam at the start of a new 16-byte
// register (offset 304) and therefore move ViewInfo to the next one (320);
// the packed C++ struct puts ViewInfo right after ScaledTimeParam at 308.
// From ViewInfo on, every per-view constant (ScreenSize, TemporalAAJitter,
// LargeWorldsChunkIndex/Size) was read 12 bytes off in every material. The
// fix keeps a single TimeParam in that slot and moves UnscaledTimeParam to
// the end behind a float3 ViewPadding0, on both sides.
// Port: D3D cbuffer -> Vulkan std140 uniform buffer (same offsets for this
// block); Flax math types -> plain float arrays; one surface material drawn
// on a quad with the editor's view setup.
#include "../../common/app.hpp"
#include <cmath>
using namespace bench;

#if defined(_MSC_VER)
#define PACK_BEGIN() __pragma(pack(push, 1))
#define PACK_END() __pragma(pack(pop))
#else
#define PACK_BEGIN() _Pragma("pack(push, 1)")
#define PACK_END() _Pragma("pack(pop)")
#endif
#define GPU_SHADER_DATA_ALIGNMENT 16
#define GPU_CB_STRUCT(_declaration) PACK_BEGIN() struct alignas(GPU_SHADER_DATA_ALIGNMENT) _declaration PACK_END()

struct Matrix { float M[16]; };
struct Float3 { float X, Y, Z; };
struct Float4 { float X, Y, Z, W; };

GPU_CB_STRUCT(MaterialShaderDataPerView {
    Matrix ViewMatrix;
    Matrix ViewProjectionMatrix;
    Matrix PrevViewProjectionMatrix;
    Matrix MainViewProjectionMatrix;
    Float4 MainScreenSize;
    Float3 ViewPos;
    float ViewFar;
    Float3 ViewDir;
#ifdef FIXED
    float TimeParam;
#else
    float UnscaledTimeParam;
    float ScaledTimeParam;
#endif
    Float4 ViewInfo;
    Float4 ScreenSize;
    Float4 TemporalAAJitter;
    Float3 LargeWorldsChunkIndex;
    float LargeWorldsChunkSize;
#ifdef FIXED
    Float3 ViewPadding0;
    float UnscaledTimeParam;
#endif
    });

struct RenderView {
  Matrix View, ViewProjection;
  Float3 Position, Direction;
  float Near, Far;
  Float4 ViewInfo, ScreenSize, TemporalAAJitter;
};

static Matrix mul(const Matrix& a, const Matrix& b) {
  Matrix r{};
  for (int c = 0; c < 4; c++) for (int rr = 0; rr < 4; rr++) {
    float s = 0; for (int k = 0; k < 4; k++) s += a.M[k * 4 + rr] * b.M[c * 4 + k]; r.M[c * 4 + rr] = s; }
  return r;
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  const float W = (float)a.extent.width, H = (float)a.extent.height;

  // RenderView for the editor viewport: camera at (0, 150, -400) looking at the origin (cm units)
  RenderView view{};
  view.Position = {0, 150, -400}; view.Direction = {0, -0.351f, 0.936f};
  view.Near = 10.f; view.Far = 40000.f;
  Matrix V{{1, 0, 0, 0, 0, 0.936f, -0.351f, 0, 0, 0.351f, 0.936f, 0, 0, 0, 0, 1}};
  V.M[12] = 0; V.M[13] = -(0.936f * 150 + 0.351f * -400); V.M[14] = -(-0.351f * 150 + 0.936f * -400);
  float t = tanf(30.f * 3.14159265f / 180.f), aspect = W / H;
  Matrix P{};   // left-handed perspective, depth 0..1, Vulkan y-down
  P.M[0] = 1 / (aspect * t); P.M[5] = -1 / t; P.M[10] = view.Far / (view.Far - view.Near); P.M[11] = 1;
  P.M[14] = -view.Near * view.Far / (view.Far - view.Near);
  view.View = V; view.ViewProjection = mul(P, V);
  view.ViewInfo = {1.f / P.M[0], 1.f / -P.M[5], view.Far / (view.Far - view.Near), (-view.Far * view.Near) / (view.Far - view.Near) / view.Far};
  view.ScreenSize = {W, H, 1.f / W, 1.f / H};
  view.TemporalAAJitter = {0, 0, 0, 0};

  Buffer viewCB = make_buffer(a, sizeof(MaterialShaderDataPerView), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "ViewData");
  struct DrawData { Matrix WorldMatrix; } drawData{};
  float s = 100.f;
  drawData.WorldMatrix = {{s, 0, 0, 0, 0, s, 0, 0, 0, 0, s, 0, 0, 0, 0, 1}};
  Buffer drawCB = make_buffer(a, sizeof drawData, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "DrawData");
  memcpy(drawCB.map, &drawData, sizeof drawData);

  // a quad facing the camera (-Z)
  struct ModelVertex { float pos[3]; float uv[2]; float n[3]; };
  const ModelVertex quad[6] = {{{-1, -1, 0}, {0, 1}, {0, 0, -1}}, {{1, -1, 0}, {1, 1}, {0, 0, -1}}, {{-1, 1, 0}, {0, 0}, {0, 0, -1}},
                               {{-1, 1, 0}, {0, 0}, {0, 0, -1}}, {{1, -1, 0}, {1, 1}, {0, 0, -1}}, {{1, 1, 0}, {1, 0}, {0, 0, -1}}};
  Buffer vb = make_buffer(a, sizeof quad, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "Plane LOD0 VB");
  memcpy(vb.map, quad, sizeof quad);

  auto setl = make_set_layout(a, {{1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}});
  auto set = alloc_set(a, setl, "material descriptors");
  write_buffer(a, set, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, viewCB);
  write_buffer(a, set, 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, drawCB);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.vs_entry = "VS"; gd.fs_entry = "PS_GBuffer";
  gd.sets = {setl}; gd.colors = {a.swap_format}; gd.vertex_stride = sizeof(ModelVertex);
  gd.attributes = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(ModelVertex, pos)},
                   {1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(ModelVertex, uv)},
                   {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(ModelVertex, n)}};
  Pipeline material = make_graphics(a, gd, "M_Stripes PS_GBuffer");

  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, material.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, material.layout, 0, 1, &set, 0, nullptr);
    VkDeviceSize off = 0; vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &off);
    vkCmdDraw(c, 6, 1, 0, 0);
    vkCmdEndRendering(c);
  }, [&](int f) {
    // IMaterial::BindParameters::BindViewData()
    MaterialShaderDataPerView cb{};
    cb.ViewMatrix = view.View;
    cb.ViewProjectionMatrix = view.ViewProjection;
    cb.PrevViewProjectionMatrix = view.ViewProjection;
    cb.MainViewProjectionMatrix = view.ViewProjection;
    cb.MainScreenSize = view.ScreenSize;
    cb.ViewPos = view.Position;
    cb.ViewFar = view.Far;
    cb.ViewDir = view.Direction;
#ifdef FIXED
    cb.TimeParam = 0.5f * f / 60.0f;           // Time::Draw.Time (time scale 0.5)
    cb.UnscaledTimeParam = f / 60.0f;          // Time::Draw.UnscaledTime
#else
    cb.UnscaledTimeParam = f / 60.0f;          // Time::Draw.UnscaledTime
    cb.ScaledTimeParam = 0.5f * f / 60.0f;     // Time::Draw.Time (time scale 0.5)
#endif
    cb.ViewInfo = view.ViewInfo;
    cb.ScreenSize = view.ScreenSize;
    cb.TemporalAAJitter = view.TemporalAAJitter;
    cb.LargeWorldsChunkIndex = {0, 0, 0};
    cb.LargeWorldsChunkSize = 0;
    memcpy(viewCB.map, &cb, sizeof cb);
  });
  return 0;
}
