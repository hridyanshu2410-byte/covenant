// Distilled from VTK merge request !13006 (https://gitlab.kitware.com/vtk/vtk/-/merge_requests/13006,
// commit 1b583cd "Fixed WebGPU LightInfo alignment bug", merged 2026-04-06).
// The WebGPU renderer packs its lights into a storage buffer read as
//   struct SceneLights { count: u32, values: array<SceneLight> }
// where SceneLight is 80 bytes and `values` starts at offset 16. On the CPU,
// vtkWebGPULight::LightInfo started with `vtkTypeUInt8 Pad[12]` ("so that Type
// begins at n modulo 16 byte"), making every cached entry 92 bytes, and
// vtkWebGPURenderer::WriteLightsBuffer() wrote the 4-byte count followed by
// the entries back to back. The first light landed at offset 16 as intended,
// but each later one drifted by another 12 bytes: with a light kit, every
// light after the first was read with shifted fields (colours read as
// directions, etc.). The fix removes Pad[12] from LightInfo (80 bytes) and
// writes the 12 alignment bytes once after the count.
// Port: VTK/Dawn -> C++/Vulkan; the shading is a reduced Blinn-Phong over the
// lights, the actor is a small cube, the light set is a 3-light kit
// (key, fill, headlight) in view coordinates.
#include "../../common/app.hpp"
#include <cmath>
using namespace bench;

// vtkWebGPULight::LightInfo
struct LightInfo {
#ifndef FIXED
  uint8_t Pad[12] = {};    // so that Type begins at n modulo 16 byte. LightCount,
                           // a 4-byte integer is the first element in lights ssbo.
#endif
  // 0 : deferred, 1 : headlight, 2 : lightkit, 3 : positional
  uint32_t Type = 0;
  // 0 : not positional, 1 : positional
  uint32_t Positional = 0;
  float ConeAngle = 0;
  float Exponent = 0;
  float Color[4] = {};
  float DirectionVC[4] = {};   // normalized
  float PositionVC[4] = {};
  float Attenuation[4] = {};
};

struct SceneTransform { float view[16]; float projection[16]; };
struct ActorTransform { float world[16]; float normal[3][4]; };
struct Vertex { float position[3]; float normal[3]; };

static void normalize3(float* v) { float l = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); for (int i = 0; i < 3; i++) v[i] /= l; }

// vtkWebGPULight::CacheLightInformation(renderer, viewTransform)
static LightInfo CacheLightInformation(uint32_t type, const float dir[3], const float color[3], float intensity) {
  LightInfo info;
  info.Type = type; info.Positional = 0; info.ConeAngle = 30.f; info.Exponent = 1.f;
  for (int i = 0; i < 3; i++) { info.Color[i] = color[i] * intensity; info.DirectionVC[i] = dir[i]; }
  normalize3(info.DirectionVC);
  info.Attenuation[0] = 1.f;
  return info;
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");

  // vtkLightKit (key, fill) + headlight, already in view coordinates
  const float key_dir[3] = {-0.3f, -0.6f, -0.75f}, fill_dir[3] = {0.6f, -0.1f, -0.8f}, head_dir[3] = {0, 0, -1};
  const float white[3] = {1, 1, 1}, warm[3] = {1.0f, 0.93f, 0.85f}, cool[3] = {0.85f, 0.9f, 1.0f};
  std::vector<LightInfo> lights = {CacheLightInformation(2, key_dir, warm, 0.75f),
                                   CacheLightInformation(2, fill_dir, cool, 0.25f),
                                   CacheLightInformation(1, head_dir, white, 0.3f)};

  // vtkWebGPURenderer::WriteLightsBuffer()
  const uint32_t count = (uint32_t)lights.size();
  const size_t size = sizeof(LightInfo);   // vtkWebGPULight::GetCacheSizeBytes()
  std::vector<uint8_t> stage;
  size_t wroteBytes = 0;
#ifdef FIXED
  constexpr size_t kLightArrayOffset = 16;  // offset of values[] in SceneLights
  stage.resize(kLightArrayOffset + count * size, 0);
  memcpy(stage.data(), &count, sizeof(uint32_t));
  wroteBytes = kLightArrayOffset;
#else
  stage.resize(sizeof(uint32_t) + count * size);
  memcpy(&stage[wroteBytes], &count, sizeof(uint32_t));
  wroteBytes += sizeof(uint32_t);
#endif
  for (const auto& light : lights) { memcpy(&stage[wroteBytes], &light, size); wroteBytes += size; }
  // CreateBuffers(): SceneLightsBuffer sized for the shader's SceneLights (count + 8 lights)
  Buffer sceneLights = make_buffer(a, 16 + 8 * 80, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, "SceneLights");
  memset(sceneLights.map, 0, 16 + 8 * 80);
  memcpy(sceneLights.map, stage.data(), std::min<size_t>(stage.size(), 16 + 8 * 80));

  // camera at (0, 0, 4) looking at the origin, 30 degree view angle
  SceneTransform st{};
  for (int i = 0; i < 16; i++) st.view[i] = (i % 5 == 0) ? 1.f : 0.f;
  st.view[14] = -4.f;
  float aspect = (float)a.extent.width / a.extent.height, t = tanf(15.f * 3.14159265f / 180.f), n = 0.1f, f = 100.f;
  st.projection[0] = 1 / (aspect * t); st.projection[5] = -1 / t; st.projection[10] = f / (n - f); st.projection[11] = -1;
  st.projection[14] = -(f * n) / (f - n);
  Buffer stBuf = make_buffer(a, sizeof st, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "SceneTransform");
  memcpy(stBuf.map, &st, sizeof st);
  ActorTransform at{};
  float ry = 0.6f, rx = 0.4f;   // actor orientation
  float cy = cosf(ry), sy = sinf(ry), cx = cosf(rx), sx = sinf(rx);
  float R[3][3] = {{cy, sx * sy, -cx * sy}, {0, cx, sx}, {sy, -sx * cy, cx * cy}};   // columns
  for (int c = 0; c < 3; c++) for (int r = 0; r < 3; r++) { at.world[c * 4 + r] = R[c][r]; at.normal[c][r] = R[c][r]; }
  at.world[15] = 1;
  Buffer atBuf = make_buffer(a, sizeof at, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "ActorTransform");
  memcpy(atBuf.map, &at, sizeof at);

  // vtkCubeSource
  std::vector<Vertex> verts;
  for (int axis = 0; axis < 3; axis++)
    for (int s = -1; s <= 1; s += 2) {
      float N[3] = {0, 0, 0}; N[axis] = (float)s;
      int u = (axis + 1) % 3, v = (axis + 2) % 3;
      float c[4][3];
      for (int k = 0; k < 4; k++) {
        c[k][axis] = 0.5f * s; c[k][u] = (k == 1 || k == 2) ? 0.5f : -0.5f; c[k][v] = (k >= 2) ? 0.5f : -0.5f;
      }
      int idx[6] = {0, 1, 2, 0, 2, 3};
      for (int k : idx) verts.push_back({{c[k][0], c[k][1], c[k][2]}, {N[0], N[1], N[2]}});
    }
  Buffer vb = make_buffer(a, verts.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "cube points+normals");
  memcpy(vb.map, verts.data(), verts.size() * sizeof(Vertex));

  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}});
  auto set = alloc_set(a, setl, "renderer + actor bind group");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stBuf);
  write_buffer(a, set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, sceneLights);
  write_buffer(a, set, 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, atBuf);
  Image depth = make_image(a, a.extent.width, a.extent.height, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, "depth");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.vs_entry = "vertexMain"; gd.fs_entry = "fragmentMain";
  gd.sets = {setl}; gd.colors = {a.swap_format}; gd.depth = VK_FORMAT_D32_SFLOAT; gd.vertex_stride = sizeof(Vertex);
  gd.attributes = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)}, {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)}};
  Pipeline pl = make_graphics(a, gd, "vtkWebGPUActor triangles");

  run_frames(a, frames_from_env(2), 1, [&](VkCommandBuffer c, uint32_t idx, int) {
    depth_barrier(c, depth.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    begin_rendering(a, c, {a.swap_views[idx]}, depth.view);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pl.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pl.layout, 0, 1, &set, 0, nullptr);
    VkDeviceSize off = 0; vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &off);
    vkCmdDraw(c, (uint32_t)verts.size(), 1, 0, 0);
    vkCmdEndRendering(c);
  });
  return 0;
}
