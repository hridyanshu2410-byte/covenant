// Distilled from KhronosGroup/Vulkan-Tutorial PR #211 ("Fix sync issues for
// color and depth attachments", merged 2025-11-03). From chapter 27 on, every
// frame's command buffer moves the single shared depth image from UNDEFINED to
// the depth attachment layout with
//   srcStageMask = TOP_OF_PIPE, srcAccessMask = 0,
//   dstStageMask = EARLY | LATE_FRAGMENT_TESTS, dstAccess = DS read | write.
// With MAX_FRAMES_IN_FLIGHT = 2 the previous frame may still be writing that
// depth image in its fragment tests. TOP_OF_PIPE / no access neither waits
// for those tests nor makes their writes available, so the layout transition
// (itself a write) and this frame's depth clear race the previous frame's
// depth writes (WRITE_AFTER_WRITE in synchronization validation). The fix
// passes srcStage = EARLY | LATE_FRAGMENT_TESTS and srcAccess =
// DEPTH_STENCIL_ATTACHMENT_WRITE through a reworked transition_image_layout().
// Port: C++ RAII/vulkan-hpp -> C API; the swapchain transitions come from the
// bench frame loop; the depth layout is DEPTH_ATTACHMENT_OPTIMAL in both
// variants (the PR also renamed DEPTH_STENCIL_ATTACHMENT_OPTIMAL to it).
#include "../../common/app.hpp"
#include <cmath>
using namespace bench;

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

struct Vertex { float pos[3]; float color[3]; };
struct UniformBufferObject { float model[16], view[16], proj[16]; };

static void identity(float* m) { for (int i = 0; i < 16; i++) m[i] = (i % 5 == 0) ? 1.f : 0.f; }
static void rotate_z(float* m, float a) { identity(m); m[0] = cosf(a); m[1] = sinf(a); m[4] = -sinf(a); m[5] = cosf(a); }
static void look_at(float* m, const float e[3], const float c[3], const float up[3]) {
  float f[3] = {c[0] - e[0], c[1] - e[1], c[2] - e[2]};
  float fl = sqrtf(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]); for (float& v : f) v /= fl;
  float s[3] = {f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]};
  float sl = sqrtf(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]); for (float& v : s) v /= sl;
  float u[3] = {s[1] * f[2] - s[2] * f[1], s[2] * f[0] - s[0] * f[2], s[0] * f[1] - s[1] * f[0]};
  identity(m);
  m[0] = s[0]; m[4] = s[1]; m[8] = s[2];
  m[1] = u[0]; m[5] = u[1]; m[9] = u[2];
  m[2] = -f[0]; m[6] = -f[1]; m[10] = -f[2];
  m[12] = -(s[0] * e[0] + s[1] * e[1] + s[2] * e[2]);
  m[13] = -(u[0] * e[0] + u[1] * e[1] + u[2] * e[2]);
  m[14] = (f[0] * e[0] + f[1] * e[1] + f[2] * e[2]);
}
static void perspective_rh_zo(float* m, float fovy, float aspect, float n, float f) {
  for (int i = 0; i < 16; i++) m[i] = 0;
  float t = tanf(fovy / 2);
  m[0] = 1 / (aspect * t); m[5] = 1 / t; m[10] = f / (n - f); m[11] = -1; m[14] = -(f * n) / (f - n);
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  const VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
  Image depth = make_image(a, a.extent.width, a.extent.height, depthFormat,
                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, "depthImage");

  const Vertex vertices[] = {
      {{-0.5f, -0.5f, 0.0f}, {1, 0, 0}}, {{0.5f, -0.5f, 0.0f}, {0, 1, 0}},
      {{0.5f, 0.5f, 0.0f}, {0, 0, 1}},   {{-0.5f, 0.5f, 0.0f}, {1, 1, 1}},
      {{-0.5f, -0.5f, -0.5f}, {1, 0, 0}}, {{0.5f, -0.5f, -0.5f}, {0, 1, 0}},
      {{0.5f, 0.5f, -0.5f}, {0, 0, 1}},   {{-0.5f, 0.5f, -0.5f}, {1, 1, 1}}};
  const uint16_t indices[] = {0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4};
  Buffer vb = make_buffer(a, sizeof vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "vertexBuffer");
  Buffer ib = make_buffer(a, sizeof indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, true, "indexBuffer");
  memcpy(vb.map, vertices, sizeof vertices); memcpy(ib.map, indices, sizeof indices);

  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}});
  std::vector<Buffer> uniformBuffers; std::vector<VkDescriptorSet> sets;
  for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    uniformBuffers.push_back(make_buffer(a, sizeof(UniformBufferObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true,
                                         "uniformBuffers[" + std::to_string(i) + "]"));
    sets.push_back(alloc_set(a, setl, "descriptorSets[" + std::to_string(i) + "]"));
    write_buffer(a, sets[i], 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, uniformBuffers[i]);
  }

  GfxDesc gd; gd.vs = gd.fs = sh; gd.vs_entry = "vertMain"; gd.fs_entry = "fragMain";
  gd.sets = {setl}; gd.colors = {a.swap_format}; gd.depth = depthFormat;
  gd.vertex_stride = sizeof(Vertex);
  gd.attributes = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)},
                   {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)}};
  Pipeline graphicsPipeline = make_graphics(a, gd, "graphicsPipeline");

  int currentFrame = 0;
  run_frames(a, frames_from_env(4), MAX_FRAMES_IN_FLIGHT, [&](VkCommandBuffer c, uint32_t imageIndex, int f) {
    // recordCommandBuffer(): transition depth image to depth attachment optimal layout
#ifdef FIXED
    depth_barrier(c, depth.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
#else
    depth_barrier(c, depth.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
#endif
    begin_rendering(a, c, {a.swap_views[imageIndex]}, depth.view, 1.0f);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline.pipe);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &off);
    vkCmdBindIndexBuffer(c, ib.buf, 0, VK_INDEX_TYPE_UINT16);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline.layout, 0, 1, &sets[currentFrame], 0, nullptr);
    vkCmdDrawIndexed(c, 12, 1, 0, 0, 0);
    vkCmdEndRendering(c);
  }, [&](int f) {
    currentFrame = f % MAX_FRAMES_IN_FLIGHT;
    // updateUniformBuffer(currentFrame)
    UniformBufferObject ubo{};
    rotate_z(ubo.model, f * 0.25f);
    const float eye[3] = {2, 2, 2}, center[3] = {0, 0, 0}, up[3] = {0, 0, 1};
    look_at(ubo.view, eye, center, up);
    perspective_rh_zo(ubo.proj, 45.0f * 3.14159265f / 180.0f, (float)a.extent.width / a.extent.height, 0.1f, 10.0f);
    ubo.proj[5] *= -1;
    memcpy(uniformBuffers[currentFrame].map, &ubo, sizeof ubo);
  });
  return 0;
}
