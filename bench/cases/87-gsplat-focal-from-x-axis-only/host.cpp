// Distilled from playcanvas/engine PR #9486 ("Use a per-axis focal length
// for the gsplat footprint", merged 2026-09-23). The gsplat vertex shader
// built the EWA Jacobian with one focal length for both screen axes,
//   float focal = viewport_size.x * projMat00;  J = mat3(J1, 0, J2.x, 0, J1, J2.y, 0, 0, 0)
// which is only right when viewport_w * P[0][0] == viewport_h * P[1][1],
// i.e. when the viewport's pixel aspect equals the projection's aspect.
// With a manual camera aspect ratio or side-by-side stereo (each eye a
// half-width viewport), splat footprints are squashed along one axis
// (stippling, cracks). The fix uses focalX = |viewport_w * P00| and
// focalY = |viewport_h * P11| (new projMat11 input; focalY uniform for the
// GPU-sort compute projector). Here: a camera with a fixed 16:9 aspect
// rendering into the left half of a 640x480 target (a 320x480 eye
// viewport). Splat data and the reduced EWA code are ours.
#include "../../common/app.hpp"
#include <cmath>
using namespace bench;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  const int kSplats = 64;
  Buffer splats = make_buffer(a, kSplats * 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, "gsplat_data");
  float* s = (float*)splats.map;
  for (int i = 0; i < kSplats; i++) {
    float* r = s + i * 16;
    r[0] = -1.5f + 3.0f * (i % 8) / 7.0f; r[1] = -1.0f + 2.0f * (i / 8) / 7.0f; r[2] = 0; r[3] = 1;
    r[4] = 0.02f; r[5] = 0; r[6] = 0; r[7] = 0;        // isotropic covariance (round splats)
    r[8] = 0.02f; r[9] = 0; r[10] = 0.02f; r[11] = 0;
    r[12] = 0.9f; r[13] = 0.6f; r[14] = 0.3f; r[15] = 0.9f;
  }
  // camera: fov 60 deg vertical, aspectRatio fixed to 16:9 (CameraComponent.aspectRatioMode = MANUAL)
  const float fy = 1.0f / std::tan(30.0f * 3.14159265f / 180.0f), aspect = 16.0f / 9.0f, n = 0.1f, fr = 100.0f;
  struct ViewU { float view[16], proj[16], viewport_size[2], pad[2]; } u{};
  float view[16] = {1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 5, 1};           // camera 5 units in front (view z > 0 forward)
  float proj[16] = {fy / aspect, 0, 0, 0,  0, -fy, 0, 0,  0, 0, fr / (fr - n), 1,  0, 0, -n * fr / (fr - n), 0};
  memcpy(u.view, view, sizeof view); memcpy(u.proj, proj, sizeof proj);
  const float eyeW = a.extent.width / 2.0f, eyeH = (float)a.extent.height;     // left eye viewport 320x480
  u.viewport_size[0] = eyeW; u.viewport_size[1] = eyeH;
  Buffer vub = make_buffer(a, sizeof u, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "view_uniforms");
  memcpy(vub.map, &u, sizeof u);
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}});
  auto set = alloc_set(a, setl, "gsplat_bindings");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, splats);
  write_buffer(a, set, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, vub);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline gs = make_graphics(a, gd, "gsplat");

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    VkViewport vp{0, 0, eyeW, eyeH, 0, 1};                 // left eye
    vkCmdSetViewport(c, 0, 1, &vp);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, gs.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, gs.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, kSplats, 0, 0);
    vkCmdEndRendering(c);
  });
  return 0;
}
