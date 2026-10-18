// Distilled from sotrh/learn-wgpu #517 ("Wrong view projection matrices",
// introduced by PR #478). Camera::build_view_projection_matrix() returns
// OPENGL_TO_WGPU_MATRIX * proj * view, where proj is cgmath's OpenGL-style
// perspective (clip z in [-w, w]) and OPENGL_TO_WGPU_MATRIX should remap z to
// WebGPU's [0, w]: z' = 0.5 z + 0.5 w. cgmath::Matrix4::new takes its
// arguments column by column; PR #478 transposed the constant, writing
//   1,0,0,0,  0,1,0,0,  0,0,0.5,0.5,  0,0,0,1
// which computes z' = 0.5 z and w' = w + 0.5 z. A point on the far plane then
// gets depth 0.33 instead of 1 and x/y shrink by a third (smaller field of
// view). The issue's fix reverts to
//   1,0,0,0,  0,1,0,0,  0,0,0.5,0,  0,0,0.5,1.
// (Rust/cgmath ported to C++; the Mat4 below keeps cgmath's column-major
// constructor order and memory layout, which is what bytemuck uploads.)
#include "../../common/app.hpp"
#include <cmath>
using namespace bench;

struct Mat4 { float m[16]; };               // column-major: m[col * 4 + row]
static Mat4 mat4_new(std::initializer_list<float> cols) { Mat4 r{}; int i = 0; for (float v : cols) r.m[i++] = v; return r; }
static Mat4 mul(const Mat4& a, const Mat4& b) {
  Mat4 r{};
  for (int c = 0; c < 4; c++)
    for (int rr = 0; rr < 4; rr++) {
      float s = 0; for (int k = 0; k < 4; k++) s += a.m[k * 4 + rr] * b.m[c * 4 + k];
      r.m[c * 4 + rr] = s;
    }
  return r;
}
// cgmath::perspective(Deg(fovy), aspect, znear, zfar) (OpenGL convention)
static Mat4 perspective(float fovy_deg, float aspect, float n, float f) {
  float t = 1.0f / std::tan(fovy_deg * 3.14159265f / 360.0f);
  return mat4_new({t / aspect, 0, 0, 0,  0, t, 0, 0,  0, 0, (f + n) / (n - f), -1,  0, 0, 2 * f * n / (n - f), 0});
}
// cgmath::Matrix4::look_at_rh(eye, target, up)
static Mat4 look_at_rh(const float e[3], const float t[3], const float up[3]) {
  float fz[3] = {t[0] - e[0], t[1] - e[1], t[2] - e[2]};
  float l = std::sqrt(fz[0] * fz[0] + fz[1] * fz[1] + fz[2] * fz[2]); for (float& v : fz) v /= l;
  float s[3] = {fz[1] * up[2] - fz[2] * up[1], fz[2] * up[0] - fz[0] * up[2], fz[0] * up[1] - fz[1] * up[0]};
  l = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]); for (float& v : s) v /= l;
  float u[3] = {s[1] * fz[2] - s[2] * fz[1], s[2] * fz[0] - s[0] * fz[2], s[0] * fz[1] - s[1] * fz[0]};
  auto dot = [](const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
  return mat4_new({s[0], u[0], -fz[0], 0,  s[1], u[1], -fz[1], 0,  s[2], u[2], -fz[2], 0,
                   -dot(s, e), -dot(u, e), dot(fz, e), 1});
}

#ifdef FIXED
static const Mat4 OPENGL_TO_WGPU_MATRIX = mat4_new({
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0});
#else
static const Mat4 OPENGL_TO_WGPU_MATRIX = mat4_new({
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0});
#endif

struct Camera { float eye[3], target[3], up[3]; float aspect, fovy, znear, zfar; };
static Mat4 build_view_projection_matrix(const Camera& c) {
  Mat4 view = look_at_rh(c.eye, c.target, c.up);
  Mat4 proj = perspective(c.fovy, c.aspect, c.znear, c.zfar);
  return mul(OPENGL_TO_WGPU_MATRIX, mul(proj, view));
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  Image depth = make_image(a, a.extent.width, a.extent.height, VK_FORMAT_D32_SFLOAT,
                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, "depth_texture");
  Buffer cam = make_buffer(a, 64, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "camera_buffer");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}});
  auto camera_bind_group = alloc_set(a, setl, "camera_bind_group");
  write_buffer(a, camera_bind_group, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, cam);
  // VERTICES: the tutorial's pentagon
  float verts[] = {-0.0868241f, 0.49240386f, 0, 0.5f, 0, 0.5f,   -0.49513406f, 0.06958647f, 0, 0.5f, 0, 0.5f,
                   -0.21918549f, -0.44939706f, 0, 0.5f, 0, 0.5f, 0.35966998f, -0.3473291f, 0, 0.5f, 0, 0.5f,
                   0.44147372f, 0.2347359f, 0, 0.5f, 0, 0.5f};
  uint16_t indices[] = {0, 1, 4, 1, 2, 4, 2, 3, 4};
  Buffer vb = make_buffer(a, sizeof verts, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "vertex_buffer");
  Buffer ib = make_buffer(a, sizeof indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, true, "index_buffer");
  memcpy(vb.map, verts, sizeof verts); memcpy(ib.map, indices, sizeof indices);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format}; gd.depth = VK_FORMAT_D32_SFLOAT;
  gd.vertex_stride = 24; gd.attributes = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12}};
  Pipeline render_pipeline = make_graphics(a, gd, "render_pipeline");

  Camera camera{{0, 1, 2}, {0, 0, 0}, {0, 1, 0}, (float)a.extent.width / a.extent.height, 45.0f, 0.1f, 100.0f};
  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    depth_barrier(c, depth.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    begin_rendering(a, c, {a.swap_views[idx]}, depth.view);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, render_pipeline.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, render_pipeline.layout, 0, 1, &camera_bind_group, 0, nullptr);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &off);
    vkCmdBindIndexBuffer(c, ib.buf, 0, VK_INDEX_TYPE_UINT16);
    vkCmdDrawIndexed(c, 9, 1, 0, 0, 0);
    vkCmdEndRendering(c);
  }, [&](int f) {
    // update(): queue.write_buffer(&camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]))
    Mat4 vp = build_view_projection_matrix(camera);
    memcpy(cam.map, vp.m, sizeof vp.m);
  });
}
