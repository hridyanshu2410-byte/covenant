// Distilled from iced-rs/iced PR #3319 ("correctly use radians instead of
// degrees for fov", merged 2026-05-26). In the custom_shader example the
// Camera's default field of view was written as
//   fov_y: 45.0,
// and passed straight to glam::Mat4::perspective_rh(fov_y_radians, ...),
// which takes radians. With 45 rad the half angle is 22.5 rad, so the focal
// scale is cot(22.5) ~ 1.79 instead of cot(22.5 deg) ~ 2.41: a frustum of
// about 58 degrees instead of 45, and the cubes render visibly too small and
// distorted toward the edges. The fix:
//   fov_y: 45.0f32.to_radians(),
// Port: Rust/wgpu/glam -> C++/Vulkan with glam's perspective_rh and
// look_at_rh written out; OPENGL_TO_WGPU_MATRIX is replaced by the wgpu ->
// Vulkan clip correction (flip Y), the cube mesh is a plain unit cube.
#include "../../common/app.hpp"
#include <cmath>
using namespace bench;

struct Vec3 { float x, y, z; };
struct Mat4 { float m[16]; };   // column-major, like glam
static Vec3 sub(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static Vec3 cross(Vec3 a, Vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static Vec3 norm(Vec3 a) { float l = sqrtf(dot(a, a)); return {a.x / l, a.y / l, a.z / l}; }
static Mat4 mul(const Mat4& a, const Mat4& b) {
  Mat4 r{};
  for (int c = 0; c < 4; c++) for (int rr = 0; rr < 4; rr++) {
    float s = 0; for (int k = 0; k < 4; k++) s += a.m[k * 4 + rr] * b.m[c * 4 + k]; r.m[c * 4 + rr] = s; }
  return r;
}
// glam::Mat4::look_at_rh
static Mat4 look_at_rh(Vec3 eye, Vec3 center, Vec3 up) {
  Vec3 f = norm(sub(center, eye)), s = norm(cross(f, up)), u = cross(s, f);
  return {{s.x, u.x, -f.x, 0, s.y, u.y, -f.y, 0, s.z, u.z, -f.z, 0, -dot(s, eye), -dot(u, eye), dot(f, eye), 1}};
}
// glam::Mat4::perspective_rh(fov_y_radians, aspect_ratio, z_near, z_far): depth 0..1
static Mat4 perspective_rh(float fov_y_radians, float aspect_ratio, float z_near, float z_far) {
  float h = cosf(0.5f * fov_y_radians) / sinf(0.5f * fov_y_radians);
  float w = h / aspect_ratio, r = z_far / (z_near - z_far);
  return {{w, 0, 0, 0, 0, h, 0, 0, 0, 0, r, -1, 0, 0, r * z_near, 0}};
}
// wgpu clip space has +Y up; Vulkan's has +Y down
static const Mat4 WGPU_TO_VULKAN_MATRIX = {{1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}};

struct Camera {
  Vec3 eye{0.0f, 2.0f, 3.0f}, target{0, 0, 0}, up{0, 1, 0};
#ifdef FIXED
  float fov_y = 45.0f * 3.14159265f / 180.0f;
#else
  float fov_y = 45.0f;
#endif
  float near_ = 0.1f, far_ = 100.0f;
  Mat4 build_view_proj_matrix(float width, float height) const {
    float aspect_ratio = width / height;
    Mat4 view = look_at_rh(eye, target, up);
    Mat4 proj = perspective_rh(fov_y, aspect_ratio, near_, far_);
    return mul(WGPU_TO_VULKAN_MATRIX, mul(proj, view));
  }
};

struct Uniforms { Mat4 camera_proj; float camera_pos[4]; float light_color[4]; };
struct Vertex { float position[3]; float normal[3]; };
struct CubeRaw { float transformation[16]; };

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");

  // unit cube, 6 faces x 2 triangles
  std::vector<Vertex> verts;
  const float n[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  for (auto& f : n) {
    Vec3 N{f[0], f[1], f[2]};
    Vec3 t = fabsf(N.y) > 0.5f ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    Vec3 b = cross(N, t);
    Vec3 c[4] = {{N.x * .5f + (-t.x - b.x) * .5f, N.y * .5f + (-t.y - b.y) * .5f, N.z * .5f + (-t.z - b.z) * .5f},
                 {N.x * .5f + (t.x - b.x) * .5f, N.y * .5f + (t.y - b.y) * .5f, N.z * .5f + (t.z - b.z) * .5f},
                 {N.x * .5f + (t.x + b.x) * .5f, N.y * .5f + (t.y + b.y) * .5f, N.z * .5f + (t.z + b.z) * .5f},
                 {N.x * .5f + (-t.x + b.x) * .5f, N.y * .5f + (-t.y + b.y) * .5f, N.z * .5f + (-t.z + b.z) * .5f}};
    int idx[6] = {0, 1, 2, 2, 3, 0};
    for (int i : idx) verts.push_back({{c[i].x, c[i].y, c[i].z}, {N.x, N.y, N.z}});
  }
  Buffer vb = make_buffer(a, verts.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "cubes vertex buffer");
  memcpy(vb.map, verts.data(), verts.size() * sizeof(Vertex));

  const int kCubes = 9;
  Buffer cubes = make_buffer(a, kCubes * sizeof(CubeRaw), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "cubes instance buffer");
  for (int i = 0; i < kCubes; i++) {
    CubeRaw r{}; for (int k = 0; k < 16; k++) r.transformation[k] = (k % 5 == 0) ? 0.4f : 0.f;
    r.transformation[15] = 1; r.transformation[12] = (i % 3 - 1) * 0.7f; r.transformation[14] = (i / 3 - 1) * 0.7f;
    ((CubeRaw*)cubes.map)[i] = r;
  }
  Buffer ubo = make_buffer(a, sizeof(Uniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "uniform buffer");

  Image depth = make_image(a, a.extent.width, a.extent.height, VK_FORMAT_D32_SFLOAT,
                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, "cubes depth texture");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "cubes uniform bind group");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo);

  // two vertex buffers (per-vertex, per-instance): build the pipeline here
  Pipeline pl{};
  {
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 1; li.pSetLayouts = &setl;
    VK(vkCreatePipelineLayout(a.dev, &li, nullptr, &pl.layout));
    VkPipelineShaderStageCreateInfo st[2]{};
    st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    st[0].module = sh; st[0].pName = "vs_main";
    st[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    st[1].module = sh; st[1].pName = "fs_main";
    VkVertexInputBindingDescription vbs[2] = {{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX},
                                              {1, sizeof(CubeRaw), VK_VERTEX_INPUT_RATE_INSTANCE}};
    VkVertexInputAttributeDescription at[6] = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)}, {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
        {2, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0}, {3, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 16},
        {4, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 32}, {5, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 48}};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 2; vi.pVertexBindingDescriptions = vbs;
    vi.vertexAttributeDescriptionCount = 6; vi.pVertexAttributeDescriptions = at;
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1; vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.lineWidth = 1; rs.cullMode = VK_CULL_MODE_NONE;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = ds.depthWriteEnable = VK_TRUE; ds.depthCompareOp = VK_COMPARE_OP_LESS;
    VkPipelineColorBlendAttachmentState cb{}; cb.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cbs.attachmentCount = 1; cbs.pAttachments = &cb;
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dy.dynamicStateCount = 2; dy.pDynamicStates = dyn;
    VkFormat df = VK_FORMAT_D32_SFLOAT;
    VkPipelineRenderingCreateInfo rinfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rinfo.colorAttachmentCount = 1; rinfo.pColorAttachmentFormats = &a.swap_format; rinfo.depthAttachmentFormat = df;
    VkGraphicsPipelineCreateInfo gi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gi.pNext = &rinfo; gi.stageCount = 2; gi.pStages = st; gi.pVertexInputState = &vi; gi.pInputAssemblyState = &ia;
    gi.pViewportState = &vp; gi.pRasterizationState = &rs; gi.pMultisampleState = &ms; gi.pDepthStencilState = &ds;
    gi.pColorBlendState = &cbs; gi.pDynamicState = &dy; gi.layout = pl.layout;
    VK(vkCreateGraphicsPipelines(a.dev, VK_NULL_HANDLE, 1, &gi, nullptr, &pl.pipe));
    a.name(pl.pipe, VK_OBJECT_TYPE_PIPELINE, "cubes pipeline");
  }

  Camera camera;
  // Uniforms::new(camera, bounds, light_color)
  Uniforms u{};
  u.camera_proj = camera.build_view_proj_matrix((float)a.extent.width, (float)a.extent.height);
  u.camera_pos[0] = camera.eye.x; u.camera_pos[1] = camera.eye.y; u.camera_pos[2] = camera.eye.z; u.camera_pos[3] = 0;
  u.light_color[0] = 1.0f; u.light_color[1] = 1.0f; u.light_color[2] = 1.0f; u.light_color[3] = 1.0f;  // Color::WHITE.into_linear()
  memcpy(ubo.map, &u, sizeof u);

  run_frames(a, frames_from_env(2), 1, [&](VkCommandBuffer c, uint32_t idx, int) {
    depth_barrier(c, depth.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    begin_rendering(a, c, {a.swap_views[idx]}, depth.view);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pl.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pl.layout, 0, 1, &set, 0, nullptr);
    VkBuffer bufs[2] = {vb.buf, cubes.buf}; VkDeviceSize offs[2] = {0, 0};
    vkCmdBindVertexBuffers(c, 0, 2, bufs, offs);
    vkCmdDraw(c, (uint32_t)verts.size(), kCubes, 0, 0);
    vkCmdEndRendering(c);
  });
  return 0;
}
