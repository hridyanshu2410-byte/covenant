// Distilled from matiaszanolli/ByroRedux issue #1516 / PR #1517 ("correct
// inverted bitangent sign on inline tangent paths", merged 2026-06-14).
// When BSTriShape vertices with inline VF_TANGENTS are unpacked (and in SSE
// skin reconstruction), the CPU computes the bitangent handedness stored in
// tangent.w as sign(dot(dP/dU, cross(N, dP/dV))) - the operands of the
// scalar triple product swapped relative to the canonical
// sign(dot(dP/dV, cross(N, dP/dU))). That negates the result: -1 instead of
// +1 (and vice versa) for every UV layout, so the shader's
// B = tangent.w * cross(N, T) points against +V and the normal map's green
// channel is applied inverted. The fix routes all five sign sites through
// one helper, types::bitangent_sign. Here one decode site is reproduced; the
// mesh is a quad with D3D-style UVs (v down), the normal map texels lean
// towards +V (green > 0.5).
#include "../../common/app.hpp"
#include <cmath>
using namespace bench;

struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static V3 mul(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// types::bitangent_sign (fixed) / the inline-tangent decode in bs_tri_shape.rs (bug)
static float bitangent_sign(V3 n, V3 dpdu, V3 dpdv) {
#ifdef FIXED
  return dot(dpdv, cross(n, dpdu)) < 0.0f ? -1.0f : 1.0f;
#else
  return dot(dpdu, cross(n, dpdv)) < 0.0f ? -1.0f : 1.0f;
#endif
}

struct Vertex { float pos[3], normal[3], tangent[4], uv[2]; };

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  // BSTriShape: a wall quad facing +Z (world), UV origin top-left (v grows downwards, -Y)
  V3 p[4] = {{-1, 1, 0}, {1, 1, 0}, {1, -1, 0}, {-1, -1, 0}};
  float uv[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  V3 N = {0, 0, 1};
  V3 T = {1, 0, 0};                                      // inline VF_TANGENTS tangent (+U direction)
  // per-triangle position derivatives w.r.t. UV (triangle 0, 1, 2)
  V3 e1 = sub(p[1], p[0]), e2 = sub(p[2], p[0]);
  float du1 = uv[1][0] - uv[0][0], dv1 = uv[1][1] - uv[0][1], du2 = uv[2][0] - uv[0][0], dv2 = uv[2][1] - uv[0][1];
  float r = 1.0f / (du1 * dv2 - du2 * dv1);
  V3 dpdu = mul(sub(mul(e1, dv2), mul(e2, dv1)), r);
  V3 dpdv = mul(sub(mul(e2, du1), mul(e1, du2)), r);
  float w = bitangent_sign(N, dpdu, dpdv);
  printf("tangent.w = %+.0f\n", w);
  Vertex verts[6];
  int tri[6] = {0, 1, 2, 0, 2, 3};
  for (int i = 0; i < 6; i++) {
    int k = tri[i];
    verts[i] = {{p[k].x, p[k].y, p[k].z}, {N.x, N.y, N.z}, {T.x, T.y, T.z, w}, {uv[k][0], uv[k][1]}};
  }
  Buffer vb = make_buffer(a, sizeof verts, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "bstrishape_vertices");
  memcpy(vb.map, verts, sizeof verts);

  // normal map: DDS, tangent-space normal leaning towards +V
  Image nm = make_image(a, 4, 4, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                        "wall_n_dds");
  Buffer px = make_buffer(a, 64, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "wall_n_pixels");
  for (int i = 0; i < 16; i++) { uint8_t* q = (uint8_t*)px.map + i * 4; q[0] = 128; q[1] = 210; q[2] = 220; q[3] = 255; }
  VkCommandBuffer up = alloc_cmd(a, a.gfx_pool);
  begin(up);
  image_barrier(up, nm.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
  VkBufferImageCopy reg{}; reg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; reg.imageExtent = {4, 4, 1};
  vkCmdCopyBufferToImage(up, px.buf, nm.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);
  image_barrier(up, nm.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  VK(vkEndCommandBuffer(up));
  submit(a.gfx, up, {}, {}, VK_NULL_HANDLE);
  VK(vkQueueWaitIdle(a.gfx));

  Buffer camera = make_buffer(a, 64, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "camera");
  float vp[16] = {0.75f, 0, 0, 0,  0, -1.0f, 0, 0,  0, 0, 0.5f, 0,  0, 0, 0.5f, 1};   // column-major, world +Y up -> Vulkan clip -Y
  memcpy(camera.map, vp, sizeof vp);
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "mesh_material");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, camera);
  write_image(a, set, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, nm.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format}; gd.vertex_stride = sizeof(Vertex);
  gd.attributes = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12},
                   {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 24}, {3, 0, VK_FORMAT_R32G32_SFLOAT, 40}};
  Pipeline mesh = make_graphics(a, gd, "mesh_pipeline");

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh.layout, 0, 1, &set, 0, nullptr);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &off);
    vkCmdDraw(c, 6, 1, 0, 0);
    vkCmdEndRendering(c);
  });
  return 0;
}
