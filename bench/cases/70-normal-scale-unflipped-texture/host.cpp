// Distilled from MrityunjayBhardwaj/TheBasher #1325 (open; fix outlined in
// the issue): a native glTF import builds its MeshStandardMaterial itself. The
// geometry has no tangent attribute, so three.js derives the tangent frame
// from UV derivatives (perturbNormal2Arb), and the normal texture is loaded
// as glTF requires, flipY = false. For an unflipped texture the derived
// bitangent points the opposite way to glTF's +Y (green) axis, which is why
// three.js' GLTFLoader sets normalScale.y = -1. The native import never sets
// normalScale (default (1, 1)): the surface leans the wrong way (lit from
// below instead of above). The fix computes normalScale from the texture's
// flipY in one shared function used by both material builders.
// (three.js / WebGL ported to Vulkan: the texture is uploaded row 0 first,
// unflipped, with glTF UVs.)
#include "../../common/app.hpp"
using namespace bench;

struct Texture { Image image; bool flipY; };

// materialBuilders.normalScaleFor(texture)
static void normal_scale_for(const Texture& t, float out[2]) {
#ifdef FIXED
  out[0] = 1.0f; out[1] = t.flipY ? 1.0f : -1.0f;
#else
  (void)t; out[0] = 1.0f; out[1] = 1.0f;          // MeshStandardMaterial default
#endif
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  // normal texture from the .glb: 4x4 RGBA8 bump, uploaded unflipped (glTF: flipY = false)
  Texture normalTex{make_image(a, 4, 4, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                               "normalTexture"), false};
  Buffer pixels = make_buffer(a, 4 * 4 * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "normalTexture_png");
  uint8_t* px = (uint8_t*)pixels.map;
  for (int y = 0; y < 4; y++)
    for (int x = 0; x < 4; x++) {     // tangent-space normal tilted towards +Y (green up, glTF convention)
      uint8_t* p = px + (y * 4 + x) * 4; p[0] = 128; p[1] = 200; p[2] = 230; p[3] = 255;
    }
  VkCommandBuffer up = alloc_cmd(a, a.gfx_pool);
  begin(up);
  image_barrier(up, normalTex.image.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
  VkBufferImageCopy region{}; region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; region.imageExtent = {4, 4, 1};
  vkCmdCopyBufferToImage(up, pixels.buf, normalTex.image.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  image_barrier(up, normalTex.image.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
  VK(vkEndCommandBuffer(up));
  submit(a.gfx, up, {}, {}, VK_NULL_HANDLE);
  VK(vkQueueWaitIdle(a.gfx));

  // native import: buildStandardMaterial(gltfMaterial)
  float normalScale[2]; normal_scale_for(normalTex, normalScale);
  Buffer ubo = make_buffer(a, 144, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "material_uniforms");
  float u[36] = {1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, -3, 1,          // modelViewMatrix
                 1.8f, 0, 0, 0,  0, -2.4f, 0, 0,  0, 0, -1.0f, -1,  0, 0, -0.1f, 0,   // projectionMatrix (Vulkan clip)
                 normalScale[0], normalScale[1], 0, 0};
  memcpy(ubo.map, u, sizeof u);
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "standard_material");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo);
  write_image(a, set, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, normalTex.image.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  // a quad facing the camera, glTF UVs (v = 0 at the top), no TANGENT attribute
  struct V { float p[3], n[3], uv[2]; };
  V quad[6] = {{{-1, 1, 0}, {0, 0, 1}, {0, 0}}, {{1, 1, 0}, {0, 0, 1}, {1, 0}}, {{-1, -1, 0}, {0, 0, 1}, {0, 1}},
               {{1, 1, 0}, {0, 0, 1}, {1, 0}}, {{1, -1, 0}, {0, 0, 1}, {1, 1}}, {{-1, -1, 0}, {0, 0, 1}, {0, 1}}};
  Buffer vb = make_buffer(a, sizeof quad, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "imported_mesh");
  memcpy(vb.map, quad, sizeof quad);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format}; gd.vertex_stride = sizeof(V);
  gd.attributes = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12}, {2, 0, VK_FORMAT_R32G32_SFLOAT, 24}};
  Pipeline mesh = make_graphics(a, gd, "mesh_standard");

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh.layout, 0, 1, &set, 0, nullptr);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &off);
    vkCmdDraw(c, 6, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
