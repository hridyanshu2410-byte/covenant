// Distilled from agutierreza/BRITE-Engine #15 (closed): glTF defines
// baseColorFactor as linear. raylib's LoadGLTF (rmodels.c) stores it in the
// material's albedo Color as base_color_factor[i]*255, i.e. as if it were an
// sRGB colour like every other Color in the engine, and pbr.fs:155 then
// linearizes it again: albedo = pow(colDiffuse.rgb*fragColor.rgb, vec3(2.2)).
// Every glTF material with a colour factor draws far too dark (a foliage
// green (49, 96, 55) reaches the shader as (6, 29, 8)); ReadModelMeshes
// propagates the same value into vertex colours. The fix converts the loaded
// factor from linear to sRGB once at load time (0.5 -> 186), so that it
// matches code-built meshes and draw tints; the shader is unchanged.
#include "../../common/app.hpp"
#include <cmath>
using namespace bench;

struct Color { unsigned char r, g, b, a; };
struct cgltf_pbr_metallic_roughness { float base_color_factor[4]; };

// LoadGLTF(): material colour from the glTF file
static Color load_base_color(const cgltf_pbr_metallic_roughness& pbr) {
  Color c;
#ifdef FIXED
  auto enc = [](float x) { return (unsigned char)std::lround(std::pow(std::fmax(0.0f, std::fmin(1.0f, x)), 1.0f / 2.2f) * 255.0f); };
  c.r = enc(pbr.base_color_factor[0]); c.g = enc(pbr.base_color_factor[1]); c.b = enc(pbr.base_color_factor[2]);
#else
  c.r = (unsigned char)(pbr.base_color_factor[0] * 255);
  c.g = (unsigned char)(pbr.base_color_factor[1] * 255);
  c.b = (unsigned char)(pbr.base_color_factor[2] * 255);
#endif
  c.a = (unsigned char)(pbr.base_color_factor[3] * 255);
  return c;
}

int main(int argc, char** argv) {
  AppDesc d; d.swap_format = VK_FORMAT_B8G8R8A8_UNORM;      // raylib default framebuffer
  App a = make_app(argc, argv, d);
  auto sh = load_shader(a, "shaders.spv");
  cgltf_pbr_metallic_roughness leaf{{0.0307f, 0.1170f, 0.0382f, 1.0f}};   // "foliage" material, linear
  Color albedo = load_base_color(leaf);                       // material.maps[MATERIAL_MAP_ALBEDO].color

  Buffer ubo = make_buffer(a, 80, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "pbr_uniforms");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "pbr_material");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo);
  // mesh: position, normal, colour (white: the model has no COLOR_0)
  struct V { float p[3], n[3]; uint8_t c[4]; };
  V verts[3] = {{{-0.6f, 0.5f, 0.5f}, {0, 0, -1}, {255, 255, 255, 255}}, {{0.6f, 0.5f, 0.5f}, {0, 0, -1}, {255, 255, 255, 255}},
                {{0, -0.6f, 0.5f}, {0, 0, -1}, {255, 255, 255, 255}}};
  Buffer vb = make_buffer(a, sizeof verts, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "leaf_mesh");
  memcpy(vb.map, verts, sizeof verts);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format}; gd.vertex_stride = sizeof(V);
  gd.attributes = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12}, {2, 0, VK_FORMAT_R8G8B8A8_UNORM, 24}};
  Pipeline pbr = make_graphics(a, gd, "pbr_shader");

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pbr.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pbr.layout, 0, 1, &set, 0, nullptr);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &off);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  }, [&](int f) {
    // DrawMesh(): SetShaderValue(mvp), colDiffuse = material colour / 255
    float u[20] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                   albedo.r / 255.0f, albedo.g / 255.0f, albedo.b / 255.0f, albedo.a / 255.0f};
    memcpy(ubo.map, u, sizeof u);
  });
}
