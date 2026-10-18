// Distilled from SpaceCorps/open-glass PR #18: the glass composite's
// uniform block OpticalUniforms had nine f32s, then `padding: f32`, then
// `tint_color: vec4<f32>`. The padding slot was meant to bring tint_color
// to byte 40, but with nine scalars ahead of it the slot sits at 36 and the
// shader still inserts 8 implicit bytes, so the shader reads tint_color at
// 48 while the host struct (C layout) puts it at 40; every later field moves
// too. Nothing faults: the values are reinterpreted. The fix pins the layout
// in one host struct, GlassCompositeUniforms (align 16): twelve leading f32s
// filling three 16-byte rows (adding the missing saturation and brightness),
// tint_color at 48, glass_bounds at 64, resolution at 80, explicit _padding
// to 96, and the shader matches it field for field.
#include "../../common/app.hpp"
using namespace bench;

#ifdef FIXED
struct alignas(16) GlassCompositeUniforms {
  float ior, blur_radius, dispersion, rim_power;
  float sheen_intensity, light_angle, roughness, saturation;
  float brightness, thickness, curvature, corner_radius;
  float tint_color[4];        // offset 48
  float glass_bounds[4];      // offset 64
  float resolution[2];        // offset 80
  float _padding[2];          // 96-byte stride
};
static_assert(offsetof(GlassCompositeUniforms, tint_color) == 48, "tint_color");
static_assert(offsetof(GlassCompositeUniforms, glass_bounds) == 64, "glass_bounds");
static_assert(offsetof(GlassCompositeUniforms, resolution) == 80, "resolution");
static_assert(sizeof(GlassCompositeUniforms) == 96, "size");
using CompositeUniforms = GlassCompositeUniforms;
#else
struct OpticalUniforms {
  float ior, blur_radius, dispersion, rim_power, sheen_intensity, light_angle, roughness, thickness, curvature;
  float padding;              // keeps tint_color on a 16-byte boundary
  float tint_color[4];
  float resolution[2];
  float glass_bounds[4];
  float corner_radius;
  float padding2[3];
};
using CompositeUniforms = OpticalUniforms;
#endif

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  Image blurred = make_image(a, 320, 240, VK_FORMAT_R16G16B16A16_SFLOAT,
                             VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, "kawase_blurred");
  Buffer ubo = make_buffer(a, sizeof(CompositeUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "glass_composite_uniforms");
  CompositeUniforms u{};
  u.ior = 1.52f; u.blur_radius = 12.0f; u.dispersion = 0.02f; u.rim_power = 3.0f; u.sheen_intensity = 0.4f;
  u.light_angle = 0.785f; u.roughness = 0.1f; u.thickness = 24.0f; u.curvature = 0.6f;
  u.tint_color[0] = 0.94f; u.tint_color[1] = 0.94f; u.tint_color[2] = 0.96f; u.tint_color[3] = 0.22f;
  u.resolution[0] = 640; u.resolution[1] = 480;
  u.glass_bounds[0] = 80; u.glass_bounds[1] = 60; u.glass_bounds[2] = 480; u.glass_bounds[3] = 360;
  u.corner_radius = 24.0f;
#ifdef FIXED
  u.saturation = 1.0f; u.brightness = 1.0f;
#endif
  memcpy(ubo.map, &u, sizeof u);

  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "composite_bind_group");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo);
  write_image(a, set, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, blurred.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline composite = make_graphics(a, gd, "glass_composite");

  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    if (f == 0)
      image_barrier(c, blurred.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, composite.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, composite.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
