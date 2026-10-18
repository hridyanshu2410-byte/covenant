// Distilled from playcanvas/engine PR #8307 ("[FIX] Fix glTF spot light
// luminance calculation", merged 2025-12-22). The KHR_lights_punctual
// importer converts the glTF spot cone angles (radians) to PlayCanvas'
// degrees, and then computed the light's luminance as
//   luminance = intensity * Light.getLightUnitConversion(type, lightProps.outerConeAngle, lightProps.innerConeAngle)
// passing the already-converted degree values to a function that calls
// Math.cos() on radians. The Light later turns luminance back into an
// intensity with the same conversion fed with outerConeAngle * DEG_TO_RAD,
// so the round trip does not cancel and the light's intensity is wrong. The
// fix passes the original glTF radian values (gltfLight.spot.outerConeAngle,
// default PI/4; innerConeAngle, default 0). Host-only error; the shader and
// its uniforms are unchanged.
#include "../../common/app.hpp"
#include <cmath>
using namespace bench;

static const float kPi = 3.14159265358979f, DEG_TO_RAD = kPi / 180.0f, RAD_TO_DEG = 180.0f / kPi;

// Light.getLightUnitConversion(LIGHTTYPE_SPOT, outerAngle, innerAngle): angles in radians
static float get_light_unit_conversion_spot(float outerAngle, float innerAngle) {
  const float falloffEnd = std::cos(outerAngle);
  const float falloffStart = std::cos(innerAngle);
  return 2 * kPi * ((1 - falloffStart) + (falloffStart - falloffEnd) / 2.0f);
}

struct GltfSpot { float intensity; float innerConeAngle, outerConeAngle; float color[3]; };
struct LightProps { float innerConeAngle, outerConeAngle; float luminance; float color[3]; };   // degrees

// glb/extensions/khr-lights-punctual.js createLight()
static LightProps create_light(const GltfSpot& g) {
  LightProps p{};
  p.innerConeAngle = g.innerConeAngle * RAD_TO_DEG;
  p.outerConeAngle = g.outerConeAngle * RAD_TO_DEG;
#ifdef FIXED
  const float outerAngleRad = g.outerConeAngle, innerAngleRad = g.innerConeAngle;
  p.luminance = g.intensity * get_light_unit_conversion_spot(outerAngleRad, innerAngleRad);
#else
  p.luminance = g.intensity * get_light_unit_conversion_spot(p.outerConeAngle, p.innerConeAngle);
#endif
  for (int i = 0; i < 3; i++) p.color[i] = g.color[i];
  return p;
}

struct SpotLightUniform { float position[4], direction[4], colorLinear[4]; float innerConeCos, outerConeCos, pad[2]; };

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  // the .glb's spot light: 100 cd, cones 0.3 / 0.6 rad
  GltfSpot g{100.0f, 0.3f, 0.6f, {1.0f, 0.9f, 0.8f}};
  LightProps props = create_light(g);
  // Light: _luminance -> intensity (scene/light.js, physical units)
  float intensity = props.luminance / get_light_unit_conversion_spot(props.outerConeAngle * DEG_TO_RAD,
                                                                     props.innerConeAngle * DEG_TO_RAD);
  printf("luminance %.2f -> intensity %.2f (glTF %.2f cd)\n", props.luminance, intensity, g.intensity);
  SpotLightUniform u{{0, 3, 0, 1}, {0, -1, 0, 0},
                     {props.color[0] * intensity, props.color[1] * intensity, props.color[2] * intensity, 1},
                     std::cos(props.innerConeAngle * DEG_TO_RAD), std::cos(props.outerConeAngle * DEG_TO_RAD), {0, 0}};
  Buffer lub = make_buffer(a, sizeof u, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "spot_light_uniforms");
  memcpy(lub.map, &u, sizeof u);
  Buffer cub = make_buffer(a, 64, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "camera_uniforms");
  float vp[16] = {0.4f, 0, 0, 0,  0, 0, 0.2f, 0,  0, 0.4f, 0, 0,  0, 0, 0.5f, 1};    // top-down view of the floor
  memcpy(cub.map, vp, sizeof vp);
  float floor[18] = {-2, 0, -2, 2, 0, -2, 2, 0, 2, -2, 0, -2, 2, 0, 2, -2, 0, 2};
  Buffer vb = make_buffer(a, sizeof floor, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "floor_mesh");
  memcpy(vb.map, floor, sizeof floor);
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "forward_lit_bindings");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, cub);
  write_buffer(a, set, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, lub);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format}; gd.vertex_stride = 12;
  gd.attributes = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}};
  Pipeline lit = make_graphics(a, gd, "forward_lit");

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, lit.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, lit.layout, 0, 1, &set, 0, nullptr);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &off);
    vkCmdDraw(c, 6, 1, 0, 0);
    vkCmdEndRendering(c);
  });
  return 0;
}
