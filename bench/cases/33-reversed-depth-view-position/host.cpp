// Distilled from mrdoob/three.js #34723 (fix PR #34724): TSL's
// getViewPosition() reconstructs a view-space position from the depth
// texture with the inverse projection. It maps the sampled depth to clip
// space as `depth` on the WebGPU coordinate system and as `depth * 2 - 1`
// otherwise (WebGL's [-1,1]). With reversedDepthBuffer on the WebGL backend
// the projection and the depth buffer are [0,1] (clip control), but the
// WebGL branch still remapped to [-1,1], so reconstructed positions were
// wrong and TRAA rejected its history while zooming. The fix takes the
// [0,1] path when coordinateSystem is WebGPU OR reversedDepthBuffer is on.
#include "../../common/app.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace bench;

enum class CoordinateSystem { WebGL, WebGPU };
struct RendererSettings {
  CoordinateSystem coordinate_system = CoordinateSystem::WebGL;   // WebGL backend
  bool reversed_depth_buffer = true;
};

struct ViewUniforms {
  glm::mat4 view;
  glm::mat4 projection;
  glm::mat4 projection_inverse;
  uint32_t depth_zero_to_one;      // how getViewPosition maps sampled depth to clip z
  float pad[3];
};

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  RendererSettings settings;
  Image depth = make_image(a, 640, 480, VK_FORMAT_D32_SFLOAT,
                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, "depth_texture");
  Buffer ubo = make_buffer(a, sizeof(ViewUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "camera_uniforms");

  ViewUniforms u{};
  u.view = glm::lookAt(glm::vec3(0, 1, 4), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
  // reversed depth: near maps to 1, far to 0, clip depth range [0,1]
  u.projection = glm::perspectiveRH_ZO(glm::radians(50.0f), 640.0f / 480.0f, 100.0f, 0.1f);
  u.projection_inverse = glm::inverse(u.projection);
#ifdef FIXED
  u.depth_zero_to_one = settings.coordinate_system == CoordinateSystem::WebGPU || settings.reversed_depth_buffer;
#else
  u.depth_zero_to_one = settings.coordinate_system == CoordinateSystem::WebGPU;
#endif
  memcpy(ubo.map, &u, sizeof u);

  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "view_set");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo);
  write_image(a, set, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, depth.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  auto sh = load_shader(a, "shaders.spv");
  auto prel = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto preset = alloc_set(a, prel, "prepass_set");
  write_buffer(a, preset, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo);
  GfxDesc pd; pd.vs = pd.fs = sh; pd.vs_entry = "vsScene"; pd.fs_entry = "fsDepth"; pd.sets = {prel};
  pd.depth = VK_FORMAT_D32_SFLOAT; pd.depth_compare = VK_COMPARE_OP_GREATER;
  Pipeline prepass = make_graphics(a, pd, "depth_prepass");
  GfxDesc td; td.vs = td.fs = sh; td.vs_entry = "vsFullscreen"; td.fs_entry = "fsTraa"; td.sets = {setl};
  td.colors = {a.swap_format};
  Pipeline traa = make_graphics(a, td, "traa_resolve");

  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    depth_barrier(c, depth.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
                  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    begin_rendering(a, c, {}, depth.view, 0.0f);         // reversed depth clears to 0 (far)
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, prepass.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, prepass.layout, 0, 1, &preset, 0, nullptr);
    vkCmdDraw(c, 6, 1, 0, 0);
    vkCmdEndRendering(c);
    depth_barrier(c, depth.img, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, traa.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, traa.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
