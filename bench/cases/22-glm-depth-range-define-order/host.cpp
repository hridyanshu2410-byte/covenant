// Distilled from SharkFinPro/VulkanRenderer PR #204: GLM builds projections
// for OpenGL's [-1,1] clip depth unless GLM_FORCE_DEPTH_ZERO_TO_ONE is set
// before GLM is first included. The renderer defined it in the light sources
// after including GLM, so it had no effect: camera and light projections kept
// [-1,1], the near plane clipped at about twice its distance and half the
// depth range was wasted. The fix defines it for the whole library (a PUBLIC
// compile definition, i.e. before every include), static_asserts the
// configuration, and halves the spot light shadow bias in the shader.
#ifdef FIXED
#define GLM_FORCE_DEPTH_ZERO_TO_ONE        // project-wide definition, before any GLM include
#endif
#include "../../common/app.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#ifndef FIXED
#define GLM_FORCE_DEPTH_ZERO_TO_ONE        // BUG: too late; GLM's setup header is already included
#endif
#ifdef FIXED
static_assert(GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_ZO_BIT, "GLM must build [0,1] depth projections for Vulkan");
#endif
using namespace bench;

struct Camera { glm::mat4 view; glm::mat4 proj; glm::mat4 lightViewProj; };

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  Image depth = make_image(a, 640, 480, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, "scene_depth");
  Image shadow = make_image(a, 512, 512, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            "spot_shadow_map");
  Buffer ubo = make_buffer(a, sizeof(Camera), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "camera");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "scene_set");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo);
  write_image(a, set, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, shadow.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  gd.depth = VK_FORMAT_D32_SFLOAT; gd.depth_compare = VK_COMPARE_OP_LESS;
  Pipeline pipe = make_graphics(a, gd, "objects");

  Camera cam;
  cam.view = glm::lookAt(glm::vec3(0, 0, 2), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
  cam.proj = glm::perspective(glm::radians(60.0f), 640.0f / 480.0f, 0.1f, 100.0f);
  cam.proj[1][1] *= -1;
  glm::mat4 lightView = glm::lookAt(glm::vec3(2, 4, 2), glm::vec3(0, 0, -3), glm::vec3(0, 1, 0));
  cam.lightViewProj = glm::perspective(glm::radians(45.0f), 1.0f, 0.5f, 50.0f) * lightView;
  memcpy(ubo.map, &cam, sizeof cam);

  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    if (f == 0)
      image_barrier(c, shadow.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    depth_barrier(c, depth.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    begin_rendering(a, c, {a.swap_views[idx]}, depth.view, 1.0f);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
