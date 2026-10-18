#ifdef ZO_FIRST
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#ifndef ZO_FIRST
#define GLM_FORCE_DEPTH_ZERO_TO_ONE   // too late
#endif
struct Settings { float gamma = 2.2f; float pad[2]; };
struct Camera { glm::mat4 proj; glm::mat4 viewProj; float gamma; };
void fill(Camera& c, const Settings& s, const glm::mat4& view) {
  c.proj = glm::perspective(1.0f, 1.5f, 0.1f, 100.0f);
  c.viewProj = glm::perspective(1.0f, 1.5f, 0.1f, 100.0f) * view;
  c.gamma = s.gamma;
}
