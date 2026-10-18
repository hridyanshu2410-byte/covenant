#include <covenant.h>
#include <glm/glm.hpp>
#include <vector>
struct COV("block=Light") Light { glm::vec3 position; glm::vec3 color; float radius; };
struct SceneLight { float x COV("unit=cm"); float y COV("unit=cm"); float z COV("unit=cm"); };
glm::vec3 to_vec(const SceneLight& s) { return {s.x, s.y, s.z}; }
void upload(std::vector<Light>& out, const std::vector<SceneLight>& in) {
  for (auto& s : in) { Light l; glm::vec3 p = to_vec(s); l.position = p * 0.01f; l.color = {1,1,1}; l.radius = 5; out.push_back(l); }
}
