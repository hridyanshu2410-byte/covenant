// A field converted with glm::radians in one place is in degrees, so passing
// it (scaled by 0.5, still degrees) to std::tan elsewhere is a unit bug.
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct Camera { float fov = 45.0f; float aspect = 1.0f; };

glm::mat4 projection(const Camera& c) { return glm::perspective(glm::radians(c.fov), c.aspect, 0.1f, 100.0f); }

float half_height(const Camera& c) {
#ifdef FIXED
  return std::tan(glm::radians(c.fov) * 0.5f);
#else
  return std::tan(c.fov * 0.5f);
#endif
}

float deg_to_rad_by_hand(const Camera& c) { return std::tan(c.fov * 0.0174533f); }   // a conversion
