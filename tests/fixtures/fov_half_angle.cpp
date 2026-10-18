// Perspective field of view: an in-tree builder (tan of half a parameter, in
// a function that fills a 4x4 matrix) is handed a degrees literal from a
// default member initializer (bug); the fix converts it with pi / 180. A
// library call (cglm glm_perspective, radians) is checked from the table.
#include <cmath>
struct Mat4 { float m[16]; };
static Mat4 make_perspective(float fovy, float aspect, float zn, float zf) {
  Mat4 r{};
  float f = 1.0f / std::tan(fovy * 0.5f);
  r.m[0] = f / aspect; r.m[5] = f; r.m[10] = zf / (zn - zf); r.m[11] = -1.0f;
  r.m[14] = zn * zf / (zn - zf);
  return r;
}
void glm_perspective(float fovy, float aspect, float nearZ, float farZ, float dest[4][4]);
struct Camera { float fov_y = 60.0f; float aspect = 1.5f; };
Mat4 build(const Camera& c, float out[4][4]) {
#ifdef FIXED
  glm_perspective(c.fov_y * 3.14159265f / 180.0f, c.aspect, 0.1f, 100.0f, out);
  return make_perspective(c.fov_y * 3.14159265f / 180.0f, c.aspect, 0.1f, 100.0f);
#else
  glm_perspective(c.fov_y, c.aspect, 0.1f, 100.0f, out);
  return make_perspective(c.fov_y, c.aspect, 0.1f, 100.0f);
#endif
}
int main() { Camera c; float o[4][4]; build(c, o); return 0; }
