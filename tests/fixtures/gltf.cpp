// glTF material factors are linear; the host stores one as code values.
#include <cmath>
struct cgltf_pbr_metallic_roughness { float base_color_factor[4]; };
struct Color { unsigned char r, g, b, a; };
static Color load(const cgltf_pbr_metallic_roughness& p) {
  Color c;
#ifdef FIXED
  c.r = (unsigned char)std::lround(std::pow(p.base_color_factor[0], 1.0f / 2.2f) * 255.0f);
  c.g = (unsigned char)std::lround(std::pow(p.base_color_factor[1], 1.0f / 2.2f) * 255.0f);
  c.b = (unsigned char)std::lround(std::pow(p.base_color_factor[2], 1.0f / 2.2f) * 255.0f);
#else
  c.r = (unsigned char)(p.base_color_factor[0] * 255);
  c.g = (unsigned char)(p.base_color_factor[1] * 255);
  c.b = (unsigned char)(p.base_color_factor[2] * 255);
#endif
  c.a = 255;
  return c;
}
void upload(float* dst) {
  cgltf_pbr_metallic_roughness leaf{{0.0307f, 0.1170f, 0.0382f, 1.0f}};
  Color c = load(leaf);
  float u[4] = {c.r / 255.0f, c.g / 255.0f, c.b / 255.0f, c.a / 255.0f};
  for (int i = 0; i < 4; i++) dst[i] = u[i];
}
