#version 450
// "lighting": combine the three G-buffer channels into an HDR value
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 0) uniform sampler2D albedo;
layout(set = 0, binding = 1) uniform sampler2D normal;
layout(set = 0, binding = 2) uniform sampler2D material;
void main() {
  vec3 a = texture(albedo, uv).rgb;
  vec3 n = texture(normal, uv).rgb * 2.0 - 1.0;
  vec3 m = texture(material, uv).rgb;
  float ndl = max(dot(normalize(n + vec3(1e-3)), normalize(vec3(0.3, 0.5, 0.8))), 0.0);
  color = vec4(a * (0.1 + 4.0 * ndl) * (0.5 + m), 1.0);
}
