#version 450
// one G-buffer channel: a procedural pattern; `iters` scales the cost
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(push_constant) uniform PC { uint pass; uint iters; uint frame; } pc;
void main() {
  vec2 z = uv * (2.0 + float(pc.pass)) + vec2(0.01 * float(pc.frame));
  float acc = 0.0;
  for (uint k = 0u; k < pc.iters; k++) {
    z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) * 0.5 + uv;
    acc += exp(-dot(z, z));
  }
  float a = acc / float(max(pc.iters, 1u));
  if (pc.pass == 0u) color = vec4(a, uv, 1.0);
  else if (pc.pass == 1u) color = vec4(normalize(vec3(z, 1.0)) * 0.5 + 0.5, 1.0);
  else color = vec4(fract(a * 7.0), fract(a * 3.0), a, 1.0);
}
