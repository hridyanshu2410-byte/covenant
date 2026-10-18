#version 450
layout(set = 0, binding = 0) uniform U { vec4 t; } u;
layout(set = 0, binding = 1) readonly buffer In { vec4 p[]; } pin;
void main() {
  vec4 c = pin.p[gl_VertexIndex / 3];
  vec2 off = vec2(float(gl_VertexIndex % 3 == 1), float(gl_VertexIndex % 3 == 2)) * 0.05;
  gl_Position = vec4(c.xy + off + vec2(u.t.y * 0.0), 0.0, 1.0);
}
