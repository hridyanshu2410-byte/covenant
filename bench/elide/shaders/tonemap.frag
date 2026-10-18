#version 450
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(set = 0, binding = 0) uniform sampler2D hdr;
void main() {
  vec3 c = texture(hdr, uv).rgb;
  color = vec4(c / (1.0 + c), 1.0);
}
