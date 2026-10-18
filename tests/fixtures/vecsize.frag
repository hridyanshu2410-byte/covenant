#version 450
layout(set = 0, binding = 0, std430) buffer FragmentBuffer { uvec3 data[]; };
layout(location = 0) out vec4 o;
void main() { data[uint(gl_FragCoord.x)] = uvec3(1, 2, 3); o = vec4(1); }
