#version 450
// GLSL variant of the clip-space Y rule: gl_Position.y negated after the store
layout(location = 0) in vec3 pos;
void main() {
    gl_Position = vec4(pos, 1.0);
    gl_Position.y = -gl_Position.y;
}
