#version 450
layout(set = 0, binding = 0) readonly buffer B { uint v[]; } b;
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) + float(b.v[0]);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
