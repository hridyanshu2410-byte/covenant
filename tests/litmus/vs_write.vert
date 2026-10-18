#version 450
layout(set = 0, binding = 0) buffer B { uint v[]; } b;
void main() {
    b.v[0] = uint(gl_VertexIndex);
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
