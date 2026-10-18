#version 450
layout(set = 0, binding = 0) buffer B { uint v[]; } b;
layout(location = 0) out vec4 c;
void main() {
    b.v[0] = uint(gl_FragCoord.x);     // a fragment at position A writes storageBuffer[0]
    c = vec4(1.0);
}
