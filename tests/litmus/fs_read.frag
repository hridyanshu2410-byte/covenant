#version 450
layout(set = 0, binding = 0) readonly buffer B { uint v[]; } b;
layout(location = 0) out vec4 c;
void main() {
    c = vec4(float(b.v[0]));           // a fragment at any position B reads storageBuffer[0]
}
