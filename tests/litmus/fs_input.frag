#version 450
layout(input_attachment_index = 0, set = 0, binding = 1) uniform subpassInput a;
layout(location = 0) out vec4 c;
void main() {
    c = subpassLoad(a);                // its own pixel of attachment A
}
