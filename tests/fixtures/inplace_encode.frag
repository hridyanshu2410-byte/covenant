#version 450
// An in-place encode in branchy code (seen in real engines). With flow-
// insensitive locals 'color' was both linear and sRGB.
layout(binding = 0) uniform sampler2D hdr;
layout(binding = 1) uniform sampler2D unusedMap;      // declared, never used by main
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;
void main() {
    vec3 color = texture(hdr, uv).rgb;
    if (uv.x > 0.5) color *= 0.8;
    color = pow(color, vec3(0.4545));
    outColor = vec4(color, 1.0);
}
