#version 450
struct Light { vec3 position; vec3 color; };
layout(set=0,binding=1) readonly buffer Lights { Light lights[]; };
layout(location=0) in vec3 vNormal;
layout(location=1) in vec3 vWorldPos;
layout(location=0) out vec4 outColor;
void main(){ vec3 L = normalize(lights[0].position - vWorldPos); outColor = vec4(max(dot(normalize(vNormal),L),0)*lights[0].color,1); }
