#version 450
layout(set=0,binding=0) uniform Camera { mat4 model; mat4 view; mat4 proj; } cam;
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=0) out vec3 vWorldPos;
layout(location=1) out vec3 vNormal;
void main(){ vec4 wp = cam.model*vec4(position,1); vWorldPos=wp.xyz; vNormal=(cam.model*vec4(normal,0)).xyz; gl_Position = cam.proj*cam.view*wp; }
