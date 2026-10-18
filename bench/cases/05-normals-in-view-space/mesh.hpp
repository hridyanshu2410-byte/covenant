// Host-side vertex data, linked to the shader's vertex input struct.
#pragma once
#include <covenant.h>
#include <vector>
struct float3 { float x, y, z; };
struct COV("block=VIn") Vertex { float3 position; float3 normal; };

// Positions and normals as the modelling tool exports them (object space),
// and the same normals after the camera transform (view space).
float3 object_normal(int i) COV("frame=object");
float3 view_normal(int i) COV("frame=view");
float3 object_position(int i) COV("frame=object");
