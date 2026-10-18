#!/bin/bash
# Build the two Vulkan examples of one Dear ImGui tree with their own Makefiles (needs GLFW and SDL2
# development packages). usage: build_imgui.sh <tree>
set -eu
for e in glfw sdl2; do make -C "$1/examples/example_${e}_vulkan" -j2 > "$1/examples/example_${e}_vulkan/build.log" 2>&1; done
echo "built $1"
