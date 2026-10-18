#!/bin/bash
# Build the whole-frame workload program `wf` and its shaders.
#
#   VKINC=<Vulkan-Headers include dir> WORK=<work dir> bash build.sh
#
# Writes $WORK/build/wf and $WORK/build/shaders/*.spv. Needs a C++17 compiler (clang++-20 when
# present, as for the paper's runs, otherwise g++; override with CXX), the libxcb headers and
# glslangValidator. wf links against libvulkan.so.1: the system Vulkan loader when one is
# installed, otherwise the model driver's copy ($MD, default model_driver/build of this
# repository). Either way it runs on the model driver through LD_LIBRARY_PATH=$MD.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
: "${WORK:?set WORK to a work directory}"
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
: "${VKINC:?set VKINC to the Vulkan-Headers include directory}"
MD=${MD:-$REPO/model_driver/build}
if [ -z "${CXX:-}" ]; then CXX=$(command -v clang++-20 || command -v g++); fi
B=$WORK/build
mkdir -p "$B/shaders"
LIBDIR=()
if ! (ldconfig -p 2>/dev/null || true) | grep 'libvulkan.so.1 ' > /dev/null; then
  [ -f "$MD/libvulkan.so.1" ] || { echo "no system libvulkan.so.1 and no $MD/libvulkan.so.1 (build the model driver first)" >&2; exit 2; }
  LIBDIR=(-L"$MD")
fi
"$CXX" -std=c++17 -O1 -I"$VKINC" -o "$B/wf" "$HERE/wf.cpp" "${LIBDIR[@]}" -l:libvulkan.so.1 -lxcb
for s in "$HERE"/shaders/*; do
  glslangValidator -V -o "$B/shaders/$(basename "$s").spv" "$s" > /dev/null
done
echo "built $B/wf and $(ls "$B/shaders" | wc -l) shaders in $B/shaders"
