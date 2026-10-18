#!/bin/bash
# Build SDL3 (e4af6f8e) with its test programs; same flags for the original and the patched tree.
# CC / CXX default to clang-20 / clang++-20, the compilers of the paper's runs.
# usage: build_sdl.sh <source tree>
set -eu
S=$1; cd "$S"
CC=${CC:-clang-20} CXX=${CXX:-clang++-20} cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSDL_TESTS=ON \
  -DSDL_INSTALL_TESTS=OFF -DSDL_WAYLAND=OFF -DSDL_X11_XTEST=OFF > build.log 2>&1
ninja -C build SDL3-shared testsprite >> build.log 2>&1
echo "built $S -> $S/build/test/testsprite"
