#!/bin/sh
# Build the three workloads (Linux). Usage: sh build.sh [outdir]   (default build/elide/bin)
# Needs a C++17 compiler and the Vulkan headers + loader (libvulkan-dev).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-build/elide/bin}
CXX=${CXX:-$(command -v clang++-20 || command -v clang++ || command -v g++)}
mkdir -p "$OUT"
for w in particles deferred postchain; do
  $CXX -std=c++17 -O2 -g -Wall -o "$OUT/$w" "$HERE/$w.cpp" -lvulkan
done
echo "built: $OUT/particles $OUT/deferred $OUT/postchain"
