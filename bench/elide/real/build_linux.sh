#!/bin/sh
# Build the measurement helpers for real samples (Linux): VK_LAYER_COVENANT_probe and
# the fake-clock LD_PRELOAD shim. Usage: sh build_linux.sh [outdir]  (default build/elide-real/probe)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-build/elide-real/probe}
CXX=${CXX:-$(command -v clang++-20 || command -v clang++ || command -v g++)}
mkdir -p "$OUT"
$CXX -std=c++17 -O2 -g -fPIC -shared -fvisibility=hidden -Wl,-Bsymbolic -Wall -o "$OUT/libVkLayer_COVENANT_probe.so" \
  "$HERE/probe_layer/covenant_probe_layer.cpp" -ldl
sed "s#@LIB@#./libVkLayer_COVENANT_probe.so#" "$HERE/probe_layer/VkLayer_COVENANT_probe.json.in" > "$OUT/VkLayer_COVENANT_probe.json"
$CXX -std=c++17 -O2 -fPIC -shared -o "$OUT/libcovenant_fakeclock.so" "$HERE/fakeclock.cpp"
echo "built: $OUT/libVkLayer_COVENANT_probe.so, VkLayer_COVENANT_probe.json, libcovenant_fakeclock.so"
