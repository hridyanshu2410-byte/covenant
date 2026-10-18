#!/bin/sh
# Build VK_LAYER_COVENANT_elide (Linux). Usage: sh build.sh [outdir]   (default /tmp/elide/layer)
# Then: export VK_LAYER_PATH=<outdir> (or VK_ADD_LAYER_PATH) and enable the layer
# with VK_INSTANCE_LAYERS=VK_LAYER_COVENANT_elide, plus COVENANT_ELIDE_PLAN=<plan.json>.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-/tmp/elide/layer}
CXX=${CXX:-$(command -v clang++-20 || command -v clang++ || command -v g++)}
mkdir -p "$OUT"
$CXX -std=c++17 -O2 -g -fPIC -shared -fvisibility=hidden -Wl,-Bsymbolic -Wall -o "$OUT/libVkLayer_COVENANT_elide.so" \
  "$HERE/covenant_elide_layer.cpp"
sed "s#@LIB@#./libVkLayer_COVENANT_elide.so#" "$HERE/VkLayer_COVENANT_elide.json.in" > "$OUT/VkLayer_COVENANT_elide.json"
echo "built: $OUT/libVkLayer_COVENANT_elide.so and $OUT/VkLayer_COVENANT_elide.json"
