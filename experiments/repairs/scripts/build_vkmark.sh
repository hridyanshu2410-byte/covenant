#!/bin/bash
# Build one vkmark variant tree $WORK/src/vkm-<variant> (a worktree of the clone at ab6e6f3, which is
# the source of the Ubuntu package 2017.08+git20220909). Same flags for every variant.
# usage: build_vkmark.sh <variant>      (needs meson, ninja, glslc, libxcb-icccm4-dev)
set -eu
. "$(dirname "$(readlink -f "$0")")/common.sh"
b=$1; W=$T/vkm-$b
[ -d "$W" ] || git -C "$T/vkmark" worktree add -q "$W" "$b"
cd "$W" && rm -rf build
meson setup build -Dwayland=false -Dkms=false > build.log 2>&1
ninja -C build src/vkmark src/xcb.so >> build.log 2>&1
echo "built $b at $(git -C "$W" rev-parse --short HEAD) -> $W/build/src/vkmark"
