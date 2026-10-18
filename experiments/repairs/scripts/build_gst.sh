#!/bin/bash
# Build the Vulkan library and plugin of gst-plugins-bad 1.24.2 (GStreamer monorepo tag 1.24.2, 2d82731515)
# for one variant tree. Needs meson, ninja, libgstreamer1.0-dev and libgstreamer-plugins-base1.0-dev 1.24.2,
# glslc, libxkbcommon-x11-dev. usage: build_gst.sh <tree>
set -eu
S=$1; cd "$S"; rm -rf build
meson setup build -Dauto_features=disabled -Dvulkan=enabled -Dx11=enabled -Dintrospection=disabled \
  -Dexamples=disabled -Dtests=disabled -Dgpl=disabled -Dnls=disabled > build.log 2>&1
ninja -C build gst-libs/gst/vulkan/libgstvulkan-1.0.so.0.2402.0 ext/vulkan/libgstvulkan.so >> build.log 2>&1
echo "built $S ($(git -C "$S" rev-parse --short HEAD))"
