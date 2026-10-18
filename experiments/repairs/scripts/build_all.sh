#!/bin/bash
# Build every variant tree under $WORK/src, one at a time.
. "$(dirname "$(readlink -f "$0")")/common.sh"
for v in all base psr clr depth ubo-min acq-slot; do
  bash "$SCRIPTS/build_vkmark.sh" "$v" || echo "BUILD FAILED vkmark $v"; done
for t in imgui-fix imgui; do bash "$SCRIPTS/build_imgui.sh" "$T/$t" || echo "BUILD FAILED $t"; done
for v in both orig wait ia; do bash "$SCRIPTS/build_gst.sh" "$T/gst/bad-$v" || echo "BUILD FAILED gst $v"; done
for t in sdl-fix sdl; do bash "$SCRIPTS/build_sdl.sh" "$T/$t" || echo "BUILD FAILED $t"; done
echo done
