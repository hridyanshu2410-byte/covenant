#!/bin/bash
# Fetch the four projects at the tested revisions into $WORK/src and apply the case patches
# (../patches/<CASE>.diff), one source tree per variant:
#   vkmark ab6e6f3:        vkm-base (original), vkm-psr, vkm-acq-slot, vkm-ubo-min, vkm-clr, vkm-depth, vkm-all
#   Dear ImGui 9d9a621f:   imgui (original), imgui-fix
#   SDL e4af6f8e:          sdl (original), sdl-fix
#   gst-plugins-bad 1.24.2 (GStreamer monorepo tag 1.24.2 = 2d82731515):
#                          gst/bad-orig (original), gst/bad-wait, gst/bad-ia, gst/bad-both
# GST_REPO overrides the GStreamer repository (default: the freedesktop.org GitLab; the GitHub
# mirror https://github.com/GStreamer/gstreamer.git has the same tag).
set -eu
. "$(dirname "$(readlink -f "$0")")/common.sh"
GST_REPO=${GST_REPO:-https://gitlab.freedesktop.org/gstreamer/gstreamer.git}
mkdir -p "$T"
if [ ! -d "$T/vkmark" ]; then
  git clone -q https://github.com/vkmark/vkmark.git "$T/vkmark"
  git -C "$T/vkmark" branch base ab6e6f3
  for v in "psr VKMARK-PSR" "clr VKMARK-CLR" "depth VKMARK-DEPTH" "ubo-min VKMARK-UBO" "acq-slot VKMARK-ACQ" \
           "all VKMARK-ALL"; do
    set -- $v; git -C "$T/vkmark" worktree add -q -b "$1" "$T/vkm-$1" base; git -C "$T/vkm-$1" apply "$PATCHES/$2.diff"
  done
  git -C "$T/vkmark" worktree add -q "$T/vkm-base" base
fi
if [ ! -d "$T/imgui" ]; then
  mkdir "$T/imgui" && git -C "$T/imgui" init -q
  git -C "$T/imgui" fetch -q --depth 1 https://github.com/ocornut/imgui.git 9d9a621f339460e8a9dc49b2ad164d6acade1aff
  git -C "$T/imgui" checkout -q FETCH_HEAD; cp -a "$T/imgui" "$T/imgui-fix"; git -C "$T/imgui-fix" apply "$PATCHES/IMGUI-PRE.diff"
fi
if [ ! -d "$T/sdl" ]; then
  mkdir "$T/sdl" && git -C "$T/sdl" init -q
  git -C "$T/sdl" fetch -q --depth 1 https://github.com/libsdl-org/SDL.git e4af6f8e283812aa693eb2c70c91249a9e1ce92c
  git -C "$T/sdl" checkout -q FETCH_HEAD; cp -a "$T/sdl" "$T/sdl-fix"; git -C "$T/sdl-fix" apply "$PATCHES/SDL-1.diff"
fi
if [ ! -d "$T/gst/bad-orig" ]; then
  mkdir -p "$T/gst"; G=$T/gst/mono
  [ -d "$G" ] || git clone -q --depth 1 -b 1.24.2 "$GST_REPO" "$G"
  echo "gstreamer monorepo: $(git -C "$G" rev-parse HEAD) (expected 2d82731515...)"
  cp -a "$G/subprojects/gst-plugins-bad" "$T/gst/bad-orig"
  (cd "$T/gst/bad-orig" && git init -q && git add -A && git -c user.email=build@localhost -c user.name=build commit -qm "gst-plugins-bad 1.24.2" && git branch -m orig)
  for v in "wait GST-WAIT" "ia GST-IA" "both GST-ALL"; do set -- $v
    git -C "$T/gst/bad-orig" worktree add -q -b "$1" "$T/gst/bad-$1" orig; git -C "$T/gst/bad-$1" apply "$PATCHES/$2.diff"; done
fi
echo "sources ready under $T"
