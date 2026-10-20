#!/bin/bash
# Capture frames of the repair programs (Figure 8 and the teaser), original or patched, on lavapipe.
# usage: WORK=<repair work dir> capall.sh <outdir> <variant: orig|fix> prog...
# WORK is the work directory of experiments/repairs (its src/ holds the built source trees).
W=$(dirname "$(readlink -f "$0")"); O=$1; V=$2; shift 2; mkdir -p "$O"
: "${WORK:?set WORK to the repair work directory (experiments/repairs/README.md)}"
. "$W/../../../experiments/repairs/scripts/progs.sh"
for p in "$@"; do
  case $p in
    vkmark_*) S=$WORK/src/$([ "$V" = orig ] && echo vkm-base || echo vkm-all);;
    gst_*) S=$WORK/src/gst/$([ "$V" = orig ] && echo bad-orig || echo bad-both);;
    imgui_*) S=$WORK/src/$([ "$V" = orig ] && echo imgui || echo imgui-fix);;
    sdl_*) S=$WORK/src/$([ "$V" = orig ] && echo sdl || echo sdl-fix);;
  esac
  prog "$p" "$S" || continue
  ARGS=${ARGS//num-buffers=60/num-buffers=100000}   # keep the pipeline running until the grab
  D=${DELAY:-6}
  (cd "$WD" && env $ENVS ${XLIB:+LD_LIBRARY_PATH=$XLIB} python3 "$W/cap.py" "$O/$p.png" 1280x720 "$D" -- $EXE $ARGS)
done
