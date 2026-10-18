#!/bin/bash
# Run the repair cases end to end: fetch and patch the sources (setup_sources.sh), build the variants
# that are missing, trace every program variant under the model driver (20 presents), check each trace
# in conservative and observed host capture, run validation on the repaired variants, and write
# $WORK/cases/<CASE>/summary.md and data/{runs,verdicts}.json (make_case.py).
#
# usage: WORK=<work dir> [VVL=<VVL build>] bash run_all.sh [CASE ...]     (default: all 11 cases)
# env: COVENANT_ROOT, MD, WORK (see common.sh); VVL (validation; NOSV=1 skips validation);
#      SV_VARIANTS  variants that get a validation run (default: the four fully repaired ones);
#      FORCE=1      re-trace and re-check runs that already have results.
# Traces and checks: $WORK/runs/{vkmark/<variant>,gst/<variant>,imgui/<variant>,sdl/<variant>}/<prog>/.
set -u
. "$(dirname "$(readlink -f "$0")")/common.sh"
H=$SCRIPTS; OUT=$WORK/runs; SUMDIR=$WORK/cases
SV_VARIANTS=${SV_VARIANTS:-"vkmark/all gst/both imgui/fix sdl/fix"}
CASES=${*:-"VKMARK-PSR VKMARK-ACQ VKMARK-UBO VKMARK-CLR VKMARK-DEPTH VKMARK-ALL IMGUI-PRE GST-WAIT GST-IA GST-ALL SDL-1"}
[ "${NOSV:-0}" = 1 ] || : "${VVL:?set VVL to a Vulkan-ValidationLayers build, or NOSV=1 to skip validation}"
bash "$H/setup_sources.sh" || exit 2
A="clear cube desktop effect2d shading texture vertex"; U="cube desktop shading texture vertex"; D="shading texture vertex"
GP="sink colorconvert overlay viewconvert shaderspv"
rc() {  # rc <run root> <prog> <tree>: trace + both checks (+ validation for the repaired variants)
  local o=$1 p=$2 S=$3 key=${1#$OUT/}
  [ -f "$o/$p/observed/check.json" ] && [ "${FORCE:-0}" != 1 ] && return 0
  bash "$H/trace.sh" "$o/$p" "$p" "$S" 20
  for m in conservative observed; do bash "$H/check.sh" "$o/$p" $m; done
  case " $SV_VARIANTS " in *" $key "*) [ "${NOSV:-0}" = 1 ] || bash "$H/sv.sh" "$o/sv_$p" "$p" "$S" 20;; esac
}
vk() { local v=$1; shift; [ -x "$T/vkm-$v/build/src/vkmark" ] || bash "$H/build_vkmark.sh" "$v"
  for p in "$@"; do rc "$OUT/vkmark/$v" "vkmark_$p" "$T/vkm-$v"; done; }
gs() { local v=$1; shift; [ -f "$T/gst/bad-$v/build/ext/vulkan/libgstvulkan.so" ] || bash "$H/build_gst.sh" "$T/gst/bad-$v"
  for p in "$@"; do rc "$OUT/gst/$v" "gst_$p" "$T/gst/bad-$v"; done; }
mc() { local c=$1; shift; mkdir -p "$SUMDIR/$c"; python3 "$H/make_case.py" "$SUMDIR/$c" "$c" "$@"; }
csv() { local pre=$1; shift; local o=""; for p in "$@"; do o="$o${o:+,}$pre$p"; done; echo "$o"; }
V=$OUT/vkmark; G=$OUT/gst
for c in $CASES; do case $c in
  VKMARK-PSR) vk base $A; vk psr $A; mc $c "$(csv vkmark_ $A)" original=$V/base fix=$V/psr;;
  VKMARK-ACQ) vk base $A; vk acq-slot $A; mc $c "$(csv vkmark_ $A)" original=$V/base fix=$V/acq-slot;;
  VKMARK-UBO) vk base $U; vk ubo-min $U; mc $c "$(csv vkmark_ $U)" original=$V/base fix=$V/ubo-min;;
  VKMARK-CLR) vk base clear; vk clr clear; mc $c vkmark_clear original=$V/base fix=$V/clr;;
  VKMARK-DEPTH) vk base $D; vk depth $D; mc $c "$(csv vkmark_ $D)" original=$V/base fix=$V/depth;;
  VKMARK-ALL) vk base $A; vk all $A; mc $c "$(csv vkmark_ $A)" original=$V/base fix=$V/all;;
  IMGUI-PRE) for v in "imgui orig" "imgui-fix fix"; do set -- $v
               [ -x "$T/$1/examples/example_glfw_vulkan/example_glfw_vulkan" ] || bash "$H/build_imgui.sh" "$T/$1"
               for p in glfw sdl2; do rc "$OUT/imgui/$2" imgui_$p "$T/$1"; done; done
             mc $c imgui_glfw,imgui_sdl2 original=$OUT/imgui/orig fix=$OUT/imgui/fix;;
  GST-WAIT) gs orig $GP; gs wait $GP; mc $c "$(csv gst_ $GP)" original=$G/orig fix=$G/wait;;
  GST-IA) gs orig $GP; gs ia $GP; mc $c "$(csv gst_ $GP)" original=$G/orig fix=$G/ia;;
  GST-ALL) gs orig $GP; gs both $GP; mc $c "$(csv gst_ $GP)" original=$G/orig fix=$G/both;;
  SDL-1) for v in "sdl orig" "sdl-fix fix"; do set -- $v
           [ -x "$T/$1/build/test/testsprite" ] || bash "$H/build_sdl.sh" "$T/$1"
           rc "$OUT/sdl/$2" sdl_testsprite "$T/$1"; done
         mc $c sdl_testsprite original=$OUT/sdl/orig fix=$OUT/sdl/fix;;
  *) echo "unknown case $c";;
esac; done
