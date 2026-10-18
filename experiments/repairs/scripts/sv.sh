#!/bin/bash
# Validation run of one program variant: lavapipe + the Khronos validation layer with synchronization
# validation (VVL 0ad67fd in the paper), ended after <presents> presents by the presentlimit shim.
# usage: sv.sh <out dir> <prog> <source tree> [presents]
# env: VVL     directory of a Vulkan-ValidationLayers build that holds VkLayer_khronos_validation.json
#      LVP_ICD lavapipe ICD manifest (default /usr/share/vulkan/icd.d/lvp_icd.json)
#      SHIMSO  presentlimit shim (default $WORK/presentlimit.so, built from presentlimit.c if missing)
set -u
. "$(dirname "$(readlink -f "$0")")/common.sh"
: "${VVL:?set VVL to a Vulkan-ValidationLayers build (the directory with VkLayer_khronos_validation.json)}"
VVL=$(readlink -m "$VVL")
LVP_ICD=$(readlink -m "${LVP_ICD:-/usr/share/vulkan/icd.d/lvp_icd.json}")
SHIMSO=$(readlink -m "${SHIMSO:-$WORK/presentlimit.so}")
[ -f "$SHIMSO" ] || cc -O2 -shared -fPIC -o "$SHIMSO" "$SCRIPTS/presentlimit.c" -ldl || exit 2
d=$(readlink -m "$1"); p=$2; S=$(readlink -m "$3"); P=${4:-20}
. "$SCRIPTS/progs.sh"
prog "$p" "$S" || exit 2
rm -rf "$d"; mkdir -p "$d"
cat > "$d/vk_layer_settings.txt" <<EOS
khronos_validation.debug_action = VK_DBG_LAYER_ACTION_LOG_MSG
khronos_validation.report_flags = error,warn
khronos_validation.log_filename = $d/vvl.txt
khronos_validation.validate_sync = true
khronos_validation.syncval_shader_accesses_heuristic = true
khronos_validation.syncval_load_op_after_store_op_validation = true
khronos_validation.enable_message_limit = false
khronos_validation.duplicate_message_limit = 100000
EOS
for try in 1 2 3; do  # retry only when the X server of xvfb-run could not be reached
  (cd "$WD" && timeout "${TMO:-1800}" xvfb-run -a -s "-screen 0 1280x720x24" env $ENVS COV_PRESENTS="$P" \
    VK_ICD_FILENAMES="$LVP_ICD" \
    VK_LAYER_PATH="$VVL" VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
    VK_LAYER_SETTINGS_PATH="$d/vk_layer_settings.txt" LD_PRELOAD="$SHIMSO" ${XLIB:+LD_LIBRARY_PATH=$XLIB} \
    $EXE $ARGS > "$d/out.txt" 2>&1)
  rc=$?; grep -q "connect to X server\|cannot open display" "$d/out.txt" || break; sleep 2; done
echo "rc=$rc sync=$(grep -c 'SYNC-HAZARD' "$d/vvl.txt" 2>/dev/null) vuid=$(grep -c 'VUID' "$d/vvl.txt" 2>/dev/null)" | tee "$d/status.txt"
