#!/bin/bash
# Trace one program variant under the model driver ($MD): trace.jsonl + spv/ in <out dir>.
# usage: trace.sh <out dir> <prog> <source tree> [presents]      (needs xvfb-run)
set -u
. "$(dirname "$(readlink -f "$0")")/common.sh"
d=$(readlink -m "$1"); p=$2; S=$(readlink -m "$3"); P=${4:-20}
. "$SCRIPTS/progs.sh"
prog "$p" "$S" || exit 2
rm -rf "$d"; mkdir -p "$d/spv"
start=$(date +%s.%N)
for try in 1 2 3; do  # retry only when the X server of xvfb-run could not be reached
  (cd "$WD" && timeout "${TMO:-900}" xvfb-run -a -s "-screen 0 1280x720x24" env $ENVS COVENANT_MAX_PRESENTS="$P" \
    COVENANT_SURFACE_EXTENT=1280x720 COVENANT_TRACE="$d/trace.jsonl" COVENANT_SPIRV_DIR="$d/spv" \
    LD_LIBRARY_PATH="$MD${XLIB:+:$XLIB}" $EXE $ARGS > "$d/out.txt" 2>&1)
  rc=$?; grep -q "connect to X server\|cannot open display" "$d/out.txt" || break; sleep 2; done
echo "rc=$rc run_s=$(echo "$(date +%s.%N) - $start" | bc) presents_req=$P" > "$d/status.txt"
echo "$p $(cat "$d/status.txt") records=$(wc -l < "$d/trace.jsonl" 2>/dev/null) presents=$(grep -c '"fn":"vkQueuePresentKHR' "$d/trace.jsonl" 2>/dev/null) hostw=$(grep -c '"fn":"host_write' "$d/trace.jsonl" 2>/dev/null) unimpl=$(grep -c unimplemented "$d/trace.jsonl" 2>/dev/null)" | tee "$d/trace_status.txt"
