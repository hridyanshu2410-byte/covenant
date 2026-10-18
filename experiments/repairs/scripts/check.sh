#!/bin/bash
# Check one trace with the checker at $COVENANT_ROOT in one host-capture mode: tools/check_trace.py --json
# and python -m covenant.report_groups --json, into <run dir>/<mode>/.
# usage: check.sh <run dir> <conservative|observed>
. "$(dirname "$(readlink -f "$0")")/common.sh"
d=$(readlink -m "$1"); mode=${2:-conservative}; o=$d/$mode; mkdir -p "$o"
export PYTHONDONTWRITEBYTECODE=1 COVENANT_HOST_CAPTURE=$mode
timeout 3600 python3 "$COVENANT_ROOT/tools/check_trace.py" "$d/trace.jsonl" "$d/spv" --json "$o/check.json" > "$o/check.txt" 2>&1
r1=$?
(cd "$COVENANT_ROOT" && timeout 3600 python3 -m covenant.report_groups "$d/trace.jsonl" --spirv-dir "$d/spv" --json "$o/groups.json" > "$o/groups.txt" 2> "$o/groups.err")
echo "$(basename "$d") $mode check_rc=$r1 groups_rc=$?" | tee "$o/check_status.txt"
