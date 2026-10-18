#!/bin/bash
# Re-check every archived whole-frame trace (1,251 programs) with a COVENANT checker and recompute the
# agreement with the STORED Khronos-validation results (validation is not re-run).
#
#   ARCHIVE=<wholeframe-traces.tar.xz> WORK=<work dir> bash recheck.sh [label]
#
# Environment:
#   COVENANT_ROOT  checker root, the directory that contains covenant/ and tools/check_trace.py
#                  (default: this repository). It is only read: Python runs with -B /
#                  PYTHONDONTWRITEBYTECODE=1 and with another working directory.
#   ARCHIVE        the traces archive; extracted once to $WORK/wholeframe
#   WORK           work dir
#   JOBS           parallel checks (default 4)
#   NO_TOOL=1      skip tools/check_trace.py (eligibility, all checkers; about 1 s per trace);
#                  the timing-only check of all 1,251 traces takes seconds
#   COVENANT_HOST_CAPTURE / COVENANT_HEURISTIC_DISCARD / COVENANT_LEGACY_SUBPASS_FLAGS: passed through
#                  (leave unset for the default, verified configuration)
# label: name for the outputs (default: the checker's covenant_version hash).
# Output: $WORK/out-<label>/{cov_<label>.jsonl, tables_<label>.md/.json, changes_<label>.json,
#         analysis_<label>{,+afu}.json}; tables_<label>.md is also printed. When the archive holds the
#         archived checker results (results/cov_release.jsonl), changes_<label>.json also compares
#         with them per program. Exit 1 on checker errors, a failed self-check (stored tables not
#         reproduced) or a missing input.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "${COVENANT_ROOT:-$HERE/../..}" && pwd) || { echo "no such checker root: ${COVENANT_ROOT:-}" >&2; exit 2; }
[ -d "$ROOT/covenant" ] || { echo "$ROOT has no covenant/ package" >&2; exit 2; }
: "${WORK:?set WORK to a work directory}"
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
JOBS=${JOBS:-4}
PY=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1
BASE=$WORK/wholeframe
if [ ! -f "$BASE/.extracted" ]; then
  : "${ARCHIVE:?set ARCHIVE to wholeframe-traces.tar.xz}"
  [ -f "$ARCHIVE" ] || { echo "archive not found: $ARCHIVE" >&2; exit 2; }
  echo "extracting $ARCHIVE -> $WORK" >&2
  rm -rf "$BASE"
  if xz --help 2>/dev/null | grep -q -- '-T'; then tar -I 'xz -T0' -xf "$ARCHIVE" -C "$WORK"; else tar -xJf "$ARCHIVE" -C "$WORK"; fi
  touch "$BASE/.extracted"
fi
n=$(ls "$BASE/runs" | wc -l)
[ "$n" -eq 1251 ] || { echo "expected 1251 run dirs in $BASE/runs, found $n" >&2; exit 1; }
for f in check_cov.py score.py analyze.py vvlparse.py; do
  [ -f "$HERE/$f" ] || { echo "missing $HERE/$f" >&2; exit 2; }
done
HASH=$(cd "$WORK" && $PY -B -c "import sys; sys.path.insert(0, sys.argv[1]); import check_cov as c; print(c.checker_version(sys.argv[2]))" "$HERE" "$ROOT")
LABEL=${1:-$HASH}
OUT=$WORK/out-$LABEL
mkdir -p "$OUT"
TOOL=--with-check-trace
if [ "${NO_TOOL:-0}" = 1 ] || [ ! -f "$ROOT/tools/check_trace.py" ]; then TOOL=; fi
echo "checker $ROOT (covenant_version $HASH), label $LABEL, jobs $JOBS, check_trace: ${TOOL:-no}" >&2
t0=$(date +%s)
(cd "$WORK" && COVENANT_ROOT="$ROOT" $PY -B "$HERE/check_cov.py" --root "$ROOT" --runs "$BASE/runs" \
   --mutants "$BASE/inputs/mutants.json" --jobs "$JOBS" $TOOL --out "$OUT/cov_$LABEL.jsonl")
t1=$(date +%s)
nerr=$($PY -B -c "import json,sys; print(sum(json.loads(l).get('error') is not None for l in open(sys.argv[1])))" "$OUT/cov_$LABEL.jsonl")
[ "$nerr" -eq 0 ] || { echo "$nerr traces failed to check; see 'error' in $OUT/cov_$LABEL.jsonl" >&2; exit 1; }
CMP=()
[ -f "$BASE/results/cov_release.jsonl" ] && CMP+=(--compare "archived=$BASE/results/cov_release.jsonl")
(cd "$WORK" && $PY -B "$HERE/score.py" --base "$BASE" --cov "$OUT/cov_$LABEL.jsonl" --name "$LABEL" \
   --out "$OUT" "${CMP[@]}")
echo "check: $((t1 - t0)) s; outputs in $OUT" >&2
