#!/bin/bash
# Regenerate the whole-frame differential test from source: build wf, list the 1,251 programs, run
# every one under the model driver, check every trace and score the result against the stored
# validation results.
#
#   VKINC=<Vulkan-Headers include dir> WORK=<work dir> STORED=<raw_results.jsonl.gz> bash regenerate.sh
#
# Environment:
#   COVENANT_ROOT  checker root (default: this repository)
#   MD             model driver build dir (default: $COVENANT_ROOT/model_driver/build; build it first
#                  with `make -C model_driver VKINC=$VKINC VKH=$VKINC/vulkan/vulkan_core.h all`)
#   VKINC          Vulkan-Headers include dir (for build.sh)
#   WORK           work dir: build/, mutants.json, runs/, runs.jsonl, out-<label>/
#   STORED         the stored validation results, default: data/raw_results.jsonl.gz next to this script
#   MUTANTS        use this program list instead of generating one (the archive's inputs/mutants.json;
#                  gen_mutants.py regenerates the same list from wf --sites and its fixed seed)
#   JOBS           parallel runs and checks (default 4); runs need Xvfb (one display per job)
#   NO_TOOL=1      skip tools/check_trace.py per trace (eligibility); timing-only is much faster
# Output: $WORK/out-<label>/tables_<label>.md (printed) and the files listed in score.py.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "${COVENANT_ROOT:-$HERE/../..}" && pwd)
: "${WORK:?set WORK to a work directory}"
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
STORED=${STORED:-$HERE/data/raw_results.jsonl.gz}
STORED=$(readlink -f "$STORED")
MD=$(readlink -m "${MD:-$ROOT/model_driver/build}")
JOBS=${JOBS:-4}
PY=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1
[ -f "$MD/libvulkan.so.1" ] || { echo "model driver not built: $MD/libvulkan.so.1" >&2; exit 2; }
MD=$MD bash "$HERE/build.sh"
if [ -n "${MUTANTS:-}" ]; then
  cp "$MUTANTS" "$WORK/mutants.json"
else
  LD_LIBRARY_PATH=$MD $PY "$HERE/gen_mutants.py" --wf "$WORK/build/wf" --out "$WORK/mutants.json"
fi
$PY "$HERE/runner.py" --mutants "$WORK/mutants.json" --runs "$WORK/runs" --wf "$WORK/build/wf" \
  --shaders "$WORK/build/shaders" --md "$MD" --jobs "$JOBS" --out "$WORK/runs.jsonl"
LABEL=$(cd "$WORK" && $PY -c "import sys; sys.path.insert(0, sys.argv[1]); import check_cov as c; print(c.checker_version(sys.argv[2]))" "$HERE" "$ROOT")
OUT=$WORK/out-$LABEL
mkdir -p "$OUT"
TOOL=--with-check-trace
[ "${NO_TOOL:-0}" = 1 ] && TOOL=
(cd "$WORK" && COVENANT_ROOT="$ROOT" $PY -B "$HERE/check_cov.py" --root "$ROOT" --runs "$WORK/runs" \
   --mutants "$WORK/mutants.json" --jobs "$JOBS" $TOOL --out "$OUT/cov_$LABEL.jsonl")
(cd "$WORK" && $PY -B "$HERE/score.py" --mutants "$WORK/mutants.json" --stored "$STORED" \
   --runs-meta "$WORK/runs.jsonl" --cov "$OUT/cov_$LABEL.jsonl" --name "$LABEL" --out "$OUT")
echo "outputs in $OUT" >&2
