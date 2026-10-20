#!/bin/bash
# Re-check every archived corpus of the paper with this repository's checker (the release) and compare with
# the archived results of the release. Nothing is re-executed on a Vulkan driver: the recorded traces are
# re-checked and re-grouped, and the stored validation verdicts are reused.
#
#   WORK=<work dir> bash experiments/recheck-release.sh [corpus ...]
#
# corpora (default: all): tests mutants elide archived wholeframe fresh repairs bench
#   tests      the test suite (pytest), with its pass/skip counts
#   mutants    the 750 barrier mutants: verdicts and diagnostics per mutant
#   elide      the barrier-elision plans of the three microbenchmarks, re-derived from their traces:
#              the remove / narrow / keep decision of every barrier call against bench/elide/plans
#   archived   the 28 archived sample, engine, adjudication and ray-tracing/mesh-shading traces
#   wholeframe the 1,251 whole-frame traces, with eligibility (tools/check_trace.py on every trace)
#   fresh      the 57 fresh-study programs in both capture modes: check, grouping, class totals
#   repairs    the 152 recorded runs of the repair cases (both modes), incl. the 15 repaired programs
#   bench      the 114 GPUBugBench reproductions (needs the model driver build, model_driver/build)
# Env: JOBS (default 4). gtk_path_walk needs about 8 GB of memory to group.
# Output: $WORK/<corpus>/..., $WORK/<corpus>.log and $WORK/SUMMARY.md (fingerprint, tool versions, one line
# per corpus). Exit 1 if any corpus differs from the archived results.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${WORK:?set WORK to a work directory}"
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
JOBS=${JOBS:-4}
CORPORA=${*:-tests mutants elide archived wholeframe fresh repairs bench}
VER=$(cd "$R" && python3 -B -c 'import sys; sys.path.insert(0, "tools"); from check_trace import covenant_version; print(covenant_version())')
X=$R/experiments
S=$WORK/SUMMARY.md
{
  echo "# Re-check of the archived corpora with checker $VER"
  echo
  echo "- commit: $(cd "$R" && git rev-parse --short HEAD 2>/dev/null || echo unknown)$(cd "$R" && git diff --quiet 2>/dev/null || echo ' (with local changes)')"
  echo "- python: $(python3 --version 2>&1); libclang: $(python3 -c 'import clang.cindex as c; print(c.conf.lib.clang_getClangVersion and "yes")' 2>/dev/null || echo no)"
  echo "- spirv-val: $(spirv-val --version 2>/dev/null | head -1 || echo none); glslangValidator: $(glslangValidator --version 2>/dev/null | head -1 || echo none)"
  echo "- host: $(uname -sr), $(nproc) cores"
  echo
} > "$S"
status=0
note() { echo "- **$1**: $2" >> "$S"; echo "$1: $2"; }

for c in $CORPORA; do
  t0=$(date +%s); L=$WORK/$c.log
  case $c in
  tests)
    (cd "$R" && python3 -m pytest -q -rs tests) > "$L" 2>&1; rc=$?
    note tests "$(grep -E '[0-9]+ (passed|failed)' "$L" | tail -1) (exit $rc)"; [ $rc -eq 0 ] || status=1 ;;
  mutants)
    M=$WORK/mutants; mkdir -p "$M"; tar -xJf "$X/mutants/data/mutants.tar.xz" -C "$M"
    python3 -B "$R/bench/elide/difftest/recheck.py" --traces "$M/traces" --mutants "$M/mutants" \
      --results "$M/results.jsonl" --out "$M/recheck.json" --per-mutant "$M/per-mutant.jsonl" > "$L" 2>&1
    python3 -B "$X/recheck_compare.py" mutants "$X/mutants/data/per-mutant-release.jsonl" "$M/per-mutant.jsonl" >> "$L" 2>&1 || status=1
    note mutants "$(grep 'barrier mutants' "$L"); cells $(python3 -c "import json,collections;d=json.load(open('$M/recheck.json'))['per_mutant'];c=collections.Counter();[c.update({k.split(':')[1]:v}) for k,v in d.items()];print(dict(c))")" ;;
  elide)
    E=$WORK/elide; mkdir -p "$E"; : > "$L"
    for w in deferred particles postchain; do
      (cd "$R" && python3 -B -m covenant.elide bench/elide/traces/$w.jsonl --spirv-dir bench/elide/traces/$w-spv -o "$E/$w.plan.json") >> "$L" 2>&1
    done
    python3 -B - "$R/bench/elide/plans" "$E" >> "$L" 2>&1 <<'PY' || status=1
import json, sys
calls = same = 0; masks = 0; tot = {}
for w in ("deferred", "particles", "postchain"):
    a = json.load(open(f"{sys.argv[1]}/{w}.plan.json")); b = json.load(open(f"{sys.argv[2]}/{w}.plan.json"))
    da, db = [x["decision"] for x in a["barriers"]], [x["decision"] for x in b["barriers"]]
    calls += len(db); same += sum(x == y for x, y in zip(da, db)) if len(da) == len(db) else 0
    masks += sum(x != y for x, y in zip(a["barriers"], b["barriers"]))
    for k, v in b["summary"]["decisions"].items(): tot[k] = tot.get(k, 0) + v
print(f"elision plans: {calls} barrier calls, {same} with the archived decision; decisions {tot}; "
      f"{masks} calls whose narrowed masks differ from the archived plans")
sys.exit(0 if same == calls else 1)
PY
    note elide "$(grep 'elision plans' "$L")" ;;
  archived)
    A=$WORK/archived; mkdir -p "$A/in" "$A/out"; tar -xJf "$X/archived-traces/data/archived-traces.tar.xz" -C "$A/in"
    (cd "$A/in" && find . -name trace.jsonl -printf '%h\n') | while read -r d; do
      mkdir -p "$A/out/$d"
      (cd "$R" && python3 -B tools/check_trace.py "$A/in/$d/trace.jsonl" "$A/in/$d/spv" --json "$A/out/$d/check.json" > /dev/null 2>&1)
    done
    python3 -B "$X/recheck_compare.py" archived "$A/in" "$A/out" > "$L" 2>&1 || status=1
    note archived "$(grep 'archived traces' "$L")" ;;
  wholeframe)
    ARCHIVE=$X/wholeframe/data/wholeframe-traces.tar.xz WORK=$WORK/wholeframe JOBS=$JOBS \
      bash "$X/wholeframe/recheck.sh" > "$L" 2>&1 || status=1
    note wholeframe "$(grep -m1 -E 'checked [0-9]+ traces' "$L"); $(grep -m1 -o 'hazards-only verdict agreement [0-9.]*%' "$L"); $(grep -m1 -o 'hazard pairs [0-9 /]*' "$L")" ;;
  fresh)
    F=$WORK/fresh; mkdir -p "$F/traces" "$F/stored"
    tar -xJf "$X/fresh-studies/data/traces.tar.xz" -C "$F/traces"; tar -xJf "$X/fresh-studies/data/release-checks.tar.xz" -C "$F/stored"
    for s in 1 2; do for p in $(ls "$F/traces/f$s/runs"); do for m in conservative observed; do echo "$s $p $m"; done; done; done | \
    xargs -P "$JOBS" -L 1 bash -c '
      s=$0 p=$1 m=$2 d='"$F"'/traces/f$0/runs/$1 o='"$F"'/out/study$0/$1/$2; mkdir -p $o
      cd '"$R"' && COVENANT_HOST_CAPTURE=$m python3 -B tools/check_trace.py $d/trace.jsonl $d/spv --json $o/check.json > $o/check.txt 2>&1
      cd '"$R"' && COVENANT_HOST_CAPTURE=$m python3 -B -m covenant.report_groups $d/trace.jsonl --spirv-dir $d/spv --json $o/groups.json > $o/groups.txt 2>&1
      echo "$0 $1 $2 done"' > "$F/progress.txt" 2>&1
    python3 -B "$X/recheck_compare.py" fresh "$F/stored" "$F/out" "$X/fresh-studies/data" > "$L" 2>&1 || status=1
    note fresh "$(grep 'fresh studies' "$L")"; grep '^  study' "$L" | sed 's/^  /  - /' >> "$S" ;;
  repairs)
    ARCHIVE=$X/repairs/data/traces.tar.xz JOBS=$JOBS bash "$X/repairs/scripts/recheck.sh" "$WORK/repairs" > "$L" 2>&1 || status=1
    note repairs "$(tail -1 "$L")" ;;
  bench)
    (cd "$R" && python3 -B bench/run.py --md "$WORK/bench.md" --json "$WORK/bench.json") > "$L" 2>&1
    if diff -q "$WORK/bench.md" "$R/bench/results-release.md" > /dev/null; then
      note bench "$(grep -c '^| [0-9]' "$WORK/bench.md") cases, table identical to bench/results-release.md"
    else note bench "table differs from bench/results-release.md (see $WORK/bench.md)"; status=1; fi ;;
  *) echo "unknown corpus $c" >&2; exit 2 ;;
  esac
  echo "  ($c: $(( $(date +%s) - t0 )) s)" >> "$S"
done
echo >> "$S"; echo "overall: $([ $status -eq 0 ] && echo 'identical to the archived results' || echo 'DIFFERENCES, see the logs')" >> "$S"
cat "$S"
exit $status
