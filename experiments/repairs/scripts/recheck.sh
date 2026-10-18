#!/bin/bash
# Re-check every archived trace of the repair cases (traces archive, 152 runs) with a COVENANT checker and
# compare with the archived check results.
#
# usage: recheck.sh [out-dir]
#   [out-dir]       where results go (default: $WORK/recheck-<covenant_version>)
# env:
#   COVENANT_ROOT=<dir>       checker root with tools/check_trace.py and covenant/ (default: this repository)
#   ARCHIVE=<traces.tar.xz>   the traces archive (extracted once into <out-dir>/archive)
#   RUNS=<dir>                use an already extracted archive (its 'runs/' dir) instead of extracting
#   WORK=<dir>                work dir for the default out-dir
#   JOBS=<n>                  parallel checks (default: nproc, at most 8; a check needs ~0.1-0.3 GB RAM)
#   MODES="conservative observed"   capture modes to check (default both)
#   TMO=<seconds>             timeout per check (default 1800)
#   ONLY=<regex>              only run dirs whose relative path matches; the 15 fully repaired programs are
#                             ONLY='^(vkmark/all|gst/both|imgui/fix|sdl/fix)/'
#
# For each run dir (one program variant: trace.jsonl + spv/), in each mode: tools/check_trace.py --json and
# python -m covenant.report_groups --json, the commands of check.sh. Then a comparison against the archived
# results (the check.json files in the archive): per run and mode, diagnostics, groups, per-code counts,
# model warnings, unimplemented commands, timing.unmodelled*, and the per-finding (code, where, message)
# sets: 'lost' = archived but not reported now, 'added' = new.
# Output: <out-dir>/runs/<run>/<mode>/{check.json,check.txt,groups.json,groups.txt},
#         <out-dir>/COMPARE.md, <out-dir>/compare.json. Exit status 0 = every check ran (rc 0, json written);
#         1 = some check failed (listed in COMPARE.md); 2 = usage / setup error.
set -u
die() { echo "recheck.sh: $*" >&2; exit 2; }
H=$(dirname "$(readlink -f "$0")")
CROOT=$(readlink -f "${COVENANT_ROOT:-$H/../../..}") || die "no such checker root: ${COVENANT_ROOT:-}"
[ -f "$CROOT/tools/check_trace.py" ] && [ -d "$CROOT/covenant" ] || die "$CROOT has no tools/check_trace.py or covenant/"
command -v python3 >/dev/null || die "python3 not found"
VER=$(cd "$CROOT" && PYTHONDONTWRITEBYTECODE=1 python3 - "$CROOT" <<'EOF'
import hashlib, sys
from pathlib import Path
R = Path(sys.argv[1]); h = hashlib.sha1()
fs = sorted((R / "covenant").glob("*.py")) + ([R / "bench" / "run.py"] if (R / "bench" / "run.py").exists() else [])
for f in fs:
    h.update(f.name.encode()); h.update(f.read_bytes())
print(h.hexdigest()[:12])
EOF
) || die "cannot hash $CROOT/covenant"
if [ $# -ge 1 ]; then OUTD=$1; else [ -n "${WORK:-}" ] || die "give an out-dir or set WORK"; OUTD=$WORK/recheck-$VER; fi
mkdir -p "$OUTD" || die "cannot create $OUTD"; OUTD=$(readlink -f "$OUTD")
MODES=${MODES:-"conservative observed"}; TMO=${TMO:-1800}
J=${JOBS:-$(nproc 2>/dev/null || echo 2)}; [ "$J" -gt 8 ] && J=8
if [ -n "${RUNS:-}" ]; then
  RUNS=$(readlink -f "$RUNS"); [ -d "$RUNS" ] || die "RUNS=$RUNS is not a directory"
else
  [ -n "${ARCHIVE:-}" ] || die "set ARCHIVE to the traces archive or RUNS to its extracted runs/ dir"
  [ -f "$ARCHIVE" ] || die "archive not found: $ARCHIVE"
  X="$OUTD/archive"
  if [ ! -f "$X/.extracted" ]; then
    rm -rf "$X"; mkdir -p "$X"
    echo "extracting $ARCHIVE -> $X"
    tar -xJf "$ARCHIVE" -C "$X" || die "cannot extract $ARCHIVE"
    touch "$X/.extracted"
  fi
  RUNS=$X/runs
fi
[ -d "$RUNS" ] || die "no runs/ dir at $RUNS"
echo "checker root: $CROOT (covenant_version $VER); runs: $RUNS; out: $OUTD; jobs: $J; modes: $MODES"

# one check: <relative run dir> <mode>
one() {
  local r=$1 m=$2 d="$RUNS/$1" o="$OUTD/runs/$1/$2"
  mkdir -p "$o"
  ( export COVENANT_ROOT="$CROOT" PYTHONDONTWRITEBYTECODE=1 COVENANT_HOST_CAPTURE=$m
    timeout "$TMO" python3 "$CROOT/tools/check_trace.py" "$d/trace.jsonl" "$d/spv" --json "$o/check.json" > "$o/check.txt" 2>&1
    r1=$?
    cd "$CROOT" && PYTHONPATH="$CROOT" timeout "$TMO" python3 -m covenant.report_groups "$d/trace.jsonl" --spirv-dir "$d/spv" \
      --json "$o/groups.json" > "$o/groups.txt" 2> "$o/groups.err"
    r2=$?
    echo "check_rc=$r1 groups_rc=$r2" > "$o/status.txt" )
  echo "$r $m $(cat "$o/status.txt")"
}
export -f one; export RUNS OUTD CROOT TMO
LIST="$OUTD/list.txt"
(cd "$RUNS" && find . -name trace.jsonl -printf '%h\n' | sed 's#^\./##' | sort) > "$LIST"
[ -n "${ONLY:-}" ] && { grep -E "$ONLY" "$LIST" > "$LIST.f" || true; mv "$LIST.f" "$LIST"; }
n=$(wc -l < "$LIST"); [ "$n" -gt 0 ] || die "no trace.jsonl under $RUNS"
echo "$n runs x $(echo $MODES | wc -w) modes"
t0=$(date +%s)
for r in $(cat "$LIST"); do for m in $MODES; do echo "$r $m"; done; done | \
  xargs -P "$J" -L 1 bash -c 'one "$0" "$1"' > "$OUTD/progress.txt" 2>&1
echo "checks done in $(( $(date +%s) - t0 )) s"

python3 - "$RUNS" "$OUTD" "$LIST" "$VER" "$MODES" <<'EOF'
import json, os, re, sys, collections
runs, outd, lst, ver, modes = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].split()
def load(d):
    r = {'ok': False}
    try:
        st = open(os.path.join(d, 'status.txt')).read().strip()
    except OSError:
        st = 'no status'
    r['status'] = st
    try:
        cj = json.load(open(os.path.join(d, 'check.json')))
    except Exception as e:
        r['error'] = f'check.json: {type(e).__name__}'; return r
    f = cj.get('findings', []); s = cj.get('stats', {})
    try:
        g = open(os.path.join(d, 'groups.txt'), errors='replace').read().splitlines()
        m = re.search(r'-> (\d+) groups', g[0]) if g else None
        groups = int(m.group(1)) if m else None
    except OSError:
        groups = None
    r.update(ok=True, diagnostics=len(f), groups=groups,
             by_code=dict(sorted(collections.Counter(x.get('code') for x in f).items())),
             warnings=s.get('warnings'), unimplemented=s.get('unimplemented') or [],
             unmodelled=sum(1 for x in f if str(x.get('code', '')).startswith('timing.unmodelled')),
             version=s.get('covenant_version'),
             keys=collections.Counter((x.get('code'), x.get('where', ''), x.get('message', '')) for x in f))
    return r
rows, failed, out = [], [], {}
for r in open(lst).read().split():
    for m in modes:
        a = load(os.path.join(runs, r, m)); b = load(os.path.join(outd, 'runs', r, m))
        st = b.get('status', '')
        if not b['ok'] or not st.startswith('check_rc=0'):
            failed.append(f'{r} {m}: {st} {b.get("error", "")}')
        e = {'old': {k: v for k, v in a.items() if k != 'keys'}, 'new': {k: v for k, v in b.items() if k != 'keys'}}
        if a['ok'] and b['ok']:
            lost = a['keys'] - b['keys']; added = b['keys'] - a['keys']
            e['lost'] = sum(lost.values()); e['added'] = sum(added.values())
            e['lost_by_code'] = dict(collections.Counter(k[0] for k in lost.elements()))
            e['added_by_code'] = dict(collections.Counter(k[0] for k in added.elements()))
            e['lost_examples'] = [list(k) for k in list(lost)[:3]]; e['added_examples'] = [list(k) for k in list(added)[:3]]
        out[f'{r}/{m}'] = e
        o, n = e['old'], e['new']
        same = a['ok'] and b['ok'] and not e.get('lost') and not e.get('added') and o.get('groups') == n.get('groups')
        fmt = lambda x: '-' if not x.get('ok') else f"{x['diagnostics']} / {x['groups']}"
        prem = lambda x: '-' if not x.get('ok') else f"{x['warnings']} / {x['unmodelled']} / {','.join(x['unimplemented']) or 'none'}"
        rows.append(f"| {r} | {m} | {fmt(o)} | {fmt(n)} | {prem(n)} | {e.get('lost', '-')} | {e.get('added', '-')} | "
                    + ('same' if same else 'DIFFERENT') + ' |')
json.dump(out, open(os.path.join(outd, 'compare.json'), 'w'), indent=1)
L = [f'# Repairs recheck: archived results vs checker {ver}', '',
     f'runs: {runs}', '', 'old = the archived check results; new = this checker. '
     '"lost"/"added" count findings by (code, where, message).', '',
     '| run | mode | old diags / groups | new diags / groups | new warnings / unmodelled / unimplemented | lost | added | |',
     '| --- | --- | ---: | ---: | --- | ---: | ---: | --- |'] + rows
L += ['', f'checks that failed: {len(failed)}'] + [f'- {x}' for x in failed]
diff = [k for k, e in out.items() if e.get('lost') or e.get('added')]
L += ['', f'runs/modes with changed findings: {len(diff)}']
for k in diff:
    e = out[k]; L.append(f"- {k}: lost {e['lost_by_code']} added {e['added_by_code']}")
open(os.path.join(outd, 'COMPARE.md'), 'w').write('\n'.join(L) + '\n')
# terse console view: the repaired (clean) variants first
print('\n'.join(x for x in rows if re.match(r'\| (vkmark/all|gst/both|imgui/fix|sdl/fix)/', x)))
print(f'{len(rows)} run/mode rows, {len(diff)} with changed findings, {len(failed)} failed checks -> {outd}/COMPARE.md')
sys.exit(1 if failed else 0)
EOF
