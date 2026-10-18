"""COVENANT verdict for model-driver traces: host_trace.load_trace + timing.check(window=False),
the timing check that the whole-frame differential test scores.

    one trace:  python3 check_cov.py --root CHECKER_ROOT --run RUN_DIR [--json out.json]
    batch:      python3 check_cov.py --root CHECKER_ROOT --runs RUNS_DIR --out results.jsonl
                                       [--jobs 4] [--ids a,b,...] [--mutants mutants.json]

RUN_DIR holds trace.jsonl and spv/. The checker is imported from CHECKER_ROOT (the directory that
contains covenant/, normally the repository root); COVENANT_HOST_CAPTURE / COVENANT_HEURISTIC_DISCARD are left at their defaults
(conservative host capture, spec-strict) unless set in the environment. Batch mode checks each
trace in a worker process (maxtasksperchild bounds any state a checker might keep between traces).
--with-check-trace additionally runs CHECKER_ROOT/tools/check_trace.py on each trace in a fresh
process and records its stats JSON (including "eligibility") and finding codes.
Run with PYTHONDONTWRITEBYTECODE=1 to leave a read-only checker tree untouched.

Per trace: {id, config, mutant, t_load, t_check, t_total, diagnostics: [{code, res, where, message,
notes, severity, hazard}], hazards: [...], warnings, unsupported, modes, desc_refine, error}.
"""
import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import time
import traceback

HAZ_CODES = {"timing.missing-barrier", "timing.barrier-scope", "timing.wait-stage-scope",
             "timing.cross-queue-race", "timing.free-while-in-use", "timing.signal-stage-scope"}
_ROOT = None
_WITH_TOOL = False


def checker_version(root):
    """sha1 over covenant/*.py and bench/run.py (= tools/check_trace.py covenant_version())."""
    h = hashlib.sha1()
    fs = sorted(f for f in os.listdir(os.path.join(root, "covenant")) if f.endswith(".py"))
    paths = [os.path.join(root, "covenant", f) for f in fs]
    if os.path.exists(os.path.join(root, "bench", "run.py")):
        paths.append(os.path.join(root, "bench", "run.py"))
    for p in paths:
        h.update(os.path.basename(p).encode())
        h.update(open(p, "rb").read())
    return h.hexdigest()[:12]


def _init(root, with_tool=False):
    global _ROOT, _WITH_TOOL
    _ROOT, _WITH_TOOL = root, with_tool
    if root not in sys.path:
        sys.path.insert(0, root)


def _jsonable(x):
    try:
        json.dumps(x)
        return x
    except TypeError:
        if isinstance(x, dict):
            return {str(k): _jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, set, frozenset)):
            return [_jsonable(v) for v in (sorted(x, key=str) if isinstance(x, (set, frozenset)) else x)]
        return str(x)


def check(run_dir):
    from covenant.host_trace import load_trace
    from covenant import timing
    from covenant.diagnostics import Report
    try:
        from covenant import modes
        mode = modes.describe()
    except Exception:
        mode = None
    t0 = time.perf_counter()
    m = load_trace(os.path.join(run_dir, "trace.jsonl"), spirv_dir=os.path.join(run_dir, "spv"))
    t1 = time.perf_counter()
    hz = []
    r = timing.check(m.frame, Report(), hazards=hz, window=False)
    t2 = time.perf_counter()
    fr = m.frame

    def op_of(key):
        q, seq = key
        ops = fr.queues.get(q, [])
        return ops[seq - 1] if 0 < seq <= len(ops) else None

    def kind(o):
        return "write" if isinstance(o, timing.Barrier) else getattr(o, "access", None)

    hazards = []
    for e, u, code in hz:
        oe, ou = op_of(e), op_of(u)
        hazards.append({"code": code, "res": getattr(ou, "res", None), "e_q": e[0], "u_q": u[0],
                        "e_kind": kind(oe), "u_kind": kind(ou),
                        "e_lt": isinstance(oe, timing.Barrier), "u_lt": isinstance(ou, timing.Barrier),
                        "e_where": (getattr(oe, "where", "") or "")[:200],
                        "u_where": (getattr(ou, "where", "") or "")[:200]})
    q = re.compile(r"'([^']+)'")
    diags = []
    for d in r.items:
        mm = q.search(d.message)
        diags.append({"code": d.code, "res": mm.group(1) if mm else None, "where": d.where,
                      "message": d.message, "notes": [str(n) for n in (d.notes or [])],
                      "severity": getattr(d, "severity", "error"), "hazard": d.code in HAZ_CODES})
    meta = getattr(r, "meta", {}) or {}
    return {"t_load": t1 - t0, "t_check": t2 - t1, "t_total": t2 - t0, "diagnostics": diags,
            "hazards": hazards, "warnings": [str(w) for w in list(m.warnings)],
            "unsupported": _jsonable(dict(m.unsupported or {})),
            "frame_unsupported": _jsonable(dict(getattr(fr, "unsupported", None) or {})),
            "modes": mode or _jsonable(meta.get("covenant.modes")),
            "desc_refine": _jsonable(meta.get("timing.desc_refine")),
            "eligibility_timing": _jsonable(meta.get("covenant.eligibility"))}


def check_trace_tool(run_dir, root, timeout=600):
    """The checker's own tools/check_trace.py (all checkers, as bench/run.py) in a fresh process,
    without writing bytecode into the checker tree: returns its stats JSON (incl. "eligibility")
    and its finding codes."""
    import subprocess
    import tempfile
    tool = os.path.join(root, "tools", "check_trace.py")
    if not os.path.exists(tool):
        return {"error": "no tools/check_trace.py"}
    fd, out = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    env = dict(os.environ, COVENANT_ROOT=root, PYTHONDONTWRITEBYTECODE="1")
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, "-B", tool, os.path.join(run_dir, "trace.jsonl"),
                            os.path.join(run_dir, "spv"), "--json", out], env=env, capture_output=True,
                           text=True, timeout=timeout, cwd=tempfile.gettempdir())
        if p.returncode != 0:
            return {"error": f"rc={p.returncode}: {p.stderr[-1500:]}", "t_wall": time.perf_counter() - t0}
        j = json.load(open(out))
        st = j.get("stats", {})
        codes = collections.Counter(f["code"] for f in j.get("findings", []) if f.get("severity") != "warning")
        wcodes = collections.Counter(f["code"] for f in j.get("findings", []) if f.get("severity") == "warning")
        return {"eligibility": st.get("eligibility"), "stats": st, "codes": dict(codes),
                "warning_codes": dict(wcodes), "t_wall": time.perf_counter() - t0}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "t_wall": time.perf_counter() - t0}
    finally:
        os.path.exists(out) and os.remove(out)


def _one(job):
    mid, run_dir, info = job
    rec = {"id": mid}
    rec.update(info)
    t0 = time.perf_counter()
    try:
        if not os.path.exists(os.path.join(run_dir, "trace.jsonl")):
            raise FileNotFoundError(os.path.join(run_dir, "trace.jsonl"))
        rec.update(check(run_dir))
        rec["error"] = None
        if _WITH_TOOL:
            rec["check_trace"] = check_trace_tool(run_dir, _ROOT)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-3000:]
    rec["t_wall"] = time.perf_counter() - t0
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run")
    ap.add_argument("--json")
    ap.add_argument("--runs")
    ap.add_argument("--out")
    ap.add_argument("--mutants")
    ap.add_argument("--ids", default="")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--with-check-trace", action="store_true",
                    help="also run the checker's tools/check_trace.py per trace and record its stats "
                         "(eligibility) and finding codes")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    # tools/check_trace.py runs in another working directory: pass it absolute paths
    a.run = a.run and os.path.abspath(a.run)
    a.runs = a.runs and os.path.abspath(a.runs)
    if not os.path.isdir(os.path.join(root, "covenant")):
        sys.exit(f"no covenant/ package under {root}")
    ver = checker_version(root)
    if a.run:
        _init(root, a.with_check_trace)
        out = _one((os.path.basename(os.path.normpath(a.run)), a.run, {"checker_version": ver}))
        s = json.dumps(out, indent=1)
        if a.json:
            open(a.json, "w").write(s)
        else:
            print(s[:8000])
        return
    info = {}
    if a.mutants:
        for m in json.load(open(a.mutants)):
            info[m["id"]] = {"config": m["config"], "mutant": m["mut"], "gen": m["gen"]}
    ids = sorted(a.ids.split(",")) if a.ids else sorted(
        x for x in os.listdir(a.runs) if os.path.isdir(os.path.join(a.runs, x)))
    if info:
        missing = sorted(set(info) - set(ids)) if not a.ids else []
        if missing:
            print(f"warning: {len(missing)} mutants have no run dir, e.g. {missing[:5]}", file=sys.stderr)
    jobs = [(i, os.path.join(a.runs, i), dict(info.get(i, {}), checker_version=ver)) for i in ids]
    t0 = time.time()
    n_err = 0
    with mp.Pool(a.jobs, initializer=_init, initargs=(root, a.with_check_trace), maxtasksperchild=50) as pool, \
            open(a.out, "w") as f:
        for k, rec in enumerate(pool.imap(_one, jobs, chunksize=4)):
            n_err += rec["error"] is not None
            f.write(json.dumps(rec) + "\n")
            if (k + 1) % 200 == 0:
                print(f"checked {k + 1}/{len(jobs)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    print(f"checked {len(jobs)} traces with {root} (checker {ver}) in {time.time() - t0:.1f}s, "
          f"{n_err} errors", file=sys.stderr)


if __name__ == "__main__":
    main()
