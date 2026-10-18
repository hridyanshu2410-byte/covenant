"""Check one model-driver trace of an UNANNOTATED real program with the same
checkers bench/run.py uses (timing, per-draw structure, meaning in
inferred-only mode, optional host-static layouts).

    python3 check_trace.py <trace.jsonl> <spv-dump-dir> [--json out.json]
        [--host-src file.cpp ...] [--host-arg -I... ...]
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("COVENANT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from covenant import spirv  # noqa: E402
from covenant.diagnostics import Report  # noqa: E402
from covenant.host_trace import load_trace  # noqa: E402
from covenant.program import check_program  # noqa: E402  (exactly what bench/run.py runs)


def load_modules(spv_dir):
    mods = {}
    for p in sorted(Path(spv_dir).glob("*.spv")):
        try:
            m = spirv.parse(p)
        except Exception as e:                      # unparseable module: skipped, reported
            print(f"warning: cannot parse {p.name}: {e}", file=sys.stderr)
            continue
        m.meta = spirv._empty_seeds()               # no annotations: inferred-only meaning
        m.meta["source"] = str(p)
        mods[hashlib.sha1(p.read_bytes()).hexdigest()] = m
    return mods


def covenant_version():
    """Release source hash: the first 12 hex digits of a sha1 over the name and bytes of each
    covenant/*.py (sorted) and then bench/run.py. It identifies the checker a result came from."""
    h = hashlib.sha1()
    for f in sorted((ROOT / "covenant").glob("*.py")) + [ROOT / "bench" / "run.py"]:
        h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()[:12]


def check(trace, spv_dir, host_src=(), host_args=()):
    t0 = time.time()
    model = load_trace(trace, spirv_dir=spv_dir)
    t_load = time.time() - t0
    mods = load_modules(spv_dir)
    r = Report()
    host_note = "skipped"
    if host_src:
        try:
            check_program(model, list(mods.values()), list(host_src), list(host_args), r)
            host_note = "parsed"
        except Exception as e:                      # libclang failure: rerun without host front end
            host_note = f"failed: {str(e)[:300]}"
            r = Report()
            check_program(model, list(mods.values()), (), (), r)
    else:
        check_program(model, list(mods.values()), (), (), r)
    meta = getattr(r, "meta", None) or {}
    if host_src and meta.get("host_static") and meta["host_static"] != "parsed":
        host_note = str(meta["host_static"])[:300]     # check_program reports host failures here
    stats = {"trace_records": len(model.records), "draws": len(model.draws),
             "pipelines_used": len({d.pipeline for d in model.draws}), "modules": len(mods),
             "load_s": round(t_load, 2),
             "check_s": round(time.time() - t0, 2), "warnings": len(model.warnings),
             "unimplemented": sorted({w.split(": ")[1] for w in model.warnings if ": " in w}),
             "presents": sum(1 for x in model.records if x["fn"] == "vkQueuePresentKHR"),
             "submits": sum(1 for x in model.records if x["fn"] == "vkQueueSubmit"),
             "host_static": host_note, "covenant_version": covenant_version(),
             "eligibility": meta.get("covenant.eligibility")}
    return r, stats


if __name__ == "__main__":
    a = sys.argv[1:]
    out = a[a.index("--json") + 1] if "--json" in a else None
    hsrc = [x for i, x in enumerate(a) if i and a[i - 1] == "--host-src"]
    harg = [x for i, x in enumerate(a) if i and a[i - 1] == "--host-arg"]
    r, st = check(a[0], a[1], hsrc, harg)
    print(r)
    print(json.dumps(st))
    if out:
        Path(out).write_text(json.dumps({"stats": st, "findings": [
            {"checker": d.checker, "code": d.code, "message": d.message, "where": d.where,
             "notes": d.notes, "severity": d.severity} for d in r.items]}, indent=1))
