"""GPUBugBench runner: build each reproduction (buggy and FIXED), run it on
the COVENANT model driver, and check it with every COVENANT checker.

    python3 bench/run.py [case-dir-substring ...] [--json out.json] [--md out.md]

A case passes when the buggy variant reports every expected code and the
fixed variant reports nothing. Cases marked "known_miss" are expected to
report nothing in both variants; they are counted as misses, honestly.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from covenant import spirv  # noqa: E402
from covenant.diagnostics import Report  # noqa: E402
from covenant.host_trace import load_trace  # noqa: E402
from covenant.program import check_program  # noqa: E402

CASES = ROOT / "bench" / "cases"
DRIVER = ROOT / "model_driver" / "build"
INCLUDE = ROOT / "covenant" / "include"
CXX = os.environ.get("CXX", "clang++-20")
WORK = Path(os.environ.get("COVBENCH_WORK", tempfile.gettempdir())) / "covbench"



def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))}\n{r.stdout}\n{r.stderr}")
    return r


def build_variant(case, fixed):
    out = WORK / case.name / ("fixed" if fixed else "bug")
    out.mkdir(parents=True, exist_ok=True)
    defs = ["-DFIXED"] if fixed else []
    modules = []
    for src in sorted(case.glob("*.slang")):
        modules.append(spirv.compile_slang(src, out, defs))
    exe = out / "host"
    t = time.time()
    sh([CXX, "-std=c++17", "-g", "-O0", f"-I{INCLUDE}", *defs, str(case / "host.cpp"), "-o", str(exe),
        f"-L{DRIVER}", "-lcovvk", f"-Wl,-rpath,{DRIVER}"])
    build_s = time.time() - t
    trace = out / "trace.jsonl"
    (out / "spv-dump").mkdir(exist_ok=True)
    env = dict(os.environ, COVENANT_TRACE=str(trace), COVENANT_SPIRV_DIR=str(out / "spv-dump"))
    io = out / "io.jsonl"
    io.unlink(missing_ok=True)
    if (DRIVER / "libcovio.so").exists():       # image-I/O interposer (model_driver/covio.c)
        env.update(LD_PRELOAD=str(DRIVER / "libcovio.so"), COVENANT_IO_TRACE=str(io))
    sh([str(exe), str(out)], env=env, timeout=120)
    return out, modules, trace, build_s


def check_variant(case, fixed):
    out, modules, trace, build_s = build_variant(case, fixed)
    t0 = time.time()
    model = load_trace(trace, spirv_dir=out / "spv-dump")
    r = Report()

    srcs = sorted(case.glob("*.cpp")) + sorted(case.glob("*.hpp"))
    io = out / "io.jsonl"
    check_program(model, modules, srcs, ["-DFIXED"] if fixed else [], r,
                  io_trace=io if io.exists() else None)
    return r, {"build_s": round(build_s, 2), "check_s": round(time.time() - t0, 2),
               "trace_records": len(model.records), "draws": len(model.draws)}


def run_case(case):
    meta = json.loads((case / "case.json").read_text())
    res = {"case": case.name, **meta}
    try:
        rb, sb = check_variant(case, False)
        rf, sf = check_variant(case, True)
    except Exception as e:                      # build/run failure is a result too
        res.update(status="error", error=str(e)[-2000:])
        return res
    expect = set(meta.get("expect", []))
    got_b = set(rb.codes())
    res.update(bug_codes=sorted(got_b), fixed_codes=sorted(set(rf.codes())),
               bug_report=str(rb), fixed_report=str(rf), stats={"bug": sb, "fixed": sf},
               bug_severity=sorted({d.severity for d in rb.items}))
    detected = bool(expect) and expect <= got_b
    clean = rf.ok
    if meta.get("known_miss"):
        res["status"] = "miss (expected)" if not got_b else "miss (expected), noisy"
    elif detected and clean:
        res["status"] = "detected"
    elif detected:
        res["status"] = "detected, fixed variant not clean"
    else:
        res["status"] = "missed"
    return res


def markdown(results):
    lines = ["| # | Case | Category | Expected | Buggy variant reports | Fixed variant | Status |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in sorted(results, key=lambda r: r.get("id", 0)):
        fixed = "clean" if not r.get("fixed_codes") else ", ".join(r["fixed_codes"])
        lines.append(f"| {r.get('id')} | {r.get('title', '').replace('|', '&#124;')} | {r.get('category')} | "
                     f"{', '.join(r.get('expect', [])) or '(none: known miss)'} | "
                     f"{', '.join(r.get('bug_codes', [])) or '-'} | {fixed if 'fixed_codes' in r else '-'} | "
                     f"{r['status']} |")
    return "\n".join(lines)


def main(argv):
    sel = [a for a in argv if not a.startswith("--")]
    out_json = argv[argv.index("--json") + 1] if "--json" in argv else None
    out_md = argv[argv.index("--md") + 1] if "--md" in argv else None
    sel = [s for s in sel if s not in (out_json, out_md)]
    cases = [c for c in sorted(CASES.iterdir()) if (c / "case.json").exists()
             and (not sel or any(s in c.name for s in sel))]
    results = []
    for c in cases:
        r = run_case(c)
        print(f"{c.name:40s} {r['status']}", flush=True)
        if r["status"] not in ("detected", "miss (expected)"):
            print("   bug:  ", r.get("bug_codes"), "\n   fixed:", r.get("fixed_codes"),
                  "\n  ", r.get("error", "")[-800:])
        results.append(r)
    if out_json:
        Path(out_json).write_text(json.dumps(results, indent=1, default=str))
    if out_md:
        Path(out_md).write_text(markdown(results) + "\n")
    return results


if __name__ == "__main__":
    main(sys.argv[1:])
