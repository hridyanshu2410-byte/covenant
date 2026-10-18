"""End-to-end check of proof-licensed barrier elision on Mesa lavapipe (CPU Vulkan).

    python3 bench/elide/run_lavapipe.py [--work DIR] [--timing-runs 5] [--timing-frames 10]

For each workload (particles, deferred, postchain):
 1. build it, and the VK_LAYER_COVENANT_elide layer, into --work;
 2. run it on the COVENANT model driver (no GPU) to record a trace;
 3. run covenant.elide on the trace -> plans/<w>.plan.json (the proved plan) and
    plans/<w>.naive.json (UNSAFE control: per-barrier redundancy, no joint check);
 4. run it on lavapipe with the Khronos validation layer + synchronization
    validation, (a) without the layer, (b) with the layer loaded but disabled,
    (c) with the plan (strict matching, the traced frame count), (d) with the
    plan in chain matching for more frames than traced, (e) with the naive
    control plan; compare the readback hashes and count syncval hazards;
 5. time (a) vs (c)/(d) without validation. These are CPU-driver timings: they say
    nothing about GPU speed.

Writes results/lavapipe.json and results/lavapipe.md next to this file.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORKLOADS = ["particles", "deferred", "postchain"]
LVP = "/usr/share/vulkan/icd.d/lvp_icd.json"


def sh(cmd, env=None, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} failed:\n{r.stdout}\n{r.stderr}")
    return r


def run(work, w, frames, layer=None, mode="strict", validation=False, disable=False, extra=()):
    """layer: plan path (or "" for loaded-but-disabled), None for no layer."""
    env = dict(os.environ, VK_ICD_FILENAMES=LVP)
    env.pop("VK_INSTANCE_LAYERS", None)
    out = Path(work) / "runs" / f"{w}-{os.getpid()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = out.with_suffix(".stats.json")
    if stats.exists():
        stats.unlink()
    if layer is not None:
        env["VK_LAYER_PATH"] = f"{work}/layer:/usr/share/vulkan/explicit_layer.d"
        env["VK_INSTANCE_LAYERS"] = "VK_LAYER_COVENANT_elide"
        env["COVENANT_ELIDE_PLAN"] = layer
        env["COVENANT_ELIDE_MODE"] = mode
        env["COVENANT_ELIDE_STATS"] = str(stats)
        env["COVENANT_ELIDE_QUIET"] = "1"
        if disable:
            env["COVENANT_ELIDE_DISABLE"] = "1"
    if validation:
        env["ELIDE_VALIDATION"] = "1"
    r = sh([f"{work}/bin/{w}", "--frames", str(frames), "--json", str(out), "--quiet", *extra], env=env)
    res = json.loads(out.read_text())
    res["layer_stats"] = json.loads(stats.read_text()) if stats.exists() else None
    res["stderr_hazard_lines"] = [l for l in r.stderr.splitlines() if "SYNC-" in l][:5]
    return res


def scale_invariance(work, w, a, driver, pj):
    """The GPU harness runs the scale-1 plan at other --scale values. Trace and
    prove each of those scales on the model driver too, and check that the
    proved plan is the same: same barrier calls (rec, idx, sig, chain) with the
    same decisions and masks, and the same chain entries."""
    def key(p):
        strip = lambda b: {k: v for k, v in b.items() if k not in ("why", "not_removable_because", "seq", "cb")}
        return json.dumps([[strip(b) for b in p["barriers"]], [strip(b) for b in p["chain"]]], sort_keys=True)
    out = {}
    for sc in [x for x in a.scales.split(",") if x]:
        d = Path(work) / "scales" / f"{w}-{sc}"
        (d / "spv").mkdir(parents=True, exist_ok=True)
        t = d / "trace.jsonl"
        env = dict(os.environ, COVENANT_TRACE=str(t), COVENANT_SPIRV_DIR=str(d / "spv"), LD_LIBRARY_PATH=str(driver))
        sh([f"{work}/bin/{w}", "--frames", str(a.trace_frames), "--scale", sc, "--quiet"], env=env)
        sh([sys.executable, "-m", "covenant.elide", str(t), "--spirv-dir", str(d / "spv"), "-o", str(d / "plan.json"),
            "-q"], env=dict(os.environ, PYTHONPATH=str(ROOT)))
        out[sc] = "same plan" if key(json.loads((d / "plan.json").read_text())) == key(pj) else "DIFFERENT plan"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(ROOT / "build" / "elide"))
    ap.add_argument("--trace-frames", type=int, default=3)
    ap.add_argument("--chain-frames", type=int, default=8)
    ap.add_argument("--timing-runs", type=int, default=5)
    ap.add_argument("--timing-frames", type=int, default=10)
    ap.add_argument("--scales", default="4,16", help="other --scale values to trace and prove (model driver)")
    ap.add_argument("--lavapipe-scale", type=float, default=4, help="--scale for the scaled lavapipe check")
    a = ap.parse_args()
    work = Path(a.work)
    sh(["sh", str(HERE / "build.sh"), str(work / "bin")])
    sh(["sh", str(ROOT / "elide_layer" / "build.sh"), str(work / "layer")])
    sh([sys.executable, str(HERE / "gen_shaders.py")]) if not (HERE / "shaders_spv.h").exists() else None
    plans = HERE / "plans"
    plans.mkdir(exist_ok=True)
    driver = ROOT / "model_driver" / "build"
    results = {}
    for w in WORKLOADS:
        print(f"== {w}", flush=True)
        tdir = HERE / "traces"            # small; kept so the plans can be re-derived without building
        (tdir / f"{w}-spv").mkdir(parents=True, exist_ok=True)
        trace = tdir / f"{w}.jsonl"
        env = dict(os.environ, COVENANT_TRACE=str(trace), COVENANT_SPIRV_DIR=str(tdir / f"{w}-spv"),
                   LD_LIBRARY_PATH=str(driver))
        sh([f"{work}/bin/{w}", "--frames", str(a.trace_frames), "--quiet"], env=env)
        plan, naive = plans / f"{w}.plan.json", plans / f"{w}.naive.json"
        sh([sys.executable, "-m", "covenant.elide", str(trace), "--spirv-dir", str(tdir / f"{w}-spv"),
            "-o", str(plan), "--naive-control", str(naive), "-q"], env=dict(os.environ, PYTHONPATH=str(ROOT)))
        pj = json.loads(plan.read_text())
        nj = json.loads(naive.read_text())
        r = {"plan_summary": pj["summary"], "proof": {k: pj["proof"][k] for k in
                                                      ("baseline_diagnostics", "new_diagnostics_with_plan",
                                                       "plain_timing_check", "oracle_runs", "seconds")},
             "naive_control": {"calls_removed": sum(1 for b in nj["barriers"] if b["decision"] == "remove"),
                               "checker_hazards": nj["proof"]["hazards_created"]}}
        r["scale_invariance"] = scale_invariance(work, w, a, driver, pj)
        F, CF = a.trace_frames, a.chain_frames
        r["base"] = run(work, w, F, validation=True)
        r["disabled"] = run(work, w, F, layer=str(plan), validation=True, disable=True)
        r["strict"] = run(work, w, F, layer=str(plan), mode="strict", validation=True)
        r["base_long"] = run(work, w, CF, validation=True)
        r["chain_long"] = run(work, w, CF, layer=str(plan), mode="chain", validation=True)
        r["naive"] = run(work, w, F, layer=str(naive), mode="strict", validation=True)
        S = a.lavapipe_scale
        r["base_scaled"] = run(work, w, F, validation=True, extra=("--scale", str(S)))
        r["chain_scaled"] = run(work, w, F, layer=str(plan), mode="chain", validation=True, extra=("--scale", str(S)))
        # timing (no validation), interleaved
        tb, tp = [], []
        for _ in range(a.timing_runs):
            tb.append(run(work, w, a.timing_frames)["gpu_ms_median"])
            tp.append(run(work, w, a.timing_frames, layer=str(plan), mode="chain")["gpu_ms_median"])
        r["timing"] = {"frames": a.timing_frames, "runs": a.timing_runs, "base_ms": tb, "plan_ms": tp,
                       "base_median": statistics.median(tb), "plan_median": statistics.median(tp)}
        print(f"  scale invariance: {r['scale_invariance']}", flush=True)
        for k in ("base", "disabled", "strict", "base_long", "chain_long", "naive", "base_scaled", "chain_scaled"):
            print(f"  {k:<10} hashes {r[k]['hashes']} sync_hazards {r[k]['sync_hazards']} "
                  f"validation {r[k]['validation_messages']} layer {r[k]['layer_stats']}", flush=True)
        results[w] = r
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "lavapipe.json").write_text(json.dumps(results, indent=1))
    (out / "lavapipe.md").write_text(markdown(results, a))
    print((out / "lavapipe.md").read_text())


def markdown(results, a):
    L = ["# Barrier elision on lavapipe (CPU Vulkan): correctness", "",
         "Generated by `bench/elide/run_lavapipe.py`. lavapipe executes on the CPU; the timings below say",
         "nothing about GPU speed and are recorded only for completeness. Correctness is the point:",
         "the readback hash of every output must not change, and the Khronos synchronization validation",
         "(syncval) must report no hazard with the plan applied.", "",
         "| workload | barrier calls | removed | narrowed | required | structs removed / narrowed / kept | "
         "hash same (strict, chain) | syncval hazards: base / plan strict / plan chain | naive control: checker / syncval |",
         "|---|---|---|---|---|---|---|---|---|"]
    for w, r in results.items():
        s = r["plan_summary"]
        d = s["decisions"]
        st = s["structs"]
        same_s = r["strict"]["hashes"] == r["base"]["hashes"] == r["disabled"]["hashes"]
        same_c = r["chain_long"]["hashes"] == r["base_long"]["hashes"]
        L.append(f"| {w} | {s['calls']} | {d.get('remove', 0)} | {d.get('narrow', 0)} | {d.get('required', 0)} | "
                 f"{st['removed']} / {st['narrowed']} / {st['kept']} | {'yes' if same_s else 'NO'}, "
                 f"{'yes' if same_c else 'NO'} | {r['base']['sync_hazards']} / {r['strict']['sync_hazards']} / "
                 f"{r['chain_long']['sync_hazards']} | {r['naive_control']['checker_hazards']} / "
                 f"{r['naive']['sync_hazards']} |")
    L += ["", "| workload | plan proved at other scales (model driver) | lavapipe at scale "
          f"{a.lavapipe_scale:g}: hash same / syncval base / syncval plan (chain) |", "|---|---|---|"]
    for w, r in results.items():
        si = ", ".join(f"x{k}: {v}" for k, v in r["scale_invariance"].items())
        same = r["chain_scaled"]["hashes"] == r["base_scaled"]["hashes"]
        L.append(f"| {w} | {si} | {'yes' if same else 'NO'} / {r['base_scaled']['sync_hazards']} / "
                 f"{r['chain_scaled']['sync_hazards']} |")
    L += ["", f"Strict = matching by (recording ordinal, barrier index), {a.trace_frames} frames (the traced run). "
          f"Chain = matching by barrier-sequence hash, {a.chain_frames} frames (more than were traced).",
          "Naive control = every barrier that is redundant *on its own*, all removed together without a joint "
          "re-check (an UNSAFE plan, run only to show that per-barrier reasoning is not enough).", "",
          "| workload | layer matched / sig mismatches (strict) | layer matched / sig mismatches (chain, long) | "
          "all validation messages: base / plan strict / plan chain | "
          "lavapipe frame ms, median of runs: no layer / plan (chain) |",
          "|---|---|---|---|---|"]
    for w, r in results.items():
        ls = r["strict"]["layer_stats"] or {}
        t = r["timing"]
        lc = r["chain_long"]["layer_stats"] or {}
        L.append(f"| {w} | {ls.get('matched')} / {ls.get('sig_mismatch')} | {lc.get('matched')} / {lc.get('sig_mismatch')} | "
                 f"{r['base']['validation_messages']} / {r['strict']['validation_messages']} / "
                 f"{r['chain_long']['validation_messages']} | {t['base_median']:.2f} / {t['plan_median']:.2f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
