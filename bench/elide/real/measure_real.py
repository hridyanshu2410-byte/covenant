"""GPU frame-time measurement of real samples with and without an elision plan.

    python measure_real.py --sw-bin <Sascha bin dir> --ks-root <Vulkan-Samples checkout> \
        --ks-bin <vulkan_samples.exe> --layers <dir with both layer manifests> [options]

Platform-neutral (Windows 11 with an NVIDIA GPU is the intended target; see ../REAL-SAMPLES.md). For each
sample it runs the UNMODIFIED program in its benchmark mode, interleaving three configurations
round-robin (the order rotates every round so drift hits all three equally):

  base   VK_LAYER_COVENANT_probe only
  off    VK_LAYER_COVENANT_elide loaded with the plan but COVENANT_ELIDE_DISABLE=1, + probe
         (cost of having the layer in the chain)
  plan   VK_LAYER_COVENANT_elide applying the plan (chain keys by default), + probe

VK_LAYER_COVENANT_probe wraps every submitted batch in TOP_OF_PIPE / BOTTOM_OF_PIPE timestamp
writes (COVENANT_PROBE_TIMESTAMPS). Per present-interval ("frame") it computes
  gpu_busy_ms = sum over batches of (end - start)     (queue-time of the frame's work)
  gpu_span_ms = max(end) - min(start)                 (wall span incl. overlap/gaps)
and per run the median over frames after --warmup-frames. Sascha samples additionally write
their own per-frame CPU times (-b -bf <csv> -bt). Khronos --benchmark fixes the SIMULATED
frame rate at 60 fps, so its printed fps is not a measurement and is ignored.

Correctness runs (once per config, before timing, unless --no-hash): COVENANT_PROBE_HASH hashes
every presented image; base / off / plan must agree for samples whose output is a function of
the frame index (all samples here: Sascha -b freezes the animation timer; Khronos --benchmark
uses a fixed 60 fps timestep). --validation additionally loads VK_LAYER_KHRONOS_validation
with synchronization validation (needs the Vulkan SDK) and counts SYNC-* messages per config.

Writes <out>/measure_real.json and prints a table: per sample, median gpu_busy and the
plan/base and off/base ratios with a bootstrap 95% CI over runs, plus the layer's own
statistics (matched calls, signature mismatches, calls removed / narrowed). If the layer
removed and narrowed nothing in the steady state, the row says so: any difference is then
noise or layer overhead, never an elision speedup.
"""
import argparse
import csv
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# name -> (app, id). Plans: plans/<app>_<id>.f2.plan.json (2-family trace: an NVIDIA GPU
# exposes a dedicated compute family, like the model driver's default) unless --family 1.
SAMPLES = {
    "sw/computecloth": ("sw", "computecloth"),
    "sw/computenbody": ("sw", "computenbody"),
    "sw/bloom": ("sw", "bloom"),
    "ks/async_compute": ("ks", "async_compute"),
    "ks/afbc": ("ks", "afbc"),
    "ks/pipeline_barriers": ("ks", "pipeline_barriers"),
    "ks/oit_depth_peeling": ("ks", "oit_depth_peeling"),
    "ks/compute_nbody": ("ks", "compute_nbody"),
}
EXE = ".exe" if os.name == "nt" else ""


def app_cmd(a, app, sid, frames, bench_csv=None):
    if app == "sw":
        cmd = [str(Path(a.sw_bin) / (sid + EXE)), "-b", "-bw", "0", "-bfs", str(frames)]
        if bench_csv:
            cmd += ["-bf", str(bench_csv), "-bt"]
        return cmd, None
    # --vsync OFF: otherwise Khronos samples may pick FIFO and GPU work waits on the display
    return [a.ks_bin, "sample", sid, "--benchmark", "--hideui", "--vsync", "OFF", "--stop-after-frame", str(frames)], a.ks_root


def run_one(a, name, config, plan, outdir, frames, hash_run=False, validation=False):
    app, sid = SAMPLES[name]
    d = Path(outdir)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("COVENANT_") or k in ("VK_INSTANCE_LAYERS", "VK_LOADER_LAYERS_ENABLE"):
            env.pop(k)
    layers, paths = [], [a.layers]
    if config in ("off", "plan"):
        layers.append("VK_LAYER_COVENANT_elide")
        env.update(COVENANT_ELIDE_PLAN=str(plan), COVENANT_ELIDE_MODE=a.mode,
                   COVENANT_ELIDE_STATS=str(d / "elide.json"), COVENANT_ELIDE_QUIET="1")
        if config == "off":
            env["COVENANT_ELIDE_DISABLE"] = "1"
    if validation:
        layers.append("VK_LAYER_KHRONOS_validation")
        (d / "vk_layer_settings.txt").write_text("\n".join([
            "khronos_validation.debug_action = VK_DBG_LAYER_ACTION_LOG_MSG",
            "khronos_validation.report_flags = error,warn",
            f"khronos_validation.log_filename = {d / 'vvl.txt'}",
            "khronos_validation.validate_sync = true",
            "khronos_validation.syncval_shader_accesses_heuristic = true",
            "khronos_validation.enable_message_limit = false", ""]))
        env["VK_LAYER_SETTINGS_PATH"] = str(d / "vk_layer_settings.txt")
    layers.append("VK_LAYER_COVENANT_probe")
    env["COVENANT_PROBE_QUIET"] = "1"
    if hash_run:
        env["COVENANT_PROBE_HASH"] = str(d / "hashes.txt")
    else:
        env["COVENANT_PROBE_TIMESTAMPS"] = str(d / "ts.json")
    sep = ";" if os.name == "nt" else ":"
    old = env.get("VK_ADD_LAYER_PATH")
    env["VK_ADD_LAYER_PATH"] = sep.join(paths + ([old] if old else []))
    env["VK_INSTANCE_LAYERS"] = sep.join(layers)
    cmd, cwd = app_cmd(a, app, sid, frames, None if hash_run or validation else d / "bench.csv")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, timeout=a.timeout)
        rc, out = r.returncode, (r.stdout or "")[-8000:] + (r.stderr or "")[-8000:]
    except subprocess.TimeoutExpired as e:
        rc, out = "timeout", str(e)
    (d / "out.txt").write_text(out, errors="replace")
    res = {"rc": rc, "seconds": round(time.time() - t0, 2)}
    if (d / "elide.json").exists():
        res["layer"] = json.loads((d / "elide.json").read_text())
    if hash_run:
        res["hashes"] = [l.split()[-1] for l in open(d / "hashes.txt")] if (d / "hashes.txt").exists() else []
    if validation:
        txt = (d / "vvl.txt").read_text(errors="replace") if (d / "vvl.txt").exists() else ""
        res["vvl_total"] = len(re.findall(r"^Validation (?:Error|Warning)", txt, re.M))
        res["vvl_sync"] = len(re.findall(r"^Validation (?:Error|Warning): \[ SYNC-", txt, re.M))
    if (d / "ts.json").exists():
        res.update(frame_times(json.loads((d / "ts.json").read_text()), a.warmup_frames))
    if (d / "bench.csv").exists():
        rows = list(csv.reader(open(d / "bench.csv")))
        ms = [float(r[1]) for r in rows if len(r) == 2 and re.match(r"^\d+$", r[0])]
        if len(ms) > a.warmup_frames:
            res["cpu_frame_ms_median"] = statistics.median(ms[a.warmup_frames:])
    return res


def frame_times(ts, warmup):
    per = ts["timestamp_period_ns"] / 1e6
    frames = {}
    for f, q, t0, t1 in ts["batches"]:
        if t1 < t0:
            continue
        frames.setdefault(f, []).append((t0, t1))
    keys = sorted(frames)[warmup:-1]          # drop warm-up and the (partial) last interval
    busy = [sum(t1 - t0 for t0, t1 in frames[k]) * per for k in keys]
    span = [(max(t1 for _, t1 in frames[k]) - min(t0 for t0, _ in frames[k])) * per for k in keys]
    if not busy:
        return {"frames_timed": 0}
    return {"frames_timed": len(busy), "gpu_busy_ms": statistics.median(busy), "gpu_span_ms": statistics.median(span),
            "device": ts.get("device")}


def boot_ratio(xs, ys, n=2000, seed=1):
    """median(xs)/median(ys) with a bootstrap 95% CI over runs."""
    rnd = random.Random(seed)
    if not xs or not ys:
        return None
    rs = sorted(statistics.median([rnd.choice(xs) for _ in xs]) / statistics.median([rnd.choice(ys) for _ in ys])
                for _ in range(n))
    return statistics.median(xs) / statistics.median(ys), rs[int(0.025 * n)], rs[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default=",".join(SAMPLES))
    ap.add_argument("--sw-bin", help="Sascha build bin dir (e.g. ...\\Vulkan\\build\\bin\\Release)")
    ap.add_argument("--ks-root", help="Vulkan-Samples checkout (the samples' working directory)")
    ap.add_argument("--ks-bin", help="vulkan_samples executable")
    ap.add_argument("--layers", required=True, help="dir holding VkLayer_COVENANT_elide.json and VkLayer_COVENANT_probe.json")
    ap.add_argument("--plans", default=str(HERE / "plans"))
    ap.add_argument("--family", default="2", choices=("1", "2"), help="which traced plan to use (default 2)")
    ap.add_argument("--mode", default="chain", choices=("chain", "strict"))
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--warmup-frames", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--no-hash", action="store_true")
    ap.add_argument("--hash-frames", type=int, default=60)
    ap.add_argument("--validation", action="store_true", help="also one syncval run per config (Vulkan SDK layer)")
    ap.add_argument("--out", default="measure_real_out")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    allres = {}
    for name in a.samples.split(","):
        app, sid = SAMPLES[name]
        if (app == "sw" and not a.sw_bin) or (app == "ks" and not (a.ks_bin and a.ks_root)):
            print(f"skip {name}: no {'--sw-bin' if app == 'sw' else '--ks-bin/--ks-root'}", flush=True)
            continue
        plan = Path(a.plans) / f"{app}_{sid}.f{a.family}.plan.json"
        if not plan.exists():
            print(f"skip {name}: no plan {plan}", flush=True)
            continue
        R = allres[name] = {"plan": str(plan), "mode": a.mode, "check": {}, "runs": {"base": [], "off": [], "plan": []}}
        base_dir = out / name.replace("/", "_")
        if not a.no_hash:
            for c in ("base", "off", "plan"):
                R["check"][c] = run_one(a, name, c, plan, base_dir / f"hash_{c}", a.hash_frames, hash_run=True)
            h = {c: R["check"][c].get("hashes") for c in R["check"]}
            R["hashes_equal"] = bool(h["base"]) and h["base"] == h["off"] == h["plan"]
            print(name, "hashes equal:", R["hashes_equal"], {c: len(v or []) for c, v in h.items()}, flush=True)
        if a.validation:
            for c in ("base", "plan"):
                R["check"]["val_" + c] = run_one(a, name, c, plan, base_dir / f"val_{c}", a.hash_frames, validation=True)
            print(name, "syncval base/plan:", R["check"]["val_base"].get("vvl_sync"), R["check"]["val_plan"].get("vvl_sync"), flush=True)
        order = ["base", "off", "plan"]
        for k in range(a.rounds):
            for c in order[k % 3:] + order[:k % 3]:
                r = run_one(a, name, c, plan, base_dir / f"r{k}_{c}", a.frames)
                R["runs"][c].append(r)
                print(name, k, c, {x: r.get(x) for x in ("rc", "gpu_busy_ms", "gpu_span_ms", "cpu_frame_ms_median", "frames_timed")},
                      flush=True)
        busy = {c: [r["gpu_busy_ms"] for r in R["runs"][c] if r.get("gpu_busy_ms")] for c in R["runs"]}
        span = {c: [r["gpu_span_ms"] for r in R["runs"][c] if r.get("gpu_span_ms")] for c in R["runs"]}
        R["summary"] = {
            "gpu_busy_ms_median": {c: statistics.median(v) if v else None for c, v in busy.items()},
            "busy_plan_over_base": boot_ratio(busy["plan"], busy["base"]),
            "busy_off_over_base": boot_ratio(busy["off"], busy["base"]),
            "span_plan_over_base": boot_ratio(span["plan"], span["base"]),
        }
        lay = (R["runs"]["plan"][-1].get("layer") if R["runs"]["plan"] else None) or {}
        R["summary"]["layer"] = lay
        # steady state = recordings submitted after the first present (plan_report.py); the layer's
        # counters also include one-time setup barriers, so use the plan's own report
        rep = Path(str(plan).replace(".plan.json", ".report.json"))
        if rep.exists():
            st = json.loads(rep.read_text())["per_frame"]["structs"]
            R["summary"]["plan_changes_steady_state"] = bool(st.get("remove") or st.get("narrow"))
        else:
            R["summary"]["plan_changes_steady_state"] = bool(lay.get("removed_calls") or lay.get("narrowed_calls"))
        (out / "measure_real.json").write_text(json.dumps(allres, indent=1))
    print("\n| sample | GPU busy ms base / off / plan | plan/base [95% CI] | off/base [95% CI] | layer: matched, sig mismatch, removed, narrowed calls | hashes equal |")
    print("|---|---|---|---|---|---|")
    for name, R in allres.items():
        s = R.get("summary")
        if not s:
            continue
        m = s["gpu_busy_ms_median"]
        f = lambda r: "n/a" if not r else f"{r[0]:.4f} [{r[1]:.4f}, {r[2]:.4f}]"
        lay = s["layer"]
        note = "" if s["plan_changes_steady_state"] else " (plan changes nothing per frame: differences are noise)"
        print(f"| {name} | {m['base']:.4f} / {m['off']:.4f} / {m['plan']:.4f} | {f(s['busy_plan_over_base'])}{note} | "
              f"{f(s['busy_off_over_base'])} | {lay.get('matched')}, {lay.get('sig_mismatch')}, {lay.get('removed_calls')}, "
              f"{lay.get('narrowed_calls')} | {R.get('hashes_equal', 'n/a')} |")
    (out / "measure_real.json").write_text(json.dumps(allres, indent=1))


if __name__ == "__main__":
    sys.exit(main())
