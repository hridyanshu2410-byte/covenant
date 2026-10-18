"""Measure proof-licensed barrier elision on a real GPU.

Run this ON THE GPU MACHINE, after building the workloads and the layer
(see README.md). It never generates plans: the plans in plans/ were proved in
advance (no GPU needed) from model-driver traces and are applied as they are.

    python3 measure.py --bin BUILD_DIR --layer LAYER_DIR [--runs 15] [--frames 300]
                       [--scales 1,4,16] [--device "<name substring>"] [--validation]

For every workload and scale it runs three configurations, interleaved in a
shuffled order each round so clock/thermal drift hits all of them alike:

  base      the workload as written (no layer)
  overhead  VK_LAYER_COVENANT_elide loaded with the plan DISABLED (cost of the layer itself)
  plan      the layer applying plans/<workload>.plan.json (chain matching, so it covers
            every frame however many are run)

Per run, the GPU time of each frame comes from two timestamp queries
(TOP_OF_PIPE .. BOTTOM_OF_PIPE around the frame's command buffer); the first
--warmup frames are dropped and the run's median is kept. The report gives the
median over runs, a bootstrap 95% interval for the speedup, the median CPU
submit-to-fence time, and checks that every configuration produced the same
readback hashes and that the layer matched the plan (no signature mismatch).

Outputs results/gpu-<host>-<time>.json and .md next to this file.
"""
import argparse
import json
import os
import platform
import random
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKLOADS = ["particles", "deferred", "postchain"]
EXE = ".exe" if os.name == "nt" else ""


def find_exe(bin_dir, w):
    for cand in (Path(bin_dir) / (w + EXE), Path(bin_dir) / "Release" / (w + EXE)):
        if cand.exists():
            return cand
    sys.exit(f"cannot find {w}{EXE} in {bin_dir} (build first, see README.md)")


def run_once(exe, w, scale, frames, mode, args, tmp, validation=False):
    env = dict(os.environ)
    env.pop("VK_INSTANCE_LAYERS", None)
    for k in list(env):
        if k.startswith("COVENANT_ELIDE"):
            env.pop(k)
    out = tmp / f"{w}-{mode}.json"
    stats = tmp / f"{w}-{mode}.stats.json"
    for p in (out, stats):
        if p.exists():
            p.unlink()
    if mode in ("overhead", "plan"):
        # VK_ADD_LAYER_PATH (loader >= 1.3.234) adds to the default search path, so the
        # Khronos validation layer stays visible; VK_LAYER_PATH would replace it.
        prev = env.get("VK_ADD_LAYER_PATH")
        env["VK_ADD_LAYER_PATH"] = str(Path(args.layer).resolve()) + (os.pathsep + prev if prev else "")
        env["VK_INSTANCE_LAYERS"] = "VK_LAYER_COVENANT_elide"
        env["COVENANT_ELIDE_PLAN"] = str((Path(args.plans) / f"{w}.plan.json").resolve())
        env["COVENANT_ELIDE_MODE"] = "chain"
        env["COVENANT_ELIDE_STATS"] = str(stats)
        env["COVENANT_ELIDE_QUIET"] = "1"
        if mode == "overhead":
            env["COVENANT_ELIDE_DISABLE"] = "1"
    if args.device:
        env["ELIDE_DEVICE"] = args.device
    if validation:
        env["ELIDE_VALIDATION"] = "1"
    cmd = [str(exe), "--frames", str(frames), "--scale", str(scale), "--json", str(out), "--quiet"]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=(45 if validation else args.timeout))
    except subprocess.TimeoutExpired as e:
        # Windows/MinGW: under the Khronos validation layer the workload can hang at
        # process exit (it never destroys the instance) after it has written its JSON. Accept the
        # JSON it wrote; any other timeout is still an error.
        if not (validation and out.exists()):
            raise
        r = subprocess.CompletedProcess(cmd, 0, e.stdout or "", e.stderr or "")
        if isinstance(r.stderr, bytes):
            r.stderr = r.stderr.decode(errors="replace")
        r.stderr += "\n[measure] process hung at exit under validation; killed after its JSON was written"
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"{' '.join(cmd)} [{mode}] failed ({r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    res = json.loads(out.read_text())
    res["layer_stats"] = json.loads(stats.read_text()) if stats.exists() else None
    if validation:
        res["validation_lines"] = [l for l in r.stderr.splitlines() if l.startswith("[validation]")][:20]
    return res


def bootstrap_speedup(base, plan, n=4000, seed=1):
    rnd = random.Random(seed)
    vals = []
    for _ in range(n):
        b = statistics.median(rnd.choice(base) for _ in base)
        p = statistics.median(rnd.choice(plan) for _ in plan)
        vals.append((b - p) / b if b else 0.0)
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bin", required=True, help="directory with particles, deferred, postchain executables")
    ap.add_argument("--layer", required=True, help="directory with the layer library and VkLayer_COVENANT_elide.json")
    ap.add_argument("--plans", default=str(HERE / "plans"))
    ap.add_argument("--workloads", default=",".join(WORKLOADS))
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--scales", default="1,4,16")
    ap.add_argument("--device", help="substring of the GPU name to use (default: first discrete GPU)")
    ap.add_argument("--validation", action="store_true",
                    help="also run each configuration once with the Khronos validation layer + syncval")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    if args.warmup >= args.frames:
        sys.exit("--warmup must be smaller than --frames")
    tmp = HERE / "results" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(args.seed)
    modes = ["base", "overhead", "plan"]
    report = {"host": socket.gethostname(), "platform": platform.platform(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
              "args": vars(args), "results": []}
    for w in args.workloads.split(","):
        exe = find_exe(args.bin, w)
        plan = json.loads((Path(args.plans) / f"{w}.plan.json").read_text())
        for scale in [float(s) for s in args.scales.split(",")]:
            print(f"== {w} scale {scale}", flush=True)
            per = {m: {"gpu": [], "cpu": [], "hashes": set(), "stats": []} for m in modes}
            device = None
            for k in range(args.runs):
                order = modes[:]
                rnd.shuffle(order)
                for m in order:
                    res = run_once(exe, w, scale, args.frames, m, args, tmp)
                    device = res["device"]
                    g = res["gpu_ms"][args.warmup:]
                    c = res["cpu_ms"][args.warmup:]
                    if g:
                        per[m]["gpu"].append(statistics.median(g))
                    per[m]["cpu"].append(statistics.median(c))
                    per[m]["hashes"].add(json.dumps(res["hashes"], sort_keys=True))
                    if res["layer_stats"]:
                        per[m]["stats"].append(res["layer_stats"])
                print(f"  round {k + 1}/{args.runs}: " + ", ".join(
                    f"{m} {per[m]['gpu'][-1]:.3f} ms" if per[m]["gpu"] else f"{m} (no timestamps)" for m in modes), flush=True)
            entry = {"workload": w, "scale": scale, "device": device, "frames": args.frames, "warmup": args.warmup,
                     "runs": args.runs, "plan_summary": plan["summary"]}
            for m in modes:
                entry[m] = {"gpu_ms_per_run": per[m]["gpu"], "cpu_ms_per_run": per[m]["cpu"],
                            "gpu_ms_median": statistics.median(per[m]["gpu"]) if per[m]["gpu"] else None,
                            "cpu_ms_median": statistics.median(per[m]["cpu"])}
            all_hashes = set().union(*(per[m]["hashes"] for m in modes))
            entry["hashes_identical"] = len(all_hashes) == 1
            entry["hashes"] = sorted(all_hashes)
            st = per["plan"]["stats"]
            entry["layer_applied"] = bool(st) and all(s["matched"] > 0 and s["sig_mismatch"] == 0 for s in st)
            entry["layer_stats_example"] = st[0] if st else None
            b, p, o = entry["base"]["gpu_ms_median"], entry["plan"]["gpu_ms_median"], entry["overhead"]["gpu_ms_median"]
            if b and p:
                entry["gpu_speedup"] = (b - p) / b
                entry["gpu_speedup_ci95"] = bootstrap_speedup(per["base"]["gpu"], per["plan"]["gpu"])
                entry["gpu_speedup_vs_overhead"] = (o - p) / o if o else None
            if args.validation:
                entry["validation"] = {}
                for m in modes:
                    res = run_once(exe, w, scale, min(args.frames, 5), m, args, tmp, validation=True)
                    entry["validation"][m] = {"messages": res["validation_messages"], "sync_hazards": res["sync_hazards"],
                                              "first": res.get("validation_lines", [])[:3]}
            report["results"].append(entry)
            if not entry["hashes_identical"]:
                print("  !! readback hashes differ between configurations", flush=True)
            if not entry["layer_applied"]:
                print("  !! the layer did not apply the plan (see layer_stats_example)", flush=True)
            if entry.get("gpu_speedup") is not None:
                lo, hi = entry["gpu_speedup_ci95"]
                print(f"  GPU median: base {b:.3f} ms, layer-off {o:.3f} ms, plan {p:.3f} ms -> "
                      f"{100 * entry['gpu_speedup']:.1f}% faster (95% CI {100 * lo:.1f}..{100 * hi:.1f}%)", flush=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = HERE / "results" / f"gpu-{socket.gethostname()}-{stamp}"
    out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    out.with_suffix(".md").write_text(markdown(report))
    print(markdown(report))
    print(f"wrote {out}.json and {out}.md")


def markdown(rep):
    L = [f"# Barrier elision on {rep['results'][0]['device'] if rep['results'] else '?'}", "",
         f"{rep['platform']}, {rep['time']}. {rep['args']['runs']} runs x {rep['args']['frames']} frames "
         f"(first {rep['args']['warmup']} dropped); GPU time from timestamp queries, median per run, median over runs.", "",
         "| workload | scale | calls removed / narrowed / kept | base ms | layer, plan off ms | plan ms | GPU speedup (95% CI) | hashes identical | plan applied |",
         "|---|---|---|---|---|---|---|---|---|"]
    for e in rep["results"]:
        d = e["plan_summary"]["decisions"]
        f = lambda x: f"{x:.3f}" if x is not None else "n/a"
        sp = (f"{100 * e['gpu_speedup']:.1f}% ({100 * e['gpu_speedup_ci95'][0]:.1f}..{100 * e['gpu_speedup_ci95'][1]:.1f})"
              if e.get("gpu_speedup") is not None else "n/a")
        L.append(f"| {e['workload']} | {e['scale']:g} | {d.get('remove', 0)} / {d.get('narrow', 0)} / {d.get('required', 0)} | "
                 f"{f(e['base']['gpu_ms_median'])} | {f(e['overhead']['gpu_ms_median'])} | {f(e['plan']['gpu_ms_median'])} | "
                 f"{sp} | {'yes' if e['hashes_identical'] else 'NO'} | {'yes' if e['layer_applied'] else 'NO'} |")
    if any("validation" in e for e in rep["results"]):
        L += ["", "| workload | scale | syncval hazards base / overhead / plan |", "|---|---|---|"]
        for e in rep["results"]:
            v = e.get("validation")
            if v:
                L.append(f"| {e['workload']} | {e['scale']:g} | {v['base']['sync_hazards']} / "
                         f"{v['overhead']['sync_hazards']} / {v['plan']['sync_hazards']} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
