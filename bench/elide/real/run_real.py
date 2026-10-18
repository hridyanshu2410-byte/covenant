"""Proof-licensed barrier elision on REAL, unmodified sample programs (Linux, no GPU needed).

    python3 bench/elide/real/run_real.py [--samples a,b] [--steps trace,prove,check,lavapipe] [--work DIR]

Samples: SaschaWillems/Vulkan @41a4410 and KhronosGroup/Vulkan-Samples @177edebf, built
unmodified (zero source changes). For each sample and each queue-family
layout (1 = one graphics+compute family, like lavapipe; 2 = an extra dedicated compute
family, like current NVIDIA GPUs and the model driver's default):

 trace     run it on the COVENANT model driver -> work/trace/<s>-f<F>/trace.jsonl. This step
           needs a driver build that honours COVVK_QUEUE_FAMILIES and COVVK_SURFACE_MIN_IMAGES,
           which bench/elide/real/driver-elide-real.diff adds to model_driver/covvk.cpp
           (apply it with `git apply`); the plans proved from those traces are in plans/.
 prove     python3 -m covenant.elide -> plans/<s>.f<F>.plan.json; plan_report.py summary
 check     refcheck.py: the plan against the reference semantics and the checker
           (epoch and pairwise) on the unstrengthened frame
 lavapipe  (1-family plan only; lavapipe has one queue family) run the sample on Mesa
           lavapipe with VK_LAYER_COVENANT_probe hashing every presented image and the
           Khronos validation layer (VVL HEAD, synchronization validation with the
           shader-access heuristic and loadOp-after-storeOp), in the configurations
             base       no elide layer
             disabled   elide layer loaded, plan disabled
             strict     plan, strict keys, the traced frame count
             chain      plan, chain keys, more frames than traced
             base_long  no elide layer, the chain frame count
           and compare hashes, syncval hazards and all validation messages.

lavapipe timings are never reported: lavapipe is a CPU driver.
Writes results/real.json and results/real.md next to this file.

External inputs (environment variables; defaults under <repo>/build/real-samples):
  COVENANT_SW_BIN      SaschaWillems/Vulkan build/bin directory
  COVENANT_KS_DIR      KhronosGroup/Vulkan-Samples checkout (built in build/)
  COVENANT_VVL_LAYERS  Khronos validation layer build (layers/ directory)
  COVENANT_VKINC       Vulkan-Headers include directory (default for --header)
"""
import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                        # repository root
LVP = "/usr/share/vulkan/icd.d/lvp_icd.json"
EXT = ROOT / "build" / "real-samples"
SW_BIN = os.environ.get("COVENANT_SW_BIN", str(EXT / "sw" / "build" / "bin"))
KS_DIR = os.environ.get("COVENANT_KS_DIR", str(EXT / "ks"))
KS_BIN = KS_DIR + "/build/app/bin/Release/x86_64/vulkan_samples"
VVL = os.environ.get("COVENANT_VVL_LAYERS", str(EXT / "vvl" / "build" / "layers"))   # Khronos validation, git HEAD 0ad67fd
VKINC = os.environ.get("COVENANT_VKINC", str(EXT / "vkh" / "include"))

# name -> (app, id, traced frames, chain frames)
SAMPLES = collections.OrderedDict([
    ("sw/computecloth", ("sw", "computecloth", 4, 10)),
    ("sw/computenbody", ("sw", "computenbody", 4, 10)),
    ("sw/bloom", ("sw", "bloom", 4, 10)),
    ("ks/async_compute", ("ks", "async_compute", 3, 6)),
    ("ks/afbc", ("ks", "afbc", 4, 10)),
    ("ks/pipeline_barriers", ("ks", "pipeline_barriers", 4, 10)),
    ("ks/oit_depth_peeling", ("ks", "oit_depth_peeling", 4, 10)),
    ("ks/compute_nbody", ("ks", "compute_nbody", 4, 10)),
])


def sh(cmd, env=None, cwd=None, timeout=3600, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} failed ({r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return r


def app_cmd(app, sid, frames):
    if app == "sw":
        return [f"{SW_BIN}/{sid}", "-b", "-bw", "0", "-bfs", str(frames)], None
    return [KS_BIN, "sample", sid, "--benchmark", "--hideui", "--stop-after-frame", str(frames)], KS_DIR


def slug(name):
    return name.replace("/", "_")


# ------------------------------------------------------------------ trace
def trace(a, name, fam):
    app, sid, frames, _ = SAMPLES[name]
    d = Path(a.work) / "trace" / f"{slug(name)}-f{fam}"
    if d.exists():
        shutil.rmtree(d)
    (d / "spv").mkdir(parents=True)
    cmd, cwd = app_cmd(app, sid, frames)
    env = dict(os.environ, COVENANT_TRACE=str(d / "trace.jsonl"), COVENANT_SPIRV_DIR=str(d / "spv"),
               LD_LIBRARY_PATH=a.driver, COVVK_QUEUE_FAMILIES=str(fam), COVVK_SURFACE_MIN_IMAGES="3",
               COVENANT_SURFACE_EXTENT="1280x720")
    env.pop("VK_ICD_FILENAMES", None)
    env.pop("VK_INSTANCE_LAYERS", None)
    t0 = time.time()
    r = sh(["xvfb-run", "-a", *cmd], env=env, cwd=cwd, check=False, timeout=1200)
    (d / "out.txt").write_text(r.stdout + r.stderr)
    n = sum(1 for _ in open(d / "trace.jsonl"))
    return {"rc": r.returncode, "records": n, "seconds": round(time.time() - t0, 1), "dir": str(d)}


# ------------------------------------------------------------------ prove
def prove(a, name, fam):
    d = Path(a.work) / "trace" / f"{slug(name)}-f{fam}"
    plans = HERE / "plans"
    plans.mkdir(exist_ok=True)
    plan = plans / f"{slug(name)}.f{fam}.plan.json"
    env = dict(os.environ, PYTHONPATH=a.covenant)
    t0 = time.time()
    r = sh([sys.executable, "-m", "covenant.elide", str(d / "trace.jsonl"), "--spirv-dir", str(d / "spv"),
            "--header", a.header, "-o", str(plan), "-q"], env=env, timeout=4 * 3600, check=False)
    if r.returncode != 0:
        return {"error": (r.stderr or r.stdout)[-2000:]}
    rep = sh([sys.executable, str(HERE / "plan_report.py"), str(plan), "--trace", str(d / "trace.jsonl"),
              "--json", str(plans / f"{slug(name)}.f{fam}.report.json")], env=env)
    out = json.loads((plans / f"{slug(name)}.f{fam}.report.json").read_text())
    out["seconds"] = round(time.time() - t0, 1)
    return out


def check(a, name, fam):
    d = Path(a.work) / "trace" / f"{slug(name)}-f{fam}"
    plan = HERE / "plans" / f"{slug(name)}.f{fam}.plan.json"
    out = HERE / "plans" / f"{slug(name)}.f{fam}.refcheck.json"
    env = dict(os.environ, PYTHONPATH=a.covenant)
    extra = ["--no-reference"] if a.no_reference else []
    r = sh([sys.executable, str(HERE / "refcheck.py"), str(d / "trace.jsonl"), str(d / "spv"), str(plan),
            "--header", a.header, "--json", str(out), *extra], env=env, timeout=6 * 3600, check=False)
    if not out.exists():
        return {"error": (r.stderr or r.stdout)[-2000:]}
    return json.loads(out.read_text())


# ------------------------------------------------------------------ lavapipe
def vvl_settings(d):
    p = d / "vk_layer_settings.txt"
    p.write_text("\n".join([
        "khronos_validation.debug_action = VK_DBG_LAYER_ACTION_LOG_MSG",
        "khronos_validation.report_flags = error,warn",
        f"khronos_validation.log_filename = {d / 'vvl.txt'}",
        "khronos_validation.validate_sync = true",
        "khronos_validation.syncval_shader_accesses_heuristic = true",
        "khronos_validation.syncval_load_op_after_store_op_validation = true",
        "khronos_validation.enable_message_limit = false",
        "khronos_validation.duplicate_message_limit = 100000", ""]))
    return p


def parse_vvl(path):
    try:
        txt = open(path, errors="replace").read()
    except FileNotFoundError:
        return {"total": 0, "sync": 0, "groups": {}}
    groups = collections.Counter()
    total = sync = 0
    for m in re.split(r"\n(?=Validation (?:Error|Warning|Information|Performance Warning): \[)", "\n" + txt):
        m = m.strip()
        if not m.startswith("Validation"):
            continue
        vuid = re.search(r"\[ ([^\]]+) \]", m).group(1).strip()
        body = m.split("\n", 1)[1] if "\n" in m else ""
        first = re.sub(r"0x[0-9a-f]+", "H", body.split("\n")[0])
        first = re.sub(r"\b\d+\b", "N", first)[:200]
        total += 1
        sync += vuid.startswith("SYNC-")
        groups[f"{vuid} | {first}"] += 1
    return {"total": total, "sync": sync, "groups": dict(groups.most_common(12))}


def lavapipe(a, name, config, plan=None, mode="strict", frames=None, validation=True):
    app, sid, tf, cf = SAMPLES[name]
    frames = frames or tf
    d = Path(a.work) / "lvp" / slug(name) / config
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    cmd, cwd = app_cmd(app, sid, frames)
    env = dict(os.environ, VK_ICD_FILENAMES=LVP, COVENANT_PROBE_HASH=str(d / "hashes.txt"), COVENANT_PROBE_QUIET="1")
    env.pop("LD_LIBRARY_PATH", None)
    layers = []
    paths = [a.probe]
    if plan is not None:
        layers.append("VK_LAYER_COVENANT_elide")
        paths.insert(0, a.layer)
        env.update(COVENANT_ELIDE_PLAN=str(plan), COVENANT_ELIDE_MODE=mode, COVENANT_ELIDE_STATS=str(d / "elide.json"),
                   COVENANT_ELIDE_QUIET="1")
        if config == "disabled":
            env["COVENANT_ELIDE_DISABLE"] = "1"
    if validation:
        layers.append("VK_LAYER_KHRONOS_validation")
        paths.append(VVL)
        env["VK_LAYER_SETTINGS_PATH"] = str(vvl_settings(d))
    layers.append("VK_LAYER_COVENANT_probe")
    env["VK_LAYER_PATH"] = ":".join(paths)
    env["VK_INSTANCE_LAYERS"] = ":".join(layers)
    t0 = time.time()
    r = sh(["xvfb-run", "-a", *cmd], env=env, cwd=cwd, check=False, timeout=4 * 3600)
    (d / "out.txt").write_text(r.stdout[-20000:] + r.stderr[-20000:])
    hashes = [l.split()[-1] for l in open(d / "hashes.txt")] if (d / "hashes.txt").exists() else []
    stats = json.loads((d / "elide.json").read_text()) if (d / "elide.json").exists() else None
    res = {"rc": r.returncode, "frames": frames, "hashes": hashes, "layer": stats,
           "validation": parse_vvl(d / "vvl.txt") if validation else None, "seconds": round(time.time() - t0, 1)}
    (d / "result.json").write_text(json.dumps(res, indent=1))
    return res


def run_lavapipe(a, name):
    app, sid, tf, cf = SAMPLES[name]
    plan = HERE / "plans" / f"{slug(name)}.f1.plan.json"
    out = collections.OrderedDict()
    out["base_noval"] = lavapipe(a, name, "base_noval", validation=False)
    out["base"] = lavapipe(a, name, "base")
    out["disabled"] = lavapipe(a, name, "disabled", plan=plan)
    out["strict"] = lavapipe(a, name, "strict", plan=plan, mode="strict")
    out["base_long"] = lavapipe(a, name, "base_long", frames=cf)
    out["chain"] = lavapipe(a, name, "chain", plan=plan, mode="chain", frames=cf)
    return out


# ------------------------------------------------------------------ report
def summarize(results):
    L = ["# Barrier elision on real samples: results (generated by run_real.py)", ""]
    L += ["| sample | family layout | barrier calls (all) | per-frame calls | per-frame structs removed / narrowed / kept | "
          "per-frame structs pinned (fraction) | one-time calls removed / narrowed / kept | baseline diagnostics | "
          "pinned for baseline hazards | proof s |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        for fam in ("1", "2"):
            p = r.get("prove", {}).get(fam)
            if not p or "error" in p:
                if p:
                    L.append(f"| {name} | f{fam} | ERROR | | | | | | | |")
                continue
            pf, ot = p["per_frame"], p["one_time"]
            st = pf["structs"]
            L.append(f"| {name} | f{fam} | {pf['calls'] + ot['calls']} | {pf['calls']} | "
                     f"{st.get('remove', 0)} / {st.get('narrow', 0)} / {st.get('keep', 0)} | "
                     f"{pf['pinned_structs']} ({pf['pinned_fraction']:.0%}) | {ot['remove']} / {ot['narrow']} / {ot['required']} | "
                     f"{p['proof']['baseline_diagnostics']} | {p['proof']['pinned_for_baseline_hazards']} | {p['seconds']} |")
    L += ["", "| sample | family | refcheck (new vs base): checker epoch / pairwise / reference, strict; chain | ops |",
          "|---|---|---|---|"]
    for name, r in results.items():
        for fam in ("1", "2"):
            c = r.get("check", {}).get(fam)
            if not c or "error" in c:
                continue
            s, ch = c["strict"], c["chain"]
            f = lambda x: f"{x.get('checker_epoch_new')} / {x.get('checker_pairwise_new')} / {x.get('reference_new', 'n/a')}"
            L.append(f"| {name} | f{fam} | {f(s)}; {f(ch)} | {c['ops']} |")
    L += ["", "| sample (lavapipe, 1-family plan) | hashes: base = base_noval = disabled = strict | chain = base_long | "
          "syncval hazards base / disabled / strict / base_long / chain | all VVL messages base / strict / chain | "
          "layer matched / sig mismatch (strict; chain) |", "|---|---|---|---|---|---|"]
    for name, r in results.items():
        lv = r.get("lavapipe")
        if not lv:
            continue
        same = lv["base"]["hashes"] == lv["base_noval"]["hashes"] == lv["disabled"]["hashes"] == lv["strict"]["hashes"]
        same_c = lv["chain"]["hashes"] == lv["base_long"]["hashes"]
        v = lambda k: lv[k]["validation"]["sync"] if lv[k]["validation"] else "-"
        t = lambda k: lv[k]["validation"]["total"] if lv[k]["validation"] else "-"
        ls, lc = lv["strict"]["layer"] or {}, lv["chain"]["layer"] or {}
        L.append(f"| {name} | {'yes' if same else 'NO'} ({len(lv['base']['hashes'])} frames) | "
                 f"{'yes' if same_c else 'NO'} ({len(lv['chain']['hashes'])} frames) | "
                 f"{v('base')} / {v('disabled')} / {v('strict')} / {v('base_long')} / {v('chain')} | "
                 f"{t('base')} / {t('strict')} / {t('chain')} | {ls.get('matched')} / {ls.get('sig_mismatch')}; "
                 f"{lc.get('matched')} / {lc.get('sig_mismatch')} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=",".join(SAMPLES))
    ap.add_argument("--steps", default="trace,prove,check,lavapipe")
    ap.add_argument("--families", default="1,2")
    ap.add_argument("--work", default=str(ROOT / "build" / "elide-real"))
    ap.add_argument("--driver", default=str(ROOT / "model_driver" / "build"),
                    help="model driver build dir (libvulkan.so* hard links, see model_driver/README.md)")
    ap.add_argument("--covenant", default=str(ROOT), help="PYTHONPATH holding the covenant package to prove with")
    ap.add_argument("--header", default=os.path.join(VKINC, "vulkan", "vulkan_core.h"))
    ap.add_argument("--layer", default=str(ROOT / "build" / "elide-real" / "layer"))
    ap.add_argument("--probe", default=str(ROOT / "build" / "elide-real" / "probe"))
    ap.add_argument("--no-reference", action="store_true")
    a = ap.parse_args()
    steps = a.steps.split(",")
    resf = HERE / "results" / "real.json"
    resf.parent.mkdir(exist_ok=True)
    results = json.loads(resf.read_text()) if resf.exists() else {}
    if "lavapipe" in steps:
        if not Path(a.layer, "libVkLayer_COVENANT_elide.so").exists():
            sh(["sh", str(ROOT / "elide_layer" / "build.sh"), a.layer])
        if not Path(a.probe, "libVkLayer_COVENANT_probe.so").exists():
            sh(["sh", str(HERE / "build_linux.sh"), a.probe])
    for name in a.samples.split(","):
        r = results.setdefault(name, {})
        for fam in a.families.split(","):
            if "trace" in steps:
                r.setdefault("trace", {})[fam] = trace(a, name, int(fam))
                print(name, fam, "trace", r["trace"][fam], flush=True)
            if "prove" in steps:
                r.setdefault("prove", {})[fam] = prove(a, name, int(fam))
                p = r["prove"][fam]
                print(name, fam, "prove", p.get("per_frame", p), flush=True)
            if "check" in steps:
                r.setdefault("check", {})[fam] = check(a, name, int(fam))
                c = r["check"][fam]
                print(name, fam, "check", {k: c.get(k) for k in ("strict", "chain")} if "strict" in c else c, flush=True)
            resf.write_text(json.dumps(results, indent=1))
        if "lavapipe" in steps:
            r["lavapipe"] = run_lavapipe(a, name)
            for k, v in r["lavapipe"].items():
                print(name, k, {x: v[x] for x in ("rc", "frames", "seconds")}, "hashes", len(v["hashes"]),
                      "sync", (v["validation"] or {}).get("sync"), "layer", v["layer"], flush=True)
        resf.write_text(json.dumps(results, indent=1))
    (HERE / "results" / "real.md").write_text(summarize(results))
    print((HERE / "results" / "real.md").read_text())


if __name__ == "__main__":
    sys.exit(main())
