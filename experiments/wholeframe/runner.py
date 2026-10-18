"""Run every whole-frame program (mutant) under the COVENANT model driver (libcovvk) and keep its
trace. Khronos validation is not run here: the scorer uses the stored validation results.

    python3 runner.py --mutants mutants.json --runs RUNS --wf BUILD/wf --shaders BUILD/shaders
                      --out runs.jsonl [--md DRIVER_BUILD_DIR] [--jobs 4] [--ids a,b] [--only W1-F2,...]
                      [--display-base 191]

Per mutant: RUNS/<id>/{trace.jsonl, spv/, out.txt, err.txt, status.txt}; one JSON line per run in
--out (rc, run time, acquire order, applied mutations, X retries), which the scorer reads.
W1-F2/W1-F3/W2/W3 use an X11 surface (WF_XCB=1) on one Xvfb per worker (displays :191, :192, ...);
W2-MQ uses VK_EXT_headless_surface (WF_XCB=0). These are the surfaces of the stored validation runs.
Environment per run: COVENANT_TRACE, COVENANT_SPIRV_DIR, COVENANT_SURFACE_EXTENT=64x64,
LD_LIBRARY_PATH=<driver build>, WF_SHADERS (directory of the compiled *.spv), WF_XCB, WF_IMAGES=3,
DISPLAY, WF_MUT. --md defaults to model_driver/build of this repository.
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
A = None


def err_info(d):
    acq, applied, other = [], [], []
    try:
        for line in open(os.path.join(d, "err.txt"), errors="replace"):
            if line.startswith("ACQ "):
                acq.append(int(line.split()[2]))
            elif line.startswith("MUT-APPLIED "):
                applied.append(line.split()[1])
            elif line.startswith(("FATAL", "NO_MQ")):
                other.append(line.strip())
    except FileNotFoundError:
        pass
    return acq, applied, other


def run_one(m):
    wid = mp.current_process()._identity[0] - 1 if mp.current_process()._identity else 0
    disp = f":{A.display_base + wid % A.jobs}"
    d = os.path.join(A.runs, m["id"])
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "spv"))
    xcb = "0" if m["config"] == "W2-MQ" else "1"
    env = dict(os.environ, WF_MUT=m["mut"], DISPLAY=disp, WF_XCB=xcb, WF_IMAGES="3",
               COVENANT_TRACE=os.path.join(d, "trace.jsonl"), COVENANT_SPIRV_DIR=os.path.join(d, "spv"),
               COVENANT_SURFACE_EXTENT="64x64", LD_LIBRARY_PATH=A.md, WF_SHADERS=A.shaders)
    x_retries = 0
    while True:
        t0 = time.perf_counter()
        with open(os.path.join(d, "out.txt"), "w") as fo, open(os.path.join(d, "err.txt"), "w") as fe:
            try:
                rc = subprocess.run([A.wf] + m["args"], env=env, stdout=fo, stderr=fe, timeout=60).returncode
            except subprocess.TimeoutExpired:
                rc = 124
        t = time.perf_counter() - t0
        # a transient X connection failure (the program exits before any frame): retry, recorded
        if rc == 2 and xcb == "1" and x_retries < 5 and "FATAL no X display" in open(
                os.path.join(d, "err.txt"), errors="replace").read():
            x_retries += 1
            time.sleep(0.5)
            continue
        break
    with open(os.path.join(d, "status.txt"), "w") as f:
        f.write(f"rc={rc} run_s={t:.6f}\n")
    acq, applied, other = err_info(d)
    try:
        nrec = sum(1 for _ in open(os.path.join(d, "trace.jsonl")))
    except FileNotFoundError:
        nrec = 0
    return {"id": m["id"], "config": m["config"], "gen": m["gen"], "mut": m["mut"], "rc": rc, "t": t,
            "acq": acq, "applied": applied, "other": other, "trace_records": nrec,
            "xcb": xcb == "1", "display": disp if xcb == "1" else None,
            "x_retries": x_retries}


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutants", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--md", default=os.environ.get("MD", os.path.join(REPO, "model_driver", "build")),
                    help="model driver build dir (holds libvulkan.so.1); default: $MD or model_driver/build")
    ap.add_argument("--wf", required=True, help="the wf binary (build.sh)")
    ap.add_argument("--shaders", required=True, help="directory of the compiled shaders (build.sh)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--ids", default="")
    ap.add_argument("--display-base", type=int, default=191)
    A = ap.parse_args()
    A.wf, A.shaders, A.md, A.runs = map(os.path.abspath, (A.wf, A.shaders, A.md, A.runs))
    ms = json.load(open(A.mutants))
    if A.only:
        ms = [m for m in ms if m["config"] in A.only.split(",")]
    if A.ids:
        ms = [m for m in ms if m["id"] in A.ids.split(",")]
    os.makedirs(A.runs, exist_ok=True)
    xs = [subprocess.Popen(["Xvfb", f":{A.display_base + i}", "-screen", "0", "320x240x24", "-nolisten", "tcp"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for i in range(A.jobs)]
    print("Xvfb pids", [x.pid for x in xs], flush=True)
    time.sleep(1.5)
    t0 = time.time()
    try:
        with mp.Pool(A.jobs) as pool, open(A.out, "w") as f:
            for i, rec in enumerate(pool.imap_unordered(run_one, ms)):
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if (i + 1) % 100 == 0:
                    print(f"{i + 1}/{len(ms)} {time.time() - t0:.0f}s", flush=True)
    finally:
        for x in xs:          # by PID, never by pattern
            x.terminate()
            x.wait()
    print(f"done {len(ms)} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
