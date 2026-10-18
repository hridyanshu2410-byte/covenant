"""Run every mutant through both oracles: the COVENANT checker and syncval.

    python3 bench/elide/difftest/run.py --traces DIR --mutants DIR --bin DIR --layer DIR --out results.jsonl

Checker verdict: the mutant is applied to the recorded frame exactly as
elide.py applies a plan (Elider.build: strengthened frame, then timing.check
plus Elider.transition_reach), and the verdict is the set of diagnostics the
unmodified (strengthened) frame does not report ("new hazards"). The plain
timing.check of the unstrengthened frame with the mutation applied is
recorded too ("raw"), for triage.

Syncval verdict: the workload runs on lavapipe for the traced number of
frames with VK_LAYER_COVENANT_elide applying the mutant (strict matching)
and the Khronos validation layer with synchronization validation enabled;
every SYNC-* message is a hazard. Other validation messages, the layer's
match statistics and the readback hashes are recorded too.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from covenant import elide, timing  # noqa: E402
from covenant.diagnostics import Report  # noqa: E402
from covenant.timing import Barrier, Frame  # noqa: E402
from mutants import WORKLOADS, act_from_json  # noqa: E402

LVP = "/usr/share/vulkan/icd.d/lvp_icd.json"
SYNC_RE = re.compile(r"Hazard (\w+) for (Vk\w+) 0x[0-9a-fA-F]+\[([^\]]*)\]")
USAGE_RE = re.compile(r"usage: (\w+), prior_usage: (\w+)")
CMD_RE = re.compile(r"\| (vk\w+)\(\):")
SEQ_RE = re.compile(r"command: (vk\w+), seq_no: (\d+)")
CB_RE = re.compile(r"in VkCommandBuffer 0x[0-9a-fA-F]+\[([^\]]*)\]")
QUOTED = re.compile(r"'([^']+)'")


def diag_json(d):
    m = QUOTED.search(d.message)
    return {"code": d.code, "res": m.group(1) if m else None, "where": d.where[:160], "message": d.message[:300],
            "notes": [n[:200] for n in (d.notes or [])][:2]}


def raw_apply(el, plan):
    """The mutation on the unstrengthened frame (as Elider._raw_check)."""
    fr = el.frame
    q2 = {}
    for qn, ops in fr.queues.items():
        out = []
        for op in ops:
            ce = el.op_elem.get(id(op)) if isinstance(op, Barrier) else None
            if ce:
                act = plan.get(ce[0].key, {}).get(ce[1].label)
                if act == "remove":
                    continue
                if isinstance(act, tuple):
                    op = Barrier(op.res, op.old, op.new, op.where, *act[1])
            out.append(op)
        q2[qn] = out
    return Frame(queues=q2, initial_layouts=dict(fr.initial_layouts), swapchains=dict(fr.swapchains),
                 semaphore_reuse=list(fr.semaphore_reuse), findings=list(fr.findings))


def checker(el, plan, raw_base):
    t0 = time.perf_counter()
    bad, _, _ = el.new_hazards(plan)
    t1 = time.perf_counter()
    after = timing.check(raw_apply(el, plan), Report()).items
    raw_new = [d for d in after if (d.code, d.where, d.message) not in raw_base]
    t2 = time.perf_counter()
    return {"new": [diag_json(d) for d in bad], "raw_new": [diag_json(d) for d in raw_new],
            "seconds": t1 - t0, "seconds_with_raw": t2 - t0}


def parse_validation(stderr):
    sync, other = [], []
    for line in stderr.splitlines():
        if not line.startswith("[validation] "):
            continue
        vid = line[len("[validation] "):].split(":", 1)[0]
        if vid.startswith("SYNC-"):
            h = {"id": vid}
            m = SYNC_RE.search(line)
            if m:
                h.update(hazard=m.group(1), objtype=m.group(2), obj=m.group(3))
            m = USAGE_RE.search(line)
            if m:
                h.update(usage=m.group(1), prior_usage=m.group(2))
            m = CMD_RE.search(line)
            if m:
                h["cmd"] = m.group(1)
            m = SEQ_RE.search(line)
            if m:
                h.update(prior_cmd=m.group(1), prior_seq=int(m.group(2)))
            m = CB_RE.search(line)
            if m:
                h["cb"] = m.group(1)
            if "objtype" not in h:
                h["text"] = line[:600]
            sync.append(h)
        else:
            other.append({"id": vid, "text": line[:400]})
    return sync, other


def syncval(binary, layer_dir, plan, frames, tmp, validation=True):
    env = dict(os.environ, VK_ICD_FILENAMES=LVP, VK_LAYER_DUPLICATE_MESSAGE_LIMIT="1000000")
    env.pop("VK_INSTANCE_LAYERS", None)
    out = Path(tmp) / "run.json"
    stats = Path(tmp) / "stats.json"
    for p in (out, stats):
        if p.exists():
            p.unlink()
    if plan is not None:
        env["VK_LAYER_PATH"] = f"{layer_dir}:/usr/share/vulkan/explicit_layer.d"
        env["VK_INSTANCE_LAYERS"] = "VK_LAYER_COVENANT_elide"
        env["COVENANT_ELIDE_PLAN"] = str(plan)
        env["COVENANT_ELIDE_MODE"] = "strict"
        env["COVENANT_ELIDE_STATS"] = str(stats)
        env["COVENANT_ELIDE_QUIET"] = "1"
    if validation:
        env["ELIDE_VALIDATION"] = "1"
    t0 = time.perf_counter()
    r = subprocess.run([binary, "--frames", str(frames), "--json", str(out), "--quiet"], env=env,
                       capture_output=True, text=True, timeout=600)
    dt = time.perf_counter() - t0
    res = {"returncode": r.returncode, "seconds": dt}
    if r.returncode != 0 or not out.exists():
        res["error"] = r.stderr[-2000:]
        return res
    j = json.loads(out.read_text())
    sync, other = parse_validation(r.stderr)
    res.update(hashes=j["hashes"], sync=sync, other=other, sync_count_app=j["sync_hazards"],
               validation_messages=j["validation_messages"],
               layer=json.loads(stats.read_text()) if stats.exists() else None)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--mutants", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--layer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--tmp", default=os.path.join(tempfile.gettempdir(), "difftest-run"))
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    Path(a.tmp).mkdir(parents=True, exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            done.add(json.loads(line)["id"])
    fo = open(a.out, "a")
    for w in WORKLOADS:
        if a.only and w not in a.only.split(","):
            continue
        el = elide.Elider(os.path.join(a.traces, f"{w}.jsonl"), os.path.join(a.traces, f"{w}-spv"))
        el.base = set(el.hazards({})[0])
        raw_base = {(d.code, d.where, d.message) for d in timing.check(el.frame, Report()).items}
        binary = os.path.join(a.bin, w)
        # controls: no layer; layer with an empty plan
        for cid, plan in ((f"{w}-base", None), (f"{w}-empty", None)):
            if cid in done:
                continue
            if cid.endswith("empty"):
                p = Path(a.tmp) / f"{w}-empty.plan.json"
                p.write_text(json.dumps({"format": "covenant-elide-plan/1", "barriers": [], "chain": []}))
                plan = p
            rec = {"id": cid, "workload": w, "control": True, "checker": checker(el, {}, raw_base),
                   "syncval": syncval(binary, a.layer, plan, a.frames, a.tmp),
                   "novalidation": syncval(binary, a.layer, plan, a.frames, a.tmp, validation=False),
                   "checker_baseline": len(el.base), "raw_baseline": len(raw_base)}
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
        manifest = json.loads((Path(a.mutants) / w / "manifest.json").read_text())
        if a.limit:
            manifest = manifest[:a.limit]
        for k, m in enumerate(manifest):
            if m["id"] in done:
                continue
            plan = {tuple(int(x) for x in key.split(",")): {l: act_from_json(v) for l, v in acts.items()}
                    for key, acts in m["internal"].items()}
            ck = checker(el, plan, raw_base)
            sv = syncval(binary, a.layer, Path(a.mutants) / m["plan"], a.frames, a.tmp)
            rec = {k2: v for k2, v in m.items() if k2 != "internal"}
            rec.update(checker=ck, syncval=sv)
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            if k % 25 == 0:
                print(f"{m['id']}: checker {len(ck['new'])} new ({ck['seconds']*1000:.1f} ms), "
                      f"syncval {len(sv.get('sync', []))} ({sv['seconds']:.2f} s), other {len(sv.get('other', []))}",
                      flush=True)
    fo.close()


if __name__ == "__main__":
    main()
