"""Minimal reproductions of the checker/syncval disagreements found by the difftest.

    python3 bench/elide/difftest/repro.py --traces DIR --bin DIR --layer DIR --out DIR

Each case is a one- or two-struct change to the recorded deferred / particles
frame. For each it prints (and writes <out>/repro.json): the checker's new
hazards (as elide.py computes them), the fixed triage oracle's new hazards,
and syncval's hazard messages on lavapipe with the change applied by the layer.
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from covenant import elide  # noqa: E402
import timing_fixed as tf  # noqa: E402
from mutants import plan_file  # noqa: E402
from run import syncval  # noqa: E402

F = frozenset
ALLC, MW, MRW = F({"ALL_COMMANDS"}), F({"MEMORY_WRITE"}), F({"MEMORY_READ", "MEMORY_WRITE"})

CASES = [
    ("A-access-scope-dst", "deferred",
     "frame 1, the hdr COLOR_ATTACHMENT->SHADER_READ_ONLY transition (rec 1 idx 6): dstStageMask "
     "ALL_COMMANDS -> VERTEX_SHADER, dstAccessMask MEMORY_READ|WRITE -> SHADER_READ. The tonemap pass samples hdr "
     "in the FRAGMENT shader: inside the execution scope (logically later than VERTEX_SHADER) but not in the access "
     "scope (only stages named in the mask), so the transition's write is not visible to it.",
     {(1, 6): {"image[0]": ("narrow", (ALLC, MW, F({"VERTEX_SHADER"}), F({"SHADER_READ"})))}}),
    ("B-same-call-chaining", "deferred",
     "readback (rec 3 idx 0): the hdr ->TRANSFER_SRC transition gets dstStageMask TRANSFER, dstAccessMask NONE; the "
     "same call's global memory barrier (ALL_COMMANDS/MEMORY_WRITE -> TRANSFER/TRANSFER_READ) is unchanged. "
     "vkCmdCopyImageToBuffer then reads hdr; the transition's write is made visible to nothing (a memory barrier "
     "of the same command does not order that command's own layout transition).",
     {(3, 0): {"image[1]": ("narrow", (ALLC, MW, F({"TRANSFER"}), F()))}}),
    ("C-syncval-storage-buffer", "particles",
     "frame 0: remove the barrier after the 8th integrate dispatch (rec 0 idx 9). integrate writes particles7 "
     "(OpStore to a storage buffer) and the bin dispatch reads it next: a real read-after-write race.",
     {(0, 9): {"memory[0]": "remove"}}),
    ("C2-syncval-storage-buffer-atomic", "particles",
     "frame 0: remove the barrier after the bin dispatch (rec 0 idx 10). bin writes histogram with atomicAdd and "
     "shade reads it next: a real read-after-write race.",
     {(0, 10): {"memory[0]": "remove"}}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--layer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default=os.path.join(tempfile.gettempdir(), "difftest-repro"))
    a = ap.parse_args()
    Path(a.tmp).mkdir(parents=True, exist_ok=True)
    out = []
    els = {}
    for name, w, desc, plan in CASES:
        if w not in els:
            el = elide.Elider(os.path.join(a.traces, f"{w}.jsonl"), os.path.join(a.traces, f"{w}-spv"))
            el.base = set(el.hazards({})[0])
            tf.FIX_ACCESS_SCOPE = tf.FIX_SAME_CALL = True
            els[w] = (el, set(tf.hazards(el, {})))
        el, fbase = els[w]
        plan = {k: el.normalize(next(c for c in el.calls if c.key == k), v) for k, v in plan.items()}
        bad, _, _ = el.new_hazards(plan)
        fixed = set(tf.hazards(el, plan)) - fbase
        pj = plan_file(el, plan, {"id": name})
        p = Path(a.tmp) / f"{name}.plan.json"
        p.write_text(json.dumps(pj, indent=1))
        sv = syncval(os.path.join(a.bin, w), a.layer, p, 3, a.tmp)
        rec = {"case": name, "workload": w, "change": desc, "plan_entries": pj["barriers"],
               "checker_new": [f"{d.code}: {d.message} @ {d.where}" for d in bad],
               "fixed_oracle_new": sorted(f"{k[0]}: {k[2]} @ {k[1]}" for k in fixed),
               "syncval": [h.get("text") or f"{h.get('id')} {h.get('obj')} usage={h.get('usage')} "
                           f"prior={h.get('prior_usage')} cmd={h.get('cmd')}" for h in sv.get("sync", [])],
               "other_validation": sv.get("other"), "layer": sv.get("layer")}
        out.append(rec)
        print(f"== {name} ({w})\n  {desc}\n  checker: {len(bad)} new hazard(s); fixed oracle: {len(fixed)}; "
              f"syncval: {len(rec['syncval'])}")
        for s in rec["syncval"][:3]:
            print("   syncval:", s[:300])
        for s in rec["checker_new"][:3]:
            print("   checker:", s[:300])
        for s in rec["fixed_oracle_new"][:3]:
            print("   fixed:", s[:300])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "repro.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
