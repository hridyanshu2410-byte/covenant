"""Random barrier-plan mutants for differential testing (checker vs syncval).

    python3 bench/elide/difftest/mutants.py --traces DIR --out DIR [--n 250] [--seed 20261005]

For every workload trace, generate N random "mutant" plans in the
covenant-elide-plan/1 format (so VK_LAYER_COVENANT_elide applies them, strict
matching). A mutant draws a removal probability p_remove ~ U(0.05, 0.60) and a
narrowing probability p_narrow ~ U(0.0, 0.30), then for every recorded barrier
call (in every recorded frame) independently:

* if no struct of the call is pinned or a layout transition, remove the whole
  call with probability p_remove;
* otherwise, per struct: remove it with probability p_remove (only if it is
  neither pinned nor a layout transition and the call has more than one
  struct), else narrow it with probability p_narrow (only if not pinned).

Pinning is elide.py's own (Elider._pin): host synchronization, queue family
ownership transfers, masks the model cannot name, unsubmitted barriers,
command buffers with unmodelled commands, batches signalling a semaphore with
a narrow stage mask. Layout transitions are narrowed but never removed, as in
elide.py. A narrowing replaces each of the four masks (with probability 1/2
each) by a random non-empty subset of a fixed vocabulary of legacy (sync1)
stage / access names that lie inside the recorded mask (as timing.py reads
masks); access flags are kept only when some chosen stage supports them, so
the plan stays valid Vulkan. sync1 calls get one stage mask per call
(Elider.normalize).

Nothing in covenant/ is modified; this file only imports it.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from covenant import elide, timing  # noqa: E402

WORKLOADS = ["particles", "deferred", "postchain"]
STAGES = ["TOP_OF_PIPE", "DRAW_INDIRECT", "VERTEX_INPUT", "VERTEX_SHADER", "EARLY_FRAGMENT_TESTS",
          "FRAGMENT_SHADER", "LATE_FRAGMENT_TESTS", "COLOR_ATTACHMENT_OUTPUT", "COMPUTE_SHADER", "TRANSFER",
          "BOTTOM_OF_PIPE", "ALL_GRAPHICS", "ALL_COMMANDS"]
ACCESS = ["INDIRECT_COMMAND_READ", "INDEX_READ", "VERTEX_ATTRIBUTE_READ", "UNIFORM_READ", "SHADER_READ",
          "SHADER_WRITE", "COLOR_ATTACHMENT_READ", "COLOR_ATTACHMENT_WRITE", "DEPTH_STENCIL_ATTACHMENT_READ",
          "DEPTH_STENCIL_ATTACHMENT_WRITE", "TRANSFER_READ", "TRANSFER_WRITE", "MEMORY_READ", "MEMORY_WRITE"]
SHADER = {"VERTEX_SHADER", "FRAGMENT_SHADER", "COMPUTE_SHADER", "ALL_GRAPHICS", "ALL_COMMANDS"}
# which stages support each access flag (Vulkan "supported access types" table, legacy names)
SUPPORT = {
    "INDIRECT_COMMAND_READ": {"DRAW_INDIRECT", "ALL_GRAPHICS", "ALL_COMMANDS"},
    "INDEX_READ": {"VERTEX_INPUT", "ALL_GRAPHICS", "ALL_COMMANDS"},
    "VERTEX_ATTRIBUTE_READ": {"VERTEX_INPUT", "ALL_GRAPHICS", "ALL_COMMANDS"},
    "UNIFORM_READ": SHADER, "SHADER_READ": SHADER, "SHADER_WRITE": SHADER,
    "COLOR_ATTACHMENT_READ": {"COLOR_ATTACHMENT_OUTPUT", "ALL_GRAPHICS", "ALL_COMMANDS"},
    "COLOR_ATTACHMENT_WRITE": {"COLOR_ATTACHMENT_OUTPUT", "ALL_GRAPHICS", "ALL_COMMANDS"},
    "DEPTH_STENCIL_ATTACHMENT_READ": {"EARLY_FRAGMENT_TESTS", "LATE_FRAGMENT_TESTS", "ALL_GRAPHICS", "ALL_COMMANDS"},
    "DEPTH_STENCIL_ATTACHMENT_WRITE": {"EARLY_FRAGMENT_TESTS", "LATE_FRAGMENT_TESTS", "ALL_GRAPHICS", "ALL_COMMANDS"},
    "TRANSFER_READ": {"TRANSFER", "ALL_COMMANDS"}, "TRANSFER_WRITE": {"TRANSFER", "ALL_COMMANDS"},
    "MEMORY_READ": set(STAGES) - {"TOP_OF_PIPE", "BOTTOM_OF_PIPE"},
    "MEMORY_WRITE": set(STAGES) - {"TOP_OF_PIPE", "BOTTOM_OF_PIPE"},
}


def _subset(rng, items):
    if not items:
        return []
    k = rng.randint(1, min(len(items), 3))
    return rng.sample(items, k)


def random_narrowing(rng, e):
    """A random (ss, sa, ds, da) inside the element's recorded masks, or None."""
    oss, osa, ods, oda = e.masks()
    if None in (oss, osa, ods, oda):
        return None
    for _ in range(20):
        ss, ds = set(oss), set(ods)
        if rng.random() < 0.5:
            ss = set(_subset(rng, [s for s in STAGES if s in oss or timing._stage_in(s, oss, True)]))
        if rng.random() < 0.5:
            ds = set(_subset(rng, [s for s in STAGES if s in ods or timing._stage_in(s, ods, False)]))
        sa, da = set(osa), set(oda)
        if rng.random() < 0.5:
            cand = [a for a in ACCESS if a.endswith("WRITE") and timing._access_in(a, "write", osa)]
            sa = set(_subset(rng, cand)) if rng.random() < 0.85 else set()
        if rng.random() < 0.5:
            cand = [a for a in ACCESS if timing._access_in(a, "write" if a.endswith("WRITE") else "read", oda)]
            da = set(_subset(rng, cand)) if rng.random() < 0.85 else set()
        # keep access flags only where a chosen stage supports them (valid Vulkan)
        sa = {a for a in sa if SUPPORT[a] & ss}
        da = {a for a in da if SUPPORT[a] & ds}
        cand = (frozenset(ss), frozenset(sa), frozenset(ds), frozenset(da))
        if cand != (oss, osa, ods, oda) and ss and ds:
            return cand
    return None


def act_to_json(a):
    if a == "remove" or a == "keep" or a is None:
        return a or "keep"
    return ["narrow", [sorted(x) for x in a[1]]]


def act_from_json(a):
    if isinstance(a, list):
        return ("narrow", tuple(frozenset(x) for x in a[1]))
    return a


def make_mutant(el, rng, p_remove, p_narrow):
    plan = {}
    for call in el.calls:
        if call.pin:
            continue
        free = [e for e in call.elems if not e.pin and not e.transition]
        if free and len(free) == len(call.elems) and rng.random() < p_remove:
            plan[call.key] = {e.label: "remove" for e in call.elems}
            continue
        acts = {}
        for e in call.elems:
            if e.pin:
                continue
            if not e.transition and len(call.elems) > 1 and rng.random() < p_remove:
                acts[e.label] = "remove"
            elif rng.random() < p_narrow:
                n = random_narrowing(rng, e)
                if n is not None:
                    acts[e.label] = ("narrow", n)
        if acts:
            plan[call.key] = el.normalize(call, acts)
    return plan


def plan_file(el, plan, meta):
    """covenant-elide-plan/1 JSON for a mutant (only mutated calls are listed)."""
    barriers = []
    for c in el.calls:
        if c.key in plan:
            j = el._call_json(c, plan[c.key])
            if j["decision"] != "required":
                barriers.append(j)
    return {"format": "covenant-elide-plan/1", "unsafe_control": True, "mutant": meta, "barriers": barriers,
            "chain": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, help="dir with <w>.jsonl and <w>-spv/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20261005)
    a = ap.parse_args()
    out = Path(a.out)
    for wi, w in enumerate(WORKLOADS):
        el = elide.Elider(os.path.join(a.traces, f"{w}.jsonl"), os.path.join(a.traces, f"{w}-spv"))
        rng = random.Random(a.seed * 10 + wi)
        d = out / w
        d.mkdir(parents=True, exist_ok=True)
        seen = set()
        manifest = []
        i = 0
        while len(manifest) < a.n:
            p_remove = round(rng.uniform(0.05, 0.60), 4)
            p_narrow = round(rng.uniform(0.0, 0.30), 4)
            plan = make_mutant(el, rng, p_remove, p_narrow)
            i += 1
            key = json.dumps(sorted((str(k), sorted((l, str(act_to_json(v))) for l, v in acts.items()))
                                    for k, acts in plan.items()))
            if not plan or key in seen:
                continue
            seen.add(key)
            mid = f"{w}-{len(manifest):03d}"
            meta = {"id": mid, "workload": w, "seed": a.seed, "p_remove": p_remove, "p_narrow": p_narrow}
            pj = plan_file(el, plan, meta)
            path = d / f"{mid}.plan.json"
            path.write_text(json.dumps(pj, indent=1))
            n_rm_calls = sum(1 for b in pj["barriers"] if b["decision"] == "remove")
            n_rm_structs = sum(1 for b in pj["barriers"] for e in b["elements"] if e["action"] == "remove")
            n_nar = sum(1 for b in pj["barriers"] for e in b["elements"] if e["action"] == "narrow")
            manifest.append(dict(meta, plan=str(path.relative_to(out)), calls_listed=len(pj["barriers"]),
                                 calls_removed=n_rm_calls, structs_removed=n_rm_structs, structs_narrowed=n_nar,
                                 internal={f"{k[0]},{k[1]}": {l: act_to_json(v) for l, v in acts.items()}
                                           for k, acts in plan.items()}))
        (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print(f"{w}: {len(manifest)} mutants ({i} drawn); calls {len(el.calls)}, "
              f"pinned calls {sum(1 for c in el.calls if c.pin)}", flush=True)


if __name__ == "__main__":
    main()
