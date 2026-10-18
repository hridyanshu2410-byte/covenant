"""Check an elision plan against the reference semantics (covenant/timing_ref.py)
and the fast checker, on the UNSTRENGTHENED frame of the trace it was proved from.

    PYTHONPATH=<repo root> python3 refcheck.py trace.jsonl spv/ plan.json [--json out.json]

For each of: the recorded program (base), the plan applied by strict keys, and
the plan applied by chain keys (to every recording whose barrier sequence has
that chain value, which is what the layer does in chain mode), it reports

* reference races: Reference(frame, discard_relaxed=True).races(), i.e. the
  spec's Haz_spec with relaxation R1 (the checker's default);
* checker diagnostics: timing.check(frame) in epoch mode and in pairwise
  (exhaustive) mode;

and the NEW ones relative to base. A race / diagnostic is identified by the
trace locations (op ``where`` strings) of both accesses, not by positions,
because removing barrier ops shifts positions.
"""
import argparse
import json
import sys
import time

from covenant import elide, timing, timing_ref
from covenant.diagnostics import Report
from covenant.timing import Barrier, Frame


def plan_actions(el, pj, mode):
    """{call key: {elem label: "remove" | ("narrow", masks)}} from a plan file."""
    def acts(entry):
        out = {}
        for e in entry["elements"]:
            lab = f"{e['kind']}[{e['index']}]"
            if e["action"] == "remove":
                out[lab] = "remove"
            elif e["action"] == "narrow":
                n = e["new"]
                out[lab] = ("narrow", (frozenset(n["src_stage"]), frozenset(n["src_access"]),
                                       frozenset(n["dst_stage"]), frozenset(n["dst_access"])))
        return out
    plan = {}
    if mode == "strict":
        for b in pj["barriers"]:
            a = acts(b)
            if a:
                plan[(b["rec"], b["idx"])] = a
    else:
        by_chain = {c["chain"]: acts(c) for c in pj.get("chain", [])}
        for c in el.calls:
            if c.chain in by_chain and c.sig == next(x["sig"] for x in pj["chain"] if x["chain"] == c.chain):
                plan[c.key] = by_chain[c.chain]
    return plan


def apply(el, plan):
    q2 = {}
    for qn, ops in el.frame.queues.items():
        out = []
        for op in ops:
            ce = el.op_elem.get(id(op)) if isinstance(op, Barrier) else None
            if ce:
                act = plan.get(ce[0].key, {}).get(ce[1].label)
                if act == "remove":
                    continue
                if isinstance(act, tuple):
                    op = Barrier(op.res, op.old, op.new, op.where, *act[1], call=op.call, struct=op.struct,
                                 region=op.region)
            out.append(op)
        q2[qn] = out
    fr = el.frame
    return Frame(queues=q2, initial_layouts=dict(fr.initial_layouts), swapchains=dict(fr.swapchains),
                 semaphore_reuse=list(fr.semaphore_reuse), findings=list(fr.findings))


def ev_id(e):
    return (e.op.where, str(e.res), "LT" if e.lt_of is not None else e.kind)


def ref_races(frame):
    return {frozenset((ev_id(a), ev_id(b))) for a, b in timing_ref.Reference(frame, True).races()}


def diag_ids(frame, exhaustive):
    rep = timing.check(frame, Report(), exhaustive=exhaustive, window=False)
    return {elide.Elider.hkey(d) for d in rep.items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("spv")
    ap.add_argument("plan")
    ap.add_argument("--json")
    ap.add_argument("--header")
    ap.add_argument("--no-reference", action="store_true", help="skip the (slow) reference semantics")
    a = ap.parse_args()
    el = elide.Elider(a.trace, a.spv, a.header)
    pj = json.load(open(a.plan))
    frames = {"base": el.frame, "strict": apply(el, plan_actions(el, pj, "strict")),
              "chain": apply(el, plan_actions(el, pj, "chain"))}
    out = {"ops": sum(len(v) for v in el.frame.queues.values())}
    base = {}
    for name, fr in frames.items():
        t0 = time.time()
        r = {}
        for kind, fn in (("checker_epoch", lambda f: diag_ids(f, False)),
                         ("checker_pairwise", lambda f: diag_ids(f, True)),
                         ("reference", None if a.no_reference else ref_races)):
            if fn is None:
                continue
            s = fn(fr)
            if name == "base":
                base[kind] = s
            r[kind] = len(s)
            r[kind + "_new"] = len(s - base[kind])
            if s - base[kind]:
                r[kind + "_new_examples"] = [str(sorted(map(str, x)))[:300] for x in list(s - base[kind])[:3]]
        r["seconds"] = round(time.time() - t0, 2)
        out[name] = r
    print(json.dumps(out, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    bad = any(out[n].get(k + "_new") for n in ("strict", "chain") for k in ("checker_epoch", "checker_pairwise", "reference"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
