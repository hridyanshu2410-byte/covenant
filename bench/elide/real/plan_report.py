"""Summarize a covenant-elide plan of a real sample: per-frame vs one-time barriers,
decisions, pin reasons, and the recorded -> planned masks of every per-frame call.

    python3 plan_report.py plan.json [--json out.json] [--detail]

"Per-frame" = a barrier call whose recording is submitted after the first
vkQueuePresentKHR of the trace (with --trace), i.e. in the steady-state frame
loop; without a present (headless) or without --trace, a call whose recording
was submitted more than once. Everything else is "one-time" (setup: uploads,
mip generation...), which cannot change the steady-state frame time.
"""
import argparse
import collections
import json
import re

ABBR = [("VK_PIPELINE_STAGE_2_", ""), ("VK_ACCESS_2_", ""), ("_BIT", "")]


def reason_class(w):
    w = re.sub(r" \((timing|elide)\..*", "", w)
    w = w.replace("pinned: ", "").replace("kept: ", "")
    if w.startswith("its command buffer: "):
        w = "command buffer: " + w[len("its command buffer: "):]
    return w[:110]


def masks(m):
    return f"{'|'.join(m['src_stage']) or 'NONE'}/{'|'.join(m['src_access']) or '-'} -> " \
           f"{'|'.join(m['dst_stage']) or 'NONE'}/{'|'.join(m['dst_access']) or '-'}"


def steady_recordings(trace):
    """Recording ordinals submitted after the first present (None: no present)."""
    first_present, begins, subs = None, [], []
    with open(trace) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            fn = r.get("fn", "")
            if fn == "vkBeginCommandBuffer":
                begins.append((r["seq"], r["cb"]))
            elif fn.startswith("vkQueueSubmit"):
                cbs = [c for b in r.get("submits") or [] for c in b.get("command_buffers") or []]
                subs.append((r["seq"], cbs))
            elif fn == "vkQueuePresentKHR" and first_present is None:
                first_present = r["seq"]
    if first_present is None:
        return None
    last = {}          # cb -> ordinal of its latest recording
    out = set()
    events = sorted([(s, 0, ("b", i, cb)) for i, (s, cb) in enumerate(begins)] +
                    [(s, 1, ("s", cbs)) for s, cbs in subs], key=lambda x: (x[0], x[1]))
    for seq, _, ev in events:
        if ev[0] == "b":
            last[ev[2]] = ev[1]
        elif seq > first_present:
            for cb in ev[1]:
                if cb in last:
                    out.add(last[cb])
    return out


def analyse(pj, trace=None):
    calls = pj["barriers"]
    steady = steady_recordings(trace) if trace else None
    if steady is not None:
        per = [b for b in calls if b["rec"] in steady]
    else:
        per = [b for b in calls if b.get("occurrences", 1) > 1]
    one = [b for b in calls if b not in per]
    out = collections.OrderedDict()
    for name, group in (("per_frame", per), ("one_time", one)):
        d = collections.Counter(b["decision"] for b in group)
        st = collections.Counter()
        reasons = collections.Counter()
        kinds = collections.Counter()
        for b in group:
            for e in b["elements"]:
                st[e["action"]] += 1
                lay = e.get("layout", "")
                fam = "transition" if lay and lay.split(" -> ")[0] != lay.split(" -> ")[1] else "plain"
                kinds[fam] += 1
            for k, w in (b.get("why") or {}).items():
                if not w.startswith(("removed", "narrowed")):
                    reasons[reason_class(w)] += 1
        out[name] = collections.OrderedDict(
            calls=len(group), remove=d.get("remove", 0), narrow=d.get("narrow", 0), required=d.get("required", 0),
            structs=dict(st), struct_kinds=dict(kinds), keep_reasons=dict(reasons.most_common()))
    # pinned fraction (per-frame): structs kept because they were never tried
    # (pinned) vs tried and needed
    pf = out["per_frame"]
    tried_needed = sum(v for k, v in pf["keep_reasons"].items() if k.startswith(("needed", "layout transition, no narrower")))
    pinned = sum(pf["keep_reasons"].values()) - tried_needed
    tot = sum(pf["structs"].values()) or 1
    pf["pinned_structs"] = pinned
    pf["pinned_fraction"] = round(pinned / tot, 3)
    pf["tried_and_needed_structs"] = tried_needed
    # distinct per-frame barrier calls (chain keys), with masks
    seen, detail = set(), []
    for b in per:
        if b["chain"] in seen:
            continue
        seen.add(b["chain"])
        inst = [x for x in per if x["chain"] == b["chain"]]
        ds = collections.Counter(x["decision"] for x in inst)
        ch = next((c for c in pj.get("chain", []) if c["chain"] == b["chain"]), None)
        for e in b["elements"]:
            ce = None
            if ch:
                ce = next((x for x in ch["elements"] if x["kind"] == e["kind"] and x["index"] == e["index"]), None)
            detail.append(collections.OrderedDict(
                chain=b["chain"], instances=len(inst), decisions=dict(ds), sync=b["sync"],
                what=f"{e['kind']}[{e['index']}] {e.get('image', e.get('buffer', ''))} {e.get('layout', '')}".strip(),
                recorded=masks(e["recorded"]),
                chain_action=(ce or {"action": "keep"})["action"],
                chain_new=masks(ce["new"]) if ce and ce.get("new") else None,
                why=sorted({reason_class((x.get("why") or {}).get(f"{e['kind']}[{e['index']}]", (x.get("why") or {}).get("*", "")))
                            for x in inst})))
    out["per_frame_detail"] = detail
    out["proof"] = {k: pj["proof"].get(k) for k in ("baseline_diagnostics", "plain_timing_check", "pairwise_recheck",
                                                    "pinned_for_baseline_hazards", "oracle_runs", "seconds")}
    out["chain_entries"] = pj["summary"]["chain_entries"]
    out["chain_note"] = pj["summary"]["chain_note"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--json")
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--trace", help="the trace the plan was proved from (to find the steady-state frames)")
    a = ap.parse_args()
    r = analyse(json.load(open(a.plan)), a.trace)
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
    for k in ("per_frame", "one_time"):
        print(k, json.dumps(r[k]))
    print("proof", json.dumps(r["proof"]))
    if a.detail:
        for d in r["per_frame_detail"]:
            s = f"  x{d['instances']} {d['decisions']} {d['what']}: {d['recorded']}"
            if d["chain_new"]:
                s += f"  ==> {d['chain_new']}"
            elif d["chain_action"] == "remove":
                s += "  ==> REMOVED"
            print(s, "|", "; ".join(x for x in d["why"] if x))


if __name__ == "__main__":
    main()
