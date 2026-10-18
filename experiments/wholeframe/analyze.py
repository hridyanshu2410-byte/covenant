"""Compare COVENANT and Khronos validation per mutant and per resource, and triage
every disagreement.

    python3 analyze.py --mutants mutants.json --results results.jsonl --runs RUNS_DIR --out results.json

Reference validator: lavapipe + VVL with synchronization validation for the
single-queue configurations; the Khronos mock ICD + the same VVL for W2-MQ
(lavapipe has one queue family). COVENANT: model driver + host_trace + timing.check.
results.jsonl holds, per mutant, the validation runs ("sv", "mock") and two COVENANT
results, "snap" and "fix" (score.py fills them; see there). RUNS_DIR/<id>/<oracle>/vvl.txt
are the validation logs; score.py replaces load_msgs with a lookup in the stored,
already parsed messages.
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vvlparse import parse  # noqa: E402

# ------------------------------------------------------------------ normalisation
HAZ_CODES = {"timing.missing-barrier", "timing.barrier-scope", "timing.wait-stage-scope",
             "timing.cross-queue-race", "timing.free-while-in-use", "timing.signal-stage-scope"}
SV_TYPE = {"READ-AFTER-WRITE": "RAW", "WRITE-AFTER-READ": "WAR", "WRITE-AFTER-WRITE": "WAW",
           "PRESENT-AFTER-WRITE": "RAW", "WRITE-AFTER-PRESENT": "WAR", "READ-RACING-WRITE": "RAW",
           "WRITE-RACING-READ": "WAR", "PRESENT-AFTER-READ": "WAR", "READ-AFTER-PRESENT": "RAW"}
PSEUDO = {"cmdbuf": "cb-in-use", "fence": "fence-in-use", "semaphore": "acq-sem"}


def cov_keys(chk):
    """COVENANT diagnostics -> {(class, object)}; hazards also -> {(object, type)}."""
    keys, types = set(), set()
    for d in chk.get("diags", []):
        if d.get("severity") == "warning":
            continue
        c, res = d["code"], d.get("res")
        if res and ":" in res and res.split(":", 1)[0] in PSEUDO:
            keys.add((PSEUDO[res.split(":", 1)[0]], res.split(":", 1)[1]))
        elif c in HAZ_CODES:
            host = "'host'" in d["message"] or "host write" in d["where"] or \
                any("host write" in n for n in d.get("notes", []))
            keys.add(("host-race" if host else "hazard", res))
        elif c == "timing.semaphore-resignal":
            keys.add(("acq-sem" if d["where"].startswith("vkAcquireNextImageKHR") else "sem-resignal", res))
        elif c == "timing.acquire-semaphore-in-use":
            keys.add(("acq-sem", res))
        elif c == "timing.present-semaphore-reuse":
            keys.add(("present-sem-reuse", res))
        elif c == "timing.swapchain-overacquire":
            keys.add(("overacquire", res))
        elif c in ("timing.wrong-layout", "timing.stale-barrier"):
            keys.add(("layout", res))
        elif c.startswith("timing.ownership"):
            keys.add(("qfot", res))
        elif c == "timing.signal-scope":
            keys.add(("signal-scope", res))
        elif c == "timing.timeline-value":
            keys.add(("timeline-value", res))
        elif c == "timing.command-buffer-in-use":
            keys.add(("cb-in-use", res))
        elif c == "timing.fence-in-use":
            keys.add(("fence-in-use", res))
        elif c == "timing.deadlock":
            keys.add(("deadlock", res))
        else:
            keys.add((c, res))
    for h in chk.get("hazards", []):
        r = h.get("res") or ""
        if ":" in r and r.split(":", 1)[0] in PSEUDO or "host" in (h["e_q"], h["u_q"]):
            continue
        t = {("write", "read"): "RAW", ("read", "write"): "WAR", ("write", "write"): "WAW"}.get(
            (h["e_kind"], h["u_kind"]))
        if t:
            types.add((r, t))
    return keys, types


def _named(m, kind):
    for k, n in m.get("named", []):
        if k == kind:
            return n
    return None


def vvl_key(m):
    i, t = m["id"], m["text"]
    if i.startswith("SYNC-HAZARD"):
        return ("hazard", m["res"])
    if i in ("VUID-vkAcquireNextImageKHR-semaphore-01286", "VUID-vkAcquireNextImageKHR-semaphore-01779"):
        return ("acq-sem", _named(m, "VkSemaphore"))
    if i in ("VUID-vkQueueSubmit2-semaphore-03868", "VUID-vkQueueSubmit-pSignalSemaphores-00067"):
        if "in use by VkSwapchainKHR" in t:
            return ("present-sem-reuse", _named(m, "VkSemaphore"))
        return ("sem-resignal", _named(m, "VkSemaphore"))
    if i in ("VUID-vkBeginCommandBuffer-commandBuffer-00049", "VUID-vkQueueSubmit2-commandBuffer-03875"):
        return ("cb-in-use", _named(m, "VkCommandBuffer"))
    if i == "VUID-vkResetFences-pFences-01123":
        return ("fence-in-use", _named(m, "VkFence"))
    if i in ("UNASSIGNED-non-acquired-swapchain-image-used",
             "UNASSIGNED-VkPresentInfoKHR-pImageIndices-MissingAcquireWait"):
        return ("acquire-not-waited", _named(m, "VkImage"))
    if i == "VUID-vkAcquireNextImageKHR-surface-07783":
        return ("overacquire", "swapchain")
    if i == "VUID-VkSubmitInfo2-semaphore-03882":
        return ("timeline-value", _named(m, "VkSemaphore"))
    if i == "VUID-vkQueueSubmit2-commandBuffer-03879":
        return ("qfot", m["res"])
    if i.endswith(("srcStageMask-09675", "dstStageMask-09676")):
        return ("invalid-mutant", None)
    return (i, m["res"])


def vvl_keys(msgs):
    keys, types = set(), set()
    for m in msgs:
        k = vvl_key(m)
        keys.add(k)
        if m["id"].startswith("SYNC-HAZARD-"):
            t = SV_TYPE.get(m["id"][len("SYNC-HAZARD-"):])
            if t:
                types.add((m["res"], t))
    return keys, types


def load_msgs(runs, mid, oracle):
    p = os.path.join(runs, mid, oracle, "vvl.txt")
    if not os.path.exists(p):
        return None
    return parse(open(p, errors="replace").read())


# ------------------------------------------------------------------ triage rules
def muts_of(rec):
    return [(s, k, f) for s, k, f in rec["muts"]]


def has(rec, site=None, kind=None):
    return any((site is None or s == site or (site.endswith("*") and s.startswith(site[:-1]))) and
               (kind is None or k == kind or (isinstance(kind, tuple) and k in kind))
               for s, k, _ in muts_of(rec))


def triage_cov_only(rec, key, cfg, ctx):
    cls, obj = key
    if cls == "host-race":
        return ("VAL-LIMIT", "host-write race on mapped memory: syncval does not track host accesses "
                "(VVL docs/syncval_usage.md, Known Limitations: 'Host memory accesses are not tracked')")
    if cls == "layout" and (has(rec, "present_wait", "drop") or has(rec, "signal_stage") or
                            has(rec, "b_sc_present") or has(rec, "acq_wait") or has(rec, "present")):
        return ("TECH", "same race reported a second time as a layout mismatch at the unordered present "
                "(COVENANT explores one order of the unordered queues for layout tracking)")
    if cls == "hazard" and (obj or "").startswith("swapchain[") and \
            any(k[0] in ("acq-sem", "sem-resignal") for k in ctx["vvl"]) and \
            any(k[0] in ("acq-sem", "sem-resignal") for k in ctx["cov"]):
        return ("TECH", "follow-on after a binary semaphore (acquire or render-finished) was signalled while a "
                "signal was pending (both tools report VUID-vkAcquireNextImageKHR-semaphore-01779 or "
                "VUID-vkQueueSubmit2-semaphore-03868): which signal a later wait consumes is then undefined; "
                "COVENANT pairs waits with signals FIFO (R-3), syncval differently, so the follow-on image "
                "race lands on a different image/frame")
    if cls == "signal-scope" and has(rec, "signal_stage", "none"):
        if ("hazard", ctx.get("img")) in ctx["vvl"] or any(k[0] == "hazard" for k in ctx["vvl"]):
            return ("TECH", "same root cause (empty signal stage mask): COVENANT names the semaphore "
                    "(signal-scope), syncval the image (PRESENT_AFTER_WRITE)")
        return ("TECH", "empty signal stage mask reported on the semaphore (signal-scope)")
    if has(rec, "present", "skip") and not rec["acq_match"] and cls in (
            "sem-resignal", "present-sem-reuse", "hazard", "layout") and (
            cls != "hazard" or (obj or "").startswith("swapchain[")):
        return ("DRIVER", "acquire order diverges after a skipped present: the model driver's documented "
                "round-robin acquire (model_driver/README: 'hands out images round-robin and never blocks') "
                "returns the image the application still holds, so its render semaphore is signalled again; "
                "lavapipe never returns a held image")
    if cls == "hazard" and any(k in ("dst_access_none",) for s, k, _ in muts_of(rec)
                               if s in ("b_sbuf_war", "b_vbuf_war", "b_state")) and ctx["cov_type"].get(obj) == {"WAW"}:
        return ("MODEL", "write-after-write through a barrier whose dst access mask is empty: COVENANT "
                "requires the later write in the second access scope (spec: WAW needs a memory "
                "dependency), syncval only an execution dependency plus availability")
    if cls in ("sem-resignal", "acq-sem", "overacquire", "present-sem-reuse", "cb-in-use", "fence-in-use",
               "hazard") and ctx["sv_truncated"]:
        return ("DRIVER", "lavapipe hung or crashed before this frame (present:skip over-acquire, or "
                "re-recording a pending command buffer), so VVL never saw it; the mock-ICD run "
                f"{'reports' if key in ctx['mock'] else 'does not report'} it")
    if cls == "hazard" and obj and obj.startswith("swapchain[") and (
            has(rec, "present_wait", "drop") or has(rec, "render_sem") or has(rec, "present", "skip")):
        return ("MODEL", "swapchain-image race that VVL reports as a semaphore/present VUID rather than a "
                "SYNC hazard")
    if cls == "hazard" and ctx["cov_type"].get(obj) == {"WAW"} and not any(
            k == ("hazard", obj) for k in ctx["vvl"]):
        return ("MODEL", "write-after-write ordered only by an execution dependency (here a semaphore chain "
                "plus a barrier on another buffer, or an empty dst access mask): COVENANT requires the later "
                "write in a second access scope (spec: WAW needs a memory dependency), syncval accepts "
                "execution dependency plus availability")
    if cls in ("cb-in-use", "fence-in-use") and any(k[0] == "timeline-value" for k in ctx["vvl"]) and \
            any(k[0] == "timeline-value" for k in ctx["cov"]):
        return ("TECH", "follow-on of a non-increasing timeline signal (both report VUID-VkSubmitInfo2-semaphore-03882): "
                "a later wait for that value resolves to the earliest matching signal (as syncval's own "
                "OnTimelineWait does), so COVENANT cannot prove the command buffer's previous submission "
                "complete; VVL core's in-use tracking retires it from observed completion")
    if cls == "qfot" and (has(rec, "qfot_acq") or has(rec, "qfot_rel")):
        return ("VAL-LIMIT", "EXCLUSIVE buffer read on the graphics family without the acquire half of the "
                "ownership transfer (contents undefined: spec 'Queue Family Ownership Transfer'); VVL only "
                "checks that an acquire matches a pending release (VUID-vkQueueSubmit2-commandBuffer-03879), "
                "it does not track which family owns a resource, and syncval does not model ownership")
    return None


def triage_vvl_only(rec, key, cfg, ctx):
    cls, obj = key
    if cls == "acquire-not-waited":
        if ("hazard", obj) in ctx["cov"]:
            return ("TECH", "VVL core reports the unwaited acquire as its own message; COVENANT reports "
                    "the resulting race on the same image")
        return ("MODEL", "first use of a never-presented swapchain image without waiting for the acquire: "
                "COVENANT adds a presentation-engine read only after a present (by design, rule R-1); "
                "WSI asks for the wait only until presentation reads complete")
    if cls == "hazard" and obj in ctx["vvl_acq_haz"] and (obj or "").startswith("swapchain["):
        return ("MODEL", "layout transition not ordered after the acquire of an image that has not been "
                "presented yet: syncval models the acquire itself as an access, COVENANT adds a "
                "presentation-engine read only after a present (rule R-1)")
    if cls == "hazard" and (obj or "").startswith("swapchain[") and \
            any(k[0] in ("acq-sem", "sem-resignal") for k in ctx["vvl"]) and \
            any(k[0] in ("acq-sem", "sem-resignal") for k in ctx["cov"]):
        return ("TECH", "follow-on after a binary semaphore (acquire or render-finished) was signalled while a "
                "signal was pending (both tools report VUID-vkAcquireNextImageKHR-semaphore-01779 or "
                "VUID-vkQueueSubmit2-semaphore-03868): which signal a later wait consumes is then undefined; "
                "COVENANT pairs waits with signals FIFO (R-3), syncval differently, so the follow-on image "
                "race lands on a different image/frame")
    if cls == "overacquire" and cfg != "W2-MQ":
        return ("DRIVER", "surface minImageCount differs: lavapipe/X11 reports 3 (1 image may be held), "
                "the model driver 2 (2 may be held), so VVL flags VUID-vkAcquireNextImageKHR-surface-07783 "
                "one acquire earlier")
    if ctx["sv_truncated"] and key not in ctx["mock"] and cfg != "W2-MQ":
        return ("DRIVER", "only in the truncated lavapipe run")
    if cfg == "W2-MQ" and cls == "hazard" and ctx["base_types"] and \
            ctx["vvl_type"].get(obj, set()) <= ctx["base_types"].get(obj, set()):
        return ("VAL-LIMIT", "baseline syncval false positive (round-trip semaphore chain "
                "compute->graphics->compute not credited), see W2-MQ baseline")
    return None


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutants", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = {}
    for line in open(a.results):
        r = json.loads(line)
        R[r["id"]] = r
    M = json.load(open(a.mutants))
    rows = []
    base = {}
    for m in M:
        r = R.get(m["id"])
        if r is None:
            continue
        cfg = m["config"]
        ref_oracle = "mock" if cfg == "W2-MQ" else "sv"
        msgs = load_msgs(a.runs, m["id"], ref_oracle) or []
        mock_msgs = load_msgs(a.runs, m["id"], "mock") or []
        vk, vt = vvl_keys(msgs)
        mk, _ = vvl_keys(mock_msgs)
        sk, st = cov_keys(r["cov"]["snap"])
        fk, ft = cov_keys(r["cov"]["fix"])
        rc = r.get(ref_oracle, {}).get("rc")
        row = {"id": m["id"], "config": cfg, "gen": m["gen"], "mut": m["mut"], "muts": m["muts"],
               "ref": ref_oracle, "ref_rc": rc, "sv_truncated": ref_oracle == "sv" and rc not in (0, None),
               "invalid": any(k[0] == "invalid-mutant" for k in vk),
               "vvl": sorted(vk, key=str), "vvl_types": sorted(vt, key=str), "mock": sorted(mk, key=str),
               "snap": sorted(sk, key=str), "fix": sorted(fk, key=str), "snap_types": sorted(st, key=str),
               "fix_types": sorted(ft, key=str),
               "acq_match": r["cov"]["acq"] == r.get(ref_oracle, {}).get("acq"),
               "applied_match": sorted(r["cov"]["applied"]) == sorted(r.get(ref_oracle, {}).get("applied", [])),
               "t_cov_check": r["cov"]["fix"].get("t_load", 0) + r["cov"]["fix"].get("t_check", 0),
               "t_cov_check_snap": r["cov"]["snap"].get("t_load", 0) + r["cov"]["snap"].get("t_check", 0),
               "t_cov_wall": r["cov"]["fix"].get("wall"), "t_cov_run": r["cov"]["t"],
               "t_sv": r.get("sv", {}).get("t"), "t_mock": r.get("mock", {}).get("t"),
               "vvl_acq_haz": sorted({x["res"] for x in msgs if x["id"].startswith("SYNC-HAZARD") and
                                      "previously accessed by vkAcquireNextImageKHR" in x["text"]}, key=str),
               "n_vvl_msgs": len(msgs), "n_cov_diags": len(r["cov"]["fix"].get("diags", []))}
        rows.append(row)
        if m["gen"] == "baseline":
            base[cfg] = row
    # baseline-relative view for W2-MQ (the mock+VVL baseline is not clean)
    for row in rows:
        b = base.get(row["config"])
        row["base_vvl"] = [list(k) for k in (b["vvl"] if b else [])]
    out = summarize(rows, base)
    json.dump(out, open(a.out, "w"), indent=1, default=list)
    print_report(out)


def as_set(lst):
    return {tuple(x) for x in lst}


def verdicts(row, side, ref_keys, strict_sync=False):
    keys = as_set(row[side])
    return bool(keys)


def summarize(rows, base):
    out = {"rows": rows, "tables": {}, "triage": {}, "unexplained": []}
    cfgs = ["W1-F2", "W1-F3", "W2", "W3", "W2-MQ"]
    tri_counts = collections.Counter()
    tri_examples = collections.defaultdict(list)
    for row in rows:
        cfg = row["config"]
        bv = as_set(base[cfg]["vvl"]) if cfg == "W2-MQ" else set()
        btypes = as_set(base[cfg]["vvl_types"]) if cfg == "W2-MQ" else set()
        vvl = as_set(row["vvl"]) - bv if cfg == "W2-MQ" else as_set(row["vvl"])
        # W2-MQ: a hazard key also present in the baseline still counts when its hazard type is new
        if cfg == "W2-MQ":
            newt = as_set(row["vvl_types"]) - btypes
            for res, t in newt:
                vvl.add(("hazard", res))
        row["vvl_eff"] = sorted(vvl, key=str)
        sync = {k for k in vvl if k[0] == "hazard"}
        row["sync_eff"] = sorted(sync, key=str)
        for side in ("snap", "fix"):
            cov = as_set(row[side])
            row[f"{side}_cov_only"] = sorted(cov - vvl, key=str)
            row[f"{side}_vvl_only"] = sorted(vvl - cov, key=str)
        # triage of the fix-side disagreements (and of the snap side, for the before/after view)
        ctx = {"vvl": vvl, "mock": as_set(row["mock"]), "sv_truncated": row["sv_truncated"],
               "vvl_acq_haz": set(row["vvl_acq_haz"]),
               "base_types": collections.defaultdict(set)}
        for res, t in btypes:
            ctx["base_types"][res].add(t)
        ctx["vvl_type"] = collections.defaultdict(set)
        for res, t in as_set(row["vvl_types"]):
            ctx["vvl_type"][res].add(t)
        for side in ("snap", "fix"):
            ctx["cov"] = as_set(row[side])
            ctx["cov_type"] = collections.defaultdict(set)
            for res, t in as_set(row[f"{side}_types"]):
                ctx["cov_type"][res].add(t)
            tri = []
            for k in row[f"{side}_cov_only"]:
                k = tuple(k)
                t = triage_cov_only(row, k, cfg, ctx) or ("UNEXPLAINED", "")
                tri.append({"side": "cov-only", "key": k, "class": t[0], "why": t[1]})
            for k in row[f"{side}_vvl_only"]:
                k = tuple(k)
                t = triage_vvl_only(row, k, cfg, ctx) or triage_known_bug(row, k, side) or ("UNEXPLAINED", "")
                tri.append({"side": "vvl-only", "key": k, "class": t[0], "why": t[1]})
            row[f"{side}_triage"] = tri
            for t in tri:
                tri_counts[(side, cfg, t["side"], t["class"], t["why"][:90])] += 1
                if len(tri_examples[(side, t["side"], t["class"], t["why"][:90])]) < 3:
                    tri_examples[(side, t["side"], t["class"], t["why"][:90])].append(row["id"])
                if t["class"] == "UNEXPLAINED":
                    out["unexplained"].append((side, row["id"], row["mut"], t["side"], t["key"]))
    # mutation kinds: how many mutants contain each (site, kind), per configuration, and for the
    # single-site mutants the verdict agreement (any finding / any hazard) of each side
    kinds = collections.defaultdict(lambda: collections.Counter())
    for row in rows:
        for s_, k_, f_ in row["muts"]:
            kinds[f"{s_}:{k_}"][row["config"]] += 1
            kinds[f"{s_}:{k_}"]["all"] += 1
        if len(row["muts"]) == 1:
            s_, k_, f_ = row["muts"][0]
            for side in ("snap", "fix"):
                cv, vv = bool(row[side]), bool(row["vvl_eff"])
                kinds[f"{s_}:{k_}"][f"single_{side}_" + ("both" if cv and vv else "cov_only" if cv else
                                                         "ref_only" if vv else "neither")] += 1
                kinds[f"{s_}:{k_}"][f"single_{side}_exact"] += set(map(tuple, row[side])) == set(map(tuple, row["vvl_eff"]))
            kinds[f"{s_}:{k_}"]["single"] += 1
    out["kinds"] = {k: dict(v) for k, v in sorted(kinds.items())}
    out["triage_counts"] = [list(k) + [v] for k, v in sorted(tri_counts.items())]
    out["triage_examples"] = [list(k) + [v] for k, v in tri_examples.items()]
    # tables
    for cfg in cfgs + ["ALL-SQ"]:
        rs = [r for r in rows if (r["config"] == cfg or (cfg == "ALL-SQ" and r["config"] != "W2-MQ"))
              and not r["invalid"]]
        t = {"n": len(rs), "invalid": sum(1 for r in rows if r["config"] == cfg and r["invalid"]),
             "truncated": sum(1 for r in rs if r["sv_truncated"])}
        for side in ("snap", "fix"):
            for refname, refkey in (("syncval", "sync_eff"), ("vvl", "vvl_eff")):
                c = collections.Counter()
                for r in rs:
                    cv = bool(r[side]) if refname == "vvl" else any(k[0] == "hazard" for k in map(tuple, r[side]))
                    vv = bool(r[refkey])
                    c[("both" if cv and vv else "cov_only" if cv else "ref_only" if vv else "neither")] += 1
                t[f"{side}_vs_{refname}"] = dict(c)
            # resource level (hazard class only): (mutant, resource) pairs
            c = collections.Counter()
            for r in rs:
                cv = {k[1] for k in map(tuple, r[side]) if k[0] == "hazard"}
                vv = {k[1] for k in map(tuple, r["sync_eff"])}
                c["both"] += len(cv & vv)
                c["cov_only"] += len(cv - vv)
                c["ref_only"] += len(vv - cv)
                c["host_race"] += len({k[1] for k in map(tuple, r[side]) if k[0] == "host-race"})
            t[f"{side}_resource"] = dict(c)
            # all classes, (mutant, key) pairs
            c = collections.Counter()
            for r in rs:
                cv, vv = as_set(r[side]), as_set(r["vvl_eff"])
                c["both"] += len(cv & vv)
                c["cov_only"] += len(cv - vv)
                c["ref_only"] += len(vv - cv)
            t[f"{side}_keys"] = dict(c)
        out["tables"][cfg] = t
    return out


def triage_known_bug(row, key, side):
    """VVL-only findings explained by a checker bug fixed in checker.patch (snap side)."""
    cls, obj = key
    if side != "snap":
        return None
    if cls == "hazard" and has(row, "signal_stage", ("wrong", "late", "none")):
        return ("CHECKER-BUG", "fix 5: semaphore signal stage mask ignored (sync2 VkSemaphoreSubmitInfo::stageMask)")
    if cls == "hazard" and any(k in ("dst_wrong", "src_wrong") for s, k, _ in muts_of(row)
                               if s in ("b_off_src", "b_sc_dst", "b_sc_present", "b_vbuf_war", "b_vbuf_raw")):
        return ("CHECKER-BUG", "fix 4: sync2 transfer sub-stages (COPY/BLIT/CLEAR) conflated with TRANSFER")
    if cls in ("timeline-value",) or (cls == "hazard" and has(row, "tl_comp_signal", "value_next")):
        return ("CHECKER-BUG", "fix 3: timeline signal value not increasing (no rule; waits aliased to the latest signal)")
    if cls in ("cb-in-use", "fence-in-use", "acq-sem"):
        return ("CHECKER-BUG", "fix 6: in-use rules (command buffer / fence / acquire semaphore reused while pending)")
    if cls in ("sem-resignal",) or (cls == "hazard" and (has(row, "acq_wait", "drop") or has(row, "render_sem") or
                                                         has(row, "acq_sem") or has(row, "present_wait")) and
                                    tuple(key) in as_set(row["fix"])):
        return ("CHECKER-BUG", "fix 1: binary semaphore signalled while a signal is pending")
    return None


def print_report(out):
    for cfg, t in out["tables"].items():
        print(cfg, json.dumps(t))
    print("triage:")
    for row in out["triage_counts"]:
        print("  ", row)
    print("unexplained:", len(out["unexplained"]))
    for u in out["unexplained"][:80]:
        print("  ", u)


if __name__ == "__main__":
    main()
