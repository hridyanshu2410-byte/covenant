"""Confusion matrices, disagreement triage and timing for the difftest run.

    python3 bench/elide/difftest/analyze.py --traces DIR --mutants DIR --results results.jsonl --out DIR

Writes <out>/results.json (every mutant, both verdicts, triage) and <out>/RESULTS.md (a summary table).
"""
import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from covenant import elide  # noqa: E402
import timing_fixed as tf  # noqa: E402
from mutants import WORKLOADS, act_from_json  # noqa: E402

OBJ_RE = re.compile(r"Hazard (\w+) for (?:\w+ )?(Vk\w+) 0x[0-9a-fA-F]+\[([^\]]*)\]")
CELLS = ["both safe", "both hazard", "checker only", "syncval only"]


def cell(ck, sv):
    return {(False, False): "both safe", (True, True): "both hazard", (True, False): "checker only",
            (False, True): "syncval only"}[(ck, sv)]


NAME_RE = re.compile(r"name = ([^,;]+), type = VK_OBJECT_TYPE_(\w+)")


def sv_res(h):
    obj = h.get("obj")
    kind = h.get("hazard") or h["id"].replace("SYNC-HAZARD-", "").replace("-", "_")
    if obj is None:
        m = OBJ_RE.search(h.get("text", ""))
        if m:
            kind, obj = m.group(1), m.group(3)
        else:
            for name, typ in NAME_RE.findall(h.get("text", "")):
                if typ in ("IMAGE", "IMAGE_VIEW", "BUFFER"):
                    obj = name
                    break
    if obj and obj.endswith("_view"):
        obj = obj[:-5]
    return obj, kind


def buffer_names(trace):
    names = set()
    for line in open(trace):
        r = json.loads(line)
        if r.get("fn") == "vkSetDebugUtilsObjectNameEXT" and r.get("object_type") == "VK_OBJECT_TYPE_BUFFER":
            names.add(r["name"])
    return names


def earlier_where(d):
    for n in d.get("notes", []):
        if n.startswith("earlier access: ") or n.startswith("other access: "):
            return n.split(": ", 1)[1]
    return ""


def is_sb_blind(d, buffers):
    """A checker hazard between two shader (dispatch/draw) accesses of a storage buffer."""
    return d["res"] in buffers and d["where"].startswith(("vkCmdDispatch", "vkCmdDraw")) and \
        earlier_where(d).startswith(("vkCmdDispatch", "vkCmdDraw"))


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--mutants", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.results)]
    controls = {r["id"]: r for r in rows if r.get("control")}
    muts = [r for r in rows if not r.get("control")]
    out = []
    els, manifests, bufs, fixed_base = {}, {}, {}, {}
    for w in WORKLOADS:
        el = elide.Elider(os.path.join(a.traces, f"{w}.jsonl"), os.path.join(a.traces, f"{w}-spv"))
        el.base = set(el.hazards({})[0])
        els[w] = el
        manifests[w] = {m["id"]: m for m in json.loads((Path(a.mutants) / w / "manifest.json").read_text())}
        bufs[w] = buffer_names(os.path.join(a.traces, f"{w}.jsonl"))
        fixed_base[w] = {}
        for fa, fs in ((True, False), (False, True), (True, True)):
            tf.FIX_ACCESS_SCOPE, tf.FIX_SAME_CALL = fa, fs
            fixed_base[w][(fa, fs)] = set(tf.hazards(el, {}))
    for r in muts:
        w = r["workload"]
        el = els[w]
        m = manifests[w][r["id"]]
        plan = {tuple(int(x) for x in k.split(",")): {l: act_from_json(v) for l, v in acts.items()}
                for k, acts in m["internal"].items()}
        sv = r["syncval"]
        base = controls[f"{w}-base"]["syncval"]
        lay = sv.get("layer") or {}
        valid = sv.get("returncode") == 0 and not sv.get("other") and lay.get("matched") == r["calls_listed"] \
            and lay.get("sig_mismatch") == 0
        ck_new = r["checker"]["new"]
        ck = bool(ck_new)
        s = bool(sv.get("sync"))
        fixed = {}
        for fa, fs in ((True, False), (False, True), (True, True)):
            tf.FIX_ACCESS_SCOPE, tf.FIX_SAME_CALL = fa, fs
            h = tf.hazards(el, plan)
            fixed[f"{'A' if fa else ''}{'B' if fs else ''}"] = sorted(
                {(k[0], (re.search(r"'([^']+)'", k[2]) or [None, None])[1]) for k in set(h) - fixed_base[w][(fa, fs)]})
        sres = sorted({sv_res(h) for h in sv.get("sync", [])}, key=str)
        cres = sorted({d["res"] for d in ck_new if d["res"]})
        rec = OrderedDict(id=r["id"], workload=w, p_remove=r["p_remove"], p_narrow=r["p_narrow"],
                          calls_mutated=r["calls_listed"], calls_removed=r["calls_removed"],
                          structs_removed=r["structs_removed"], structs_narrowed=r["structs_narrowed"],
                          valid=valid, checker_hazard=ck, syncval_hazard=s, cell=cell(ck, s),
                          checker_new=[{k: d[k] for k in ("code", "res", "where")} | {"earlier": earlier_where(d)[:120]}
                                       for d in ck_new],
                          checker_raw_new=len(r["checker"]["raw_new"]),
                          syncval_count=len(sv.get("sync", [])),
                          syncval_kinds=dict(Counter(sv_res(h)[1] for h in sv.get("sync", []))),
                          syncval_resources=[x[0] for x in sres],
                          syncval_hazards=[{k: h.get(k) for k in ("hazard", "obj", "usage", "prior_usage", "cmd",
                                                                   "prior_cmd")} | ({"text": h["text"][:400]} if "text" in h else {})
                                           for h in sv.get("sync", [])][:12],
                          other_validation=[o["id"] for o in sv.get("other", [])][:5],
                          hashes_same_as_base=sv.get("hashes") == base.get("hashes"),
                          layer=lay, fixed_oracle=fixed,
                          checker_seconds=r["checker"]["seconds"], syncval_seconds=sv["seconds"])
        # resource-level comparison (on the hazards both report)
        svr = {x[0] for x in sres}
        rec["syncval_resources_not_in_checker"] = sorted(map(str, svr - set(cres)))
        fres = {x[1] for x in fixed["AB"] if x[1]}
        rec["resource_cells"] = {str(x): cell(x in cres, x in svr) for x in set(cres) | svr}
        rec["resource_cells_fixed"] = {str(x): cell(x in fres, x in svr) for x in fres | svr}
        def expl(x):
            ina, inb = x in {y[1] for y in fixed["A"]}, x in {y[1] for y in fixed["B"]}
            if ina and inb:
                return "A or B (either fix alone)"
            if ina:
                return "A (access scope)"
            if inb:
                return "B (same-call chaining)"
            return "A+B (needs both)" if x in fres else "unexplained"
        rec["syncval_only_resources_explained"] = {str(x): expl(x) for x in svr - set(cres)}
        pjson = json.loads((Path(a.mutants) / m["plan"]).read_text())
        rec["triage"] = triage(rec, ck_new, bufs[w], pjson)
        out.append(rec)
    summary = summarize(out, controls)
    res = OrderedDict(description="Differential test: COVENANT timing checker (elide.py's strengthened frame + "
                                  "timing.check + transition_reach) vs Khronos syncval 1.3.275 on lavapipe, on random "
                                  "mutant barrier plans", summary=summary, controls=controls, mutants=out)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "results.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


CK_WHY = {
    "syncval-blind-storage-buffer": "real hazard syncval misses: a race between two shader accesses of a storage "
                                    "buffer; syncval 1.3.275 classifies these storage-buffer descriptor accesses as "
                                    "reads (bin's atomicAdd on 'histogram' is reported as SHADER_STORAGE_READ, and "
                                    "integrate's OpStore to particlesN is never seen as a write)",
    "syncval-all-graphics-covers-copy": "real hazard (by the spec) syncval misses: the readback transition's dst stage "
                                        "is ALL_GRAPHICS and the same call's TRANSFER memory barrier was removed; "
                                        "ALL_GRAPHICS does not contain the COPY/TRANSFER stage of the later "
                                        "vkCmdCopyImageToBuffer, but syncval 1.3.275 accepts it (with FRAGMENT_SHADER "
                                        "or COMPUTE_SHADER instead, syncval reports the RAW)",
    "checker-conservative-waw-proxy": "checker false positive (conservative): elide's WAW strengthening checks the "
                                      "later write's 'needs visibility' proxy as a READ, so a transition whose "
                                      "dstAccessMask has MEMORY_WRITE (or another write bit) but no read bit is "
                                      "rejected; Vulkan and syncval only need the write in the dst access scope",
}


def ck_resource_cause(res, ck_new, buffers, plan_json):
    ds = [d for d in ck_new if d["res"] == res]
    if ds and all(is_sb_blind(d, buffers) for d in ds):
        return "syncval-blind-storage-buffer"
    if ds and all(d["where"].endswith("[the write needs visibility]") for d in ds):
        return "checker-conservative-waw-proxy"
    if ds and all(d["where"].startswith("vkCmdCopy") for d in ds):
        for b in plan_json["barriers"]:
            for e in b["elements"]:
                if e.get("image") == res and e["action"] == "narrow" and "ALL_GRAPHICS" in e["new"]["dst_stage"] \
                        and "TRANSFER" not in e["new"]["dst_stage"] and "ALL_COMMANDS" not in e["new"]["dst_stage"]:
                    return "syncval-all-graphics-covers-copy"
    return "UNTRIAGED"


def triage(rec, ck_new, buffers, plan_json=None):
    c = rec["cell"]
    fx = rec["fixed_oracle"]
    rc = {res: ck_resource_cause(res, ck_new, buffers, plan_json)
          for res, cl in rec["resource_cells"].items() if cl == "checker only"}
    rec["checker_only_resource_causes"] = rc
    if c == "checker only":
        causes = sorted(set(rc.values()))
        cause = "+".join(causes) if causes else "UNTRIAGED-checker-only"
        return {"cause": cause, "why": " / ".join(CK_WHY.get(x, "untriaged") for x in causes)}
    if c == "syncval only":
        a_, b_, ab = bool(fx["A"]), bool(fx["B"]), bool(fx["AB"])
        if a_ and b_:
            cause = "checker-bug-access-scope+same-call"
        elif a_:
            cause = "checker-bug-access-scope"
        elif b_:
            cause = "checker-bug-same-call-chaining"
        elif ab:
            cause = "checker-bug-both-needed"
        else:
            cause = "UNTRIAGED-syncval-only"
        why = {"checker-bug-access-scope": "timing.py extends ACCESS scopes to logically earlier/later stages and "
                                           "TOP/BOTTOM_OF_PIPE (it uses the execution-scope test _stage_in for "
                                           "availability/visibility); the fixed oracle (explicit access scopes) "
                                           "reports the hazard syncval reports",
               "checker-bug-same-call-chaining": "elide.build lays one call's structs out as consecutive barriers, so "
                                                 "another struct of the same vkCmdPipelineBarrier2 (e.g. its global "
                                                 "memory barrier) captures/chains from an image struct's layout "
                                                 "transition; the fixed oracle (no same-call chaining) reports it",
               "checker-bug-access-scope+same-call": "both suspected bugs independently reproduce the hazard",
               "checker-bug-both-needed": "only the oracle with both fixes reports it"}.get(cause, "")
        return {"cause": cause, "why": why}
    return None


def summarize(out, controls):
    S = OrderedDict()
    for scope in WORKLOADS + ["overall"]:
        rs = [r for r in out if scope == "overall" or r["workload"] == scope]
        valid = [r for r in rs if r["valid"]]
        cm = Counter(r["cell"] for r in valid)
        tri = Counter(r["triage"]["cause"] for r in valid if r["triage"])
        # what the fixed oracle would give
        cm_fixed = Counter(cell(bool(r["fixed_oracle"]["AB"]), r["syncval_hazard"]) for r in valid)
        rcm = Counter(c for r in valid for c in r["resource_cells"].values())
        rcmf = Counter(c for r in valid for c in r["resource_cells_fixed"].values())
        expl = Counter(v for r in valid for v in r["syncval_only_resources_explained"].values())
        ckc = Counter(v for r in valid for v in r["checker_only_resource_causes"].values())
        cs = [r["checker_seconds"] for r in rs]
        ss = [r["syncval_seconds"] for r in rs]
        both = [r for r in valid if r["cell"] == "both hazard"]
        S[scope] = OrderedDict(
            mutants=len(rs), valid=len(valid), invalid=[r["id"] for r in rs if not r["valid"]],
            confusion={k: cm.get(k, 0) for k in CELLS},
            agreement=round((cm["both safe"] + cm["both hazard"]) / max(1, len(valid)), 4),
            triage=dict(tri),
            confusion_fixed_oracle={k: cm_fixed.get(k, 0) for k in CELLS},
            resource_level={k: rcm.get(k, 0) for k in CELLS[1:]},
            resource_level_fixed_oracle={k: rcmf.get(k, 0) for k in CELLS[1:]},
            syncval_only_resources_explained_by=dict(expl),
            checker_only_resources_cause=dict(ckc),
            both_hazard_resource_mismatch=sum(1 for r in both if r["syncval_resources_not_in_checker"]),
            hashes_differ=sum(1 for r in rs if not r["hashes_same_as_base"]),
            hashes_differ_when_checker_safe=sum(1 for r in rs if not r["hashes_same_as_base"] and not r["checker_hazard"]),
            mean_calls_mutated=round(statistics.mean(r["calls_mutated"] for r in rs), 2),
            mean_structs_removed=round(statistics.mean(r["structs_removed"] for r in rs), 2),
            mean_structs_narrowed=round(statistics.mean(r["structs_narrowed"] for r in rs), 2),
            checker_seconds=dict(median=statistics.median(cs), mean=statistics.mean(cs), p95=pct(cs, 0.95),
                                 max=max(cs)),
            syncval_seconds=dict(median=statistics.median(ss), mean=statistics.mean(ss), p95=pct(ss, 0.95),
                                 max=max(ss)),
            speedup_median=round(statistics.median(ss) / statistics.median(cs), 1))
    S["controls"] = {k: {"syncval_hazards": len(v["syncval"].get("sync", [])), "other": len(v["syncval"].get("other", [])),
                         "seconds_with_validation": v["syncval"]["seconds"],
                         "seconds_without_validation": v["novalidation"]["seconds"],
                         "checker_new": len(v["checker"]["new"])} for k, v in controls.items()}
    return S


if __name__ == "__main__":
    main()
