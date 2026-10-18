#!/usr/bin/env python3
"""Score a COVENANT check of the whole-frame traces against the STORED Khronos-validation results
(validation is not re-run), and compare it per program with the stored COVENANT results.

    python3 score.py --base BASE --cov CHECK.jsonl --name NAME --out OUTDIR
                     [--compare NAME=OTHER.jsonl ...]
    (or give --mutants / --stored / --runs-meta instead of --base)

BASE is the extracted traces archive. It holds inputs/mutants.json (the 1,251 programs),
inputs/raw_results.jsonl.gz (per program: the stored validation results, lavapipe + VVL `sv` and
mock ICD + VVL `mock`, with their messages already parsed by vvlparse.parse, and the stored COVENANT
results `cov.snap` / `cov.fix` of the original prototype checker before and after its bug fixes)
and runs.jsonl (the model-driver run records: rc, acquire order, applied mutations). CHECK.jsonl is
check_cov.py batch output. --runs-meta may point at a fresh runner.py output instead.

Method: analyze.load_msgs is replaced by a lookup in the stored, parsed messages (the raw vvl.txt
logs are not archived); the stored `snap` results stay on analyze.py's "snap" side and the new
results go on its "fix" side; then analyze.py's own main() runs unchanged (normalisation, W2-MQ
baseline-relative view, tables and triage rules). It runs twice: as is, and "+afu"
(timing.acquire-first-use warnings counted as an error-severity wait-stage-scope hazard). A first
run puts the stored `snap`/`fix` results on both sides and must reproduce the stored tables
(self-check; the script aborts if it does not).

Writes OUTDIR/analysis_<name>.json, analysis_<name>+afu.json, tables_<name>.json,
changes_<name>.json and tables_<name>.md (also printed).
"""
import argparse
import collections
import contextlib
import copy
import gzip
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze  # noqa: E402

CFGS = ["W1-F2", "W1-F3", "W2", "W3", "W2-MQ"]


def load_jsonl(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as f:
        return {r["id"]: r for r in map(json.loads, f)}


def to_chk(r):
    """check_cov.py record -> the dict analyze.cov_keys reads."""
    if r.get("error"):
        raise SystemExit(f"{r['id']}: checker error {r['error']} -- fix before scoring")
    return {"t_load": r["t_load"], "t_check": r["t_check"], "wall": r.get("t_wall"),
            "diags": [{k: d.get(k) for k in ("code", "res", "where", "message", "notes", "severity")}
                      for d in r["diagnostics"]],
            "hazards": r["hazards"], "warnings": r.get("warnings", []), "unsupported": r.get("unsupported", {})}


def run_analyze(raw, mutants_path, snap_of, fix_of, runs_meta, afu, tmpdir, tag):
    """analyze.main() on a synthetic results.jsonl; returns its output dict."""
    rp = os.path.join(tmpdir, f"_results_{tag}.jsonl")
    with open(rp, "w") as f:
        for mid, r in raw.items():
            r = copy.deepcopy(r)
            if runs_meta is not None:
                t = runs_meta[mid]
                r["cov"].update(rc=t["rc"], t=t["t"], acq=t["acq"], applied=t["applied"], other=t["other"])
            for side, get in (("snap", snap_of), ("fix", fix_of)):
                x = copy.deepcopy(get(mid))
                if afu:
                    for d in x.get("diags", []):
                        if d["code"] == "timing.acquire-first-use":
                            d["code"], d["severity"] = "timing.wait-stage-scope", "error"
                r["cov"][side] = x
            f.write(json.dumps(r) + "\n")
    op = os.path.join(tmpdir, f"_analysis_{tag}.json")
    analyze.load_msgs = lambda runs, mid, oracle: (raw[mid].get(oracle) or {}).get("msgs")
    sys.argv = ["analyze.py", "--mutants", mutants_path, "--results", rp, "--runs", "/nonexistent", "--out", op]
    with contextlib.redirect_stdout(io.StringIO()):
        analyze.main()
    res = json.load(open(op))
    os.remove(rp)
    os.remove(op)
    return res


def summary(res, side):
    """Over all programs (W2-MQ baseline-relative): verdict agreement
    (hazards only / all classes) and (mutant, key) pair counts."""
    out = {}
    for kind in ("hazards", "all"):
        c, p = collections.Counter(), collections.Counter()
        for r in res["rows"]:
            if kind == "hazards":
                cv = any(k[0] == "hazard" for k in map(tuple, r[side])); vv = bool(r["sync_eff"])
                ck = {k[1] for k in map(tuple, r[side]) if k[0] == "hazard"}
                vk = {k[1] for k in map(tuple, r["sync_eff"])}
            else:
                cv = bool(r[side]); vv = bool(r["vvl_eff"])
                ck = {tuple(k) for k in r[side]}; vk = {tuple(k) for k in r["vvl_eff"]}
            c["both" if cv and vv else "cov-only" if cv else "vvl-only" if vv else "neither"] += 1
            p["both"] += len(ck & vk); p["cov-only"] += len(ck - vk); p["vvl-only"] += len(vk - ck)
        n = sum(c.values())
        out[kind] = {"agreement_pct": round(100 * (c["both"] + c["neither"]) / n, 1), "n": n,
                     "verdicts": dict(c), "pairs": dict(p)}
    return out


def triage_table(res, side):
    t = collections.Counter()
    for r in res["rows"]:
        for x in r[f"{side}_triage"]:
            t[(x["side"], x["class"], x["why"][:90], r["config"])] += 1
    rows = collections.defaultdict(lambda: collections.Counter())
    for (s, c, w, cfg), v in t.items():
        rows[(s, c, w)][cfg] += v
    return [{"direction": s, "class": c, "reason": w, **{k: v[k] for k in CFGS}, "total": sum(v.values())}
            for (s, c, w), v in sorted(rows.items(), key=lambda kv: (kv[0][0], -sum(kv[1].values())))]


def dset(diags, full=True, warn=False):
    return collections.Counter((d["code"], d.get("res"), d.get("where"), d.get("message") if full else "")
                               for d in diags if warn or d.get("severity", "error") != "warning")


def per_mutant(mutants, old_of, new_of, oldname, newname):
    """Verdict / key / diagnostic changes from old to new (both chk dicts) per mutant."""
    out = {"from": oldname, "to": newname, "verdict_changed_all": [], "verdict_changed_hazard": [],
           "keys_changed": [], "diag_changed_codewhere": {}, "diag_text_only": 0, "warn_changed": 0,
           "diag_delta_by_code": {}}
    delta = collections.Counter()
    for m in mutants:
        i = m["id"]
        a, b = old_of(i), new_of(i)
        ka, _ = analyze.cov_keys(a); kb, _ = analyze.cov_keys(b)
        if bool(ka) != bool(kb):
            out["verdict_changed_all"].append({"id": i, "mut": m["mut"], "old": bool(ka), "new": bool(kb)})
        ha, hb = any(k[0] == "hazard" for k in ka), any(k[0] == "hazard" for k in kb)
        if ha != hb:
            out["verdict_changed_hazard"].append({"id": i, "mut": m["mut"], "old": ha, "new": hb})
        if ka != kb:
            out["keys_changed"].append({"id": i, "mut": m["mut"], "lost": sorted(ka - kb, key=str),
                                        "gained": sorted(kb - ka, key=str)})
        da, db = dset(a["diags"], False), dset(b["diags"], False)
        if da != db:
            ch = {}
            for k, v in (da - db).items():
                ch.setdefault(f"lost {k[0]}", 0); ch[f"lost {k[0]}"] += v; delta[f"lost {k[0]}"] += v
            for k, v in (db - da).items():
                ch.setdefault(f"gained {k[0]}", 0); ch[f"gained {k[0]}"] += v; delta[f"gained {k[0]}"] += v
            out["diag_changed_codewhere"][i] = {"mut": m["mut"], **ch}
        elif dset(a["diags"]) != dset(b["diags"]):
            out["diag_text_only"] += 1
        if dset(a["diags"], False, True) - dset(a["diags"], False) != dset(b["diags"], False, True) - dset(b["diags"], False):
            out["warn_changed"] += 1
    out["diag_delta_by_code"] = dict(sorted(delta.items()))
    out["n_verdict_changed_all"] = len(out["verdict_changed_all"])
    out["n_verdict_changed_hazard"] = len(out["verdict_changed_hazard"])
    out["n_keys_changed"] = len(out["keys_changed"])
    out["n_diag_changed_codewhere"] = len(out["diag_changed_codewhere"])
    return out


def config_tables(res, side):
    t = {}
    for cfg in CFGS + ["ALL-SQ"]:
        x = res["tables"][cfg]
        t[cfg] = {k: x[f"{side}_{k}"] for k in ("vs_syncval", "vs_vvl", "resource", "keys")}
        t[cfg]["n"] = x["n"]
    return t


def md(tab, name):
    L = [f"## {name}", ""]
    for view in (name, f"{name}+afu"):
        s = tab[view]["summary"]
        L.append(f"**{view}** over all {s['hazards']['n']} programs: hazards-only verdict agreement "
                 f"{s['hazards']['agreement_pct']}% {s['hazards']['verdicts']}, all classes "
                 f"{s['all']['agreement_pct']}% {s['all']['verdicts']}; (mutant, finding) pairs all classes "
                 f"both/COVENANT-only/VVL-only = {s['all']['pairs'].get('both', 0)} / "
                 f"{s['all']['pairs'].get('cov-only', 0)} / {s['all']['pairs'].get('vvl-only', 0)}; hazard "
                 f"pairs {s['hazards']['pairs'].get('both', 0)} / {s['hazards']['pairs'].get('cov-only', 0)} / "
                 f"{s['hazards']['pairs'].get('vvl-only', 0)}")
        L.append("")
    L += ["| config | n | hazards: both / neither / COV only / VVL only | agree | all: both / neither / COV only / VVL only | agree |",
          "|---|---|---|---|---|---|"]
    for cfg, x in tab[name]["configs"].items():
        h, a = x["vs_syncval"], x["vs_vvl"]
        def f(c):
            return f"{c.get('both', 0)} / {c.get('neither', 0)} / {c.get('cov_only', 0)} / {c.get('ref_only', 0)}"
        def ag(c):
            n = sum(c.values())
            return f"{100 * (c.get('both', 0) + c.get('neither', 0)) / n:.1f}%"
        L.append(f"| {cfg} | {x['n']} | {f(h)} | {ag(h)} | {f(a)} | {ag(a)} |")
    L += ["", "| direction | class | reason | " + " | ".join(CFGS) + " | total |",
          "|---|---|---|" + "---|" * (len(CFGS) + 1)]
    for r in tab[name]["triage"]:
        L.append(f"| {r['direction']} | {r['class']} | {r['reason']} | " +
                 " | ".join(str(r[c]) for c in CFGS) + f" | {r['total']} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="extracted traces archive (sets the defaults of the next three)")
    ap.add_argument("--mutants", help="mutants.json (default BASE/inputs/mutants.json)")
    ap.add_argument("--stored", help="stored results, raw_results.jsonl.gz (default BASE/inputs/...)")
    ap.add_argument("--runs-meta", help="run records, runner.py --out (default BASE/runs.jsonl)")
    ap.add_argument("--cov", required=True, help="check_cov.py batch output to score")
    ap.add_argument("--name", required=True, help="label for the outputs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", action="append", default=[],
                    help="NAME=PATH of another check_cov.py output to compare per program (repeatable)")
    a = ap.parse_args()
    for k, rel in (("mutants", ("inputs", "mutants.json")), ("stored", ("inputs", "raw_results.jsonl.gz")),
                   ("runs_meta", ("runs.jsonl",))):
        if getattr(a, k) is None:
            if a.base is None:
                ap.error(f"--{k.replace('_', '-')} or --base is required")
            setattr(a, k, os.path.join(a.base, *rel))
    os.makedirs(a.out, exist_ok=True)
    mp = a.mutants
    mutants = json.load(open(mp))
    raw = load_jsonl(a.stored)
    runs_meta = load_jsonl(a.runs_meta)
    new = {i: to_chk(r) for i, r in load_jsonl(a.cov).items()}
    ids = [m["id"] for m in mutants]
    miss = [i for i in ids if i not in new or i not in raw or i not in runs_meta]
    if miss:
        raise SystemExit(f"{len(miss)} mutants missing from the inputs, e.g. {miss[:5]}")
    bad_runs = [i for i in ids if runs_meta[i]["rc"] != 0]
    if bad_runs:
        print(f"warning: {len(bad_runs)} runs with rc != 0: {bad_runs[:10]}", file=sys.stderr)

    # self-check: the stored snap/fix results must reproduce the stored tables
    ref = run_analyze(raw, mp, lambda i: raw[i]["cov"]["snap"], lambda i: raw[i]["cov"]["fix"], None, False,
                      a.out, "stored")
    rs = {s: summary(ref, s) for s in ("snap", "fix")}
    want = {"snap": (88.4, 84.3, (1270, 413, 1199)), "fix": (97.9, 95.4, (2353, 478, 116))}
    for s, (hz, al, pr) in want.items():
        got = (rs[s]["hazards"]["agreement_pct"], rs[s]["all"]["agreement_pct"],
               tuple(rs[s]["all"]["pairs"].get(k, 0) for k in ("both", "cov-only", "vvl-only")))
        if got != (hz, al, pr):
            raise SystemExit(f"self-check failed: stored {s} gives {got}, expected {(hz, al, pr)}")
    if any(t["class"] == "UNEXPLAINED" for r in ref["rows"] for t in r["fix_triage"]):
        raise SystemExit("self-check failed: stored fix side has unexplained keys")

    tab = {"stored_snap": {"summary": rs["snap"], "configs": config_tables(ref, "snap"), "triage": triage_table(ref, "snap")},
           "stored_fix": {"summary": rs["fix"], "configs": config_tables(ref, "fix"), "triage": triage_table(ref, "fix")}}
    for variant, afu in (("", False), ("+afu", True)):
        res = run_analyze(raw, mp, lambda i: raw[i]["cov"]["snap"], lambda i: new[i], runs_meta, afu,
                          a.out, a.name + variant)
        json.dump(res, open(os.path.join(a.out, f"analysis_{a.name}{variant}.json"), "w"), indent=1, default=list)
        tab[a.name + variant] = {"summary": summary(res, "fix"), "configs": config_tables(res, "fix"),
                                 "triage": triage_table(res, "fix"), "unexplained": res["unexplained"]}
        # per-mutant triage changes vs the stored fix side (same keys -> same triage expected)
        tri_ch = []
        for r0, r1 in zip(ref["rows"], res["rows"]):
            assert r0["id"] == r1["id"]
            t0 = sorted((t["side"], tuple(t["key"]), t["class"], t["why"]) for t in r0["fix_triage"])
            t1 = sorted((t["side"], tuple(t["key"]), t["class"], t["why"]) for t in r1["fix_triage"])
            if t0 != t1:
                tri_ch.append({"id": r1["id"], "mut": r1["mut"], "stored_fix": t0, a.name + variant: t1})
        tab[a.name + variant]["triage_changed_vs_stored_fix"] = tri_ch
    json.dump(tab, open(os.path.join(a.out, f"tables_{a.name}.json"), "w"), indent=1, default=list)

    ch = {"vs_stored_fix": per_mutant(mutants, lambda i: raw[i]["cov"]["fix"], lambda i: new[i], "stored cov.fix", a.name),
          "vs_stored_snap": per_mutant(mutants, lambda i: raw[i]["cov"]["snap"], lambda i: new[i], "stored cov.snap", a.name)}
    for spec in a.compare:
        pname, ppath = spec.split("=", 1)
        if not os.path.exists(ppath):
            print(f"warning: --compare {spec}: no such file, skipped", file=sys.stderr)
            continue
        prev = {i: to_chk(r) for i, r in load_jsonl(ppath).items()}
        ch[f"vs_{pname}"] = per_mutant(mutants, lambda i, prev=prev: prev[i], lambda i: new[i], pname, a.name)
    # guarantee eligibility per program: tools/check_trace.py's stats "eligibility" (if recorded),
    # and timing.check's report.meta["covenant.eligibility"]
    newraw = load_jsonl(a.cov)
    el = {"per_config": {}, "tool_vs_timing_mismatch": [], "not_eligible": [], "missing": 0}
    for m in mutants:
        r = newraw[m["id"]]
        e_tool = (r.get("check_trace") or {}).get("eligibility")
        e_tim = r.get("eligibility_timing")
        e = e_tool if e_tool is not None else e_tim
        c = el["per_config"].setdefault(m["config"], collections.Counter())
        if e is None:
            el["missing"] += 1
            c["unknown"] += 1
            continue
        c["eligible" if e.get("guarantee_eligible") else "not eligible"] += 1
        if not e.get("guarantee_eligible"):
            el["not_eligible"].append({"id": m["id"], "mut": m["mut"], "eligibility": e})
        if e_tool is not None and e_tim is not None and e_tool != e_tim:
            el["tool_vs_timing_mismatch"].append({"id": m["id"], "tool": e_tool, "timing": e_tim})
        if r.get("check_trace", {}).get("error"):
            el.setdefault("tool_errors", []).append({"id": m["id"], "error": r["check_trace"]["error"][:300]})
    el["per_config"] = {k: dict(v) for k, v in el["per_config"].items()}
    el["source"] = ("tools/check_trace.py stats.eligibility" if any((newraw[i].get("check_trace") or {}).get("eligibility")
                                                                   for i in ids) else "timing.check meta")
    ch["eligibility"] = el
    tl = sorted(new[i]["t_load"] for i in ids); tc = sorted(new[i]["t_check"] for i in ids)
    ch["timing"] = {"t_load_median": tl[len(tl) // 2], "t_check_median": tc[len(tc) // 2],
                    "t_load_sum": sum(tl), "t_check_sum": sum(tc)}
    json.dump(ch, open(os.path.join(a.out, f"changes_{a.name}.json"), "w"), indent=1, default=list)

    text = md(tab, a.name)
    for k, v in ch.items():
        if k.startswith("vs_"):
            text += (f"\n**{v['from']} -> {v['to']}:** verdict changed (all classes) {v['n_verdict_changed_all']}, "
                     f"(hazards) {v['n_verdict_changed_hazard']}; normalised key sets changed {v['n_keys_changed']}; "
                     f"diagnostics changed in code/where {v['n_diag_changed_codewhere']}, text only "
                     f"{v['diag_text_only']}; warnings changed {v['warn_changed']}; "
                     f"delta {v['diag_delta_by_code']}\n")
    el = ch["eligibility"]
    text += (f"\n**guarantee eligibility** ({el['source']}): {el['per_config']}; not eligible "
             f"{len(el['not_eligible'])}; unknown {el['missing']}; tool vs timing-meta mismatches "
             f"{len(el['tool_vs_timing_mismatch'])}; check_trace errors {len(el.get('tool_errors', []))}\n")
    for variant in ("", "+afu"):
        text += f"\ntriage rows changed vs stored fix ({a.name}{variant}): {len(tab[a.name + variant]['triage_changed_vs_stored_fix'])}; unexplained: {len(tab[a.name + variant]['unexplained'])}\n"
    open(os.path.join(a.out, f"tables_{a.name}.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
