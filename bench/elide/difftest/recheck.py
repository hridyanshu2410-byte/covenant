"""Re-run the checker side of the difftest with the current covenant/ checker
(syncval verdicts are reused from raw_results.jsonl; the layer applies the same
plans, so they do not change).

    python3 bench/elide/difftest/recheck.py --traces T --mutants M --results R --out OUT.json
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))

from covenant import elide  # noqa: E402
from analyze import act_from_json, sv_res, WORKLOADS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    for k in ("traces", "mutants", "results", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.results)]
    els, man = {}, {}
    for w in WORKLOADS:
        els[w] = elide.Elider(os.path.join(a.traces, f"{w}.jsonl"), os.path.join(a.traces, f"{w}-spv"))
        els[w].base = set(els[w].hazards({})[0])
        man[w] = {m["id"]: m for m in json.loads((Path(a.mutants) / w / "manifest.json").read_text())}
    cells, res_cells, disagree = Counter(), Counter(), []
    for r in rows:
        if r.get("control"):
            continue
        w = r["workload"]
        m = man[w][r["id"]]
        plan = {tuple(int(x) for x in k.split(",")): {l: act_from_json(v) for l, v in acts.items()}
                for k, acts in m["internal"].items()}
        bad, _, _ = els[w].new_hazards(plan)
        ck = bool(bad)
        sv = bool(r["syncval"].get("sync"))
        c = {(True, True): "both", (False, False): "neither", (True, False): "checker_only",
             (False, True): "syncval_only"}[(ck, sv)]
        cells[(w, c)] += 1
        cres = {d.where and __import__("re").search(r"'([^']+)'", d.message).group(1)
                for d in bad if __import__("re").search(r"'([^']+)'", d.message)}
        sres = {sv_res(h)[0] for h in r["syncval"].get("sync", [])}
        for x in cres | sres:
            res_cells[{(True, True): "both", (True, False): "checker_only",
                       (False, True): "syncval_only"}[(x in cres, x in sres)]] += 1
        if c == "syncval_only":
            disagree.append({"id": r["id"], "syncval": sorted(map(str, sres))})
    out = {"per_mutant": {f"{w}:{c}": n for (w, c), n in sorted(cells.items())},
           "per_resource": dict(res_cells), "syncval_only_mutants": disagree}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
