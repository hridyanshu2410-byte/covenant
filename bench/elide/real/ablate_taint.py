"""Diagnostic ablation (NOT a sound plan): prove with Elider.taint_baseline_hazards = False.

    PYTHONPATH=<repo root> python3 ablate_taint.py trace.jsonl spv/ out.plan.json [--header H]

Without taint pinning, 'no new hazard' is measured only against diagnostics the program already
has, so a change that makes an existing hazard worse is invisible (for example in
computenbody). The output plan carries "unsafe_ablation": true and must never be applied; it
only answers "which barriers did taint pinning hold back, and would the search have changed them?"
"""
import argparse
import json

from covenant import elide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("spv")
    ap.add_argument("out")
    ap.add_argument("--header")
    a = ap.parse_args()
    elide.Elider.taint_baseline_hazards = False
    el = elide.Elider(a.trace, a.spv, a.header)
    el.run()
    pj = el.plan_json()
    pj["unsafe_ablation"] = "taint_baseline_hazards=False: NOT a verified plan, never apply it"
    pj["chain"] = []
    json.dump(pj, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
