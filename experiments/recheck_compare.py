"""Compare a re-check with the archived results of the release (used by recheck-release.sh).

    python3 experiments/recheck_compare.py archived <extracted archive dir> <out dir>
    python3 experiments/recheck_compare.py fresh <stored release-checks dir> <out dir> <classes dir>
    python3 experiments/recheck_compare.py mutants <stored per-mutant jsonl> <new per-mutant jsonl>

archived: for every trace, the findings (code, where, message) of <out>/<trace>/check.json against the
  release-check.json beside the archived trace, and its eligibility.
fresh: for every program and capture mode, the findings of <out>/<study>/<prog>/<mode>/check.json against the
  stored check.json, and the report groups of groups.json against the classified groups of
  classes-release-study<n>.json (id, number of diagnostics, repair or site description); then the class
  totals (true / technicality / unresolved / false alarm) recomputed from the classification of every group.
mutants: per mutant, the checker verdict and the set of diagnostics.
Prints one summary line per corpus and a table of differences; exit status 1 if anything differs.
"""
import collections
import json
import sys
from pathlib import Path


def keys(check):
    return collections.Counter((f.get("code"), f.get("where", ""), f.get("message", ""))
                               for f in json.loads(Path(check).read_text()).get("findings", []))


def eligibility(check):
    e = json.loads(Path(check).read_text()).get("stats", {}).get("eligibility", {})
    return e.get("guarantee_eligible") if isinstance(e, dict) else e


def archived(arch, out):
    diff, n, elig = [], 0, 0
    for old in sorted(Path(arch).rglob("release-check.json")):
        rel = old.parent.relative_to(arch)
        new = Path(out) / rel / "check.json"
        n += 1
        if not new.exists():
            diff.append(f"{rel}: no new check.json")
            continue
        a, b = keys(old), keys(new)
        elig += bool(eligibility(new))
        if a != b or eligibility(old) != eligibility(new):
            diff.append(f"{rel}: lost {sum((a - b).values())}, added {sum((b - a).values())}, "
                        f"eligible {eligibility(old)} -> {eligibility(new)}")
    print(f"archived traces: {n} re-checked, {len(diff)} differ, {elig} eligible")
    return diff


def group_sig(g):
    """(id, diagnostics, description) of every report group of a groups.json, in the order the
    classification files use: R<n> for repairs, G<n> for other groups."""
    out = {}
    for i, r in enumerate(g.get("repairs", []), 1):
        out[f"R{i}"] = (r["diagnostics"], r["repair"])
    for i, o in enumerate(g.get("other_groups", []), 1):
        out[f"G{i}"] = (o["diagnostics"], f"{o['code']} on {o['resource']}: {o['earlier']} -> {o['later']}")
    return out


def fresh(stored, out, classes):
    diff, totals, n = [], collections.Counter(), 0
    for study in (1, 2):
        cls = json.loads((Path(classes) / f"classes-release-study{study}.json").read_text())
        for d in sorted((Path(out) / f"study{study}").glob("*/*")):
            prog, mode = d.parent.name, d.name
            n += 1
            old = Path(stored) / f"study{study}" / prog / mode / "check.json"
            if keys(old) != keys(d / "check.json"):
                diff.append(f"study{study}/{prog}/{mode}: findings differ")
            if eligibility(old) != eligibility(d / "check.json"):
                diff.append(f"study{study}/{prog}/{mode}: eligibility {eligibility(old)} -> "
                            f"{eligibility(d / 'check.json')}")
            g = json.loads((d / "groups.json").read_text())
            sig = group_sig(g)
            ck = f"{prog}/{mode}"
            if ck not in cls:      # a mode that was not classified separately (its result equals the other mode's)
                ck = f"{prog}/{'observed' if mode == 'conservative' else 'conservative'}"
            want = {c["id"]: (c["diagnostics"], c["what"]) for c in cls.get(ck, [])}
            if sig != want:
                ids = sorted(set(sig) ^ set(want)) + sorted(k for k in set(sig) & set(want) if sig[k] != want[k])
                diff.append(f"study{study}/{prog}/{mode}: groups differ ({', '.join(ids[:6])})")
            for c in cls.get(ck, []):
                part = "incomplete" if g.get("status") == "incomplete" else "complete"
                totals[(study, mode, part, c["class"])] += 1
    print(f"fresh studies: {n} program/mode runs re-checked and re-grouped, {len(diff)} differ")
    for study in (1, 2):
        for mode in ("conservative", "observed"):
            for part in ("complete", "incomplete"):
                t = [totals[(study, mode, part, k)] for k in ("TRUE", "TECH", "UNRESOLVED", "FA")]
                if sum(t):
                    print(f"  study {study}, {mode}, {part} grouping: {' / '.join(map(str, t))} "
                          f"(true / tech / unresolved / FA), {100 * t[0] / sum(t):.1f}% true")
    return diff


def mutants(old, new):
    a = {r["id"]: r for r in map(json.loads, open(old))}
    b = {r["id"]: r for r in map(json.loads, open(new))}
    diff = [i for i in sorted(set(a) | set(b))
            if i not in a or i not in b or a[i]["checker"] != b[i]["checker"]
            or sorted(map(tuple, a[i]["diags"])) != sorted(map(tuple, b[i]["diags"]))]
    print(f"barrier mutants: {len(b)} re-checked, {len(diff)} differ in verdict or diagnostics")
    return diff


if __name__ == "__main__":
    kind = sys.argv[1]
    diff = {"archived": archived, "fresh": fresh, "mutants": mutants}[kind](*sys.argv[2:])
    for x in diff[:50]:
        print("  DIFF", x)
    sys.exit(1 if diff else 0)
