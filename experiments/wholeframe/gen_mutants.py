"""Generate the whole-frame mutants.

    python3 gen_mutants.py --wf BUILD/wf --out mutants.json [--seed 20261008] [--random 150]

(wf --sites only prints the mutation catalogue; it needs the model driver on LD_LIBRARY_PATH
to load, but creates no Vulkan objects.)

Configurations (workload, args, oracles):
  W1-F2  offscreen --fif 2   lavapipe+VVL, model driver, mock+VVL
  W1-F3  offscreen --fif 3   same
  W2     timeline  --fif 2   same
  W3     upload    --fif 2   same
  W2-MQ  timeline  --fif 2 --mq   model driver (2 queue families) and mock ICD+VVL only:
                                  lavapipe exposes one queue family with one queue
Per configuration: the unmutated baseline; every (site, kind) applied in every frame
("single-all"); every (site, kind) applied only in frame 3 ("single-one", a steady-state
frame with frames in flight); and N random mutants of 1-3 mutations on distinct sites,
each in every frame (p = 1/2) or in one random frame 1..5.
"""
import argparse
import json
import random
import subprocess

FRAMES = 6
CONFIGS = [
    ("W1-F2", ["offscreen", "--fif", "2", "--frames", str(FRAMES)], True),
    ("W1-F3", ["offscreen", "--fif", "3", "--frames", str(FRAMES)], True),
    ("W2", ["timeline", "--fif", "2", "--frames", str(FRAMES)], True),
    ("W3", ["upload", "--fif", "2", "--frames", str(FRAMES)], True),
    ("W2-MQ", ["timeline", "--fif", "2", "--frames", str(FRAMES), "--mq"], False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--random", type=int, default=150)
    a = ap.parse_args()
    out = []
    for ci, (cfg, args, lavapipe) in enumerate(CONFIGS):
        cat = json.loads(subprocess.check_output([a.wf] + args + ["--sites"]))
        rng = random.Random(a.seed * 10 + ci)
        def add(kind, muts):
            out.append({"id": f"{cfg}-{len([m for m in out if m['config'] == cfg]):04d}", "config": cfg,
                        "args": args, "lavapipe": lavapipe, "gen": kind, "muts": muts,
                        "mut": ";".join(f"{s}:{k}@{f}" for s, k, f in muts)})
        add("baseline", [])
        for e in cat:
            for k in e["kinds"]:
                add("single-all", [(e["site"], k, "*")])
        for e in cat:
            for k in e["kinds"]:
                add("single-one", [(e["site"], k, 3)])
        for _ in range(a.random):
            n = rng.choices([1, 2, 3], [0.4, 0.35, 0.25])[0]
            sites = rng.sample(cat, n)
            muts = []
            for e in sites:
                k = rng.choice(e["kinds"])
                f = "*" if rng.random() < 0.5 else rng.randint(1, FRAMES - 1)
                muts.append((e["site"], k, f))
            add("random", muts)
    json.dump(out, open(a.out, "w"), indent=0)
    from collections import Counter
    print(Counter((m["config"], m["gen"]) for m in out))
    print(len(out), "mutants")


if __name__ == "__main__":
    main()
