# Barrier mutants: checker against synchronization validation

The 750 barrier mutants of the paper's E1 (Supplement §H.2). Each mutant takes one of the three
barrier-elision workloads of `bench/elide/` (`deferred`, `particles`, `postchain`; 250 each) and removes
or narrows a random subset of its barriers. Both tools judge every mutant: the checker on the recorded
frame with the mutation applied, and the Khronos validation layer with synchronization validation on
lavapipe, with `VK_LAYER_COVENANT_elide` applying the same mutation to the running program.

| File | Contents |
|---|---|
| `data/mutants.tar.xz` | `traces/`: the recorded trace and SPIR-V of each workload; `mutants/<workload>/`: `manifest.json` and one `.plan.json` per mutant; `results.jsonl`: one row per mutant with the validation verdict (`syncval.sync` lists the SYNC-* messages); rows with `"control": true` are the unmutated programs |
| `data/per-mutant-release.jsonl` | this repository's checker on every mutant: `checker` (any new hazard), `syncval` (any SYNC-* message) and the checker's diagnostics |

## Re-check

```sh
mkdir -p mut && tar -xJf experiments/mutants/data/mutants.tar.xz -C mut
python3 bench/elide/difftest/recheck.py --traces mut/traces --mutants mut/mutants \
    --results mut/results.jsonl --out recheck.json
```

`recheck.json` counts the mutants per workload in four cells: `both` (both tools report a hazard), `neither`,
`checker_only` and `syncval_only`. With this release the totals are 633 / 86 / 31 / 0 (95.9% agreement); the
triage of the 31 checker-only mutants is in Supplement §H.2. The validation verdicts are reused, so no
GPU or Vulkan driver is needed. `bench/elide/difftest/run.py` regenerates them (it needs lavapipe, the
validation layer and the elision layer of `bench/elide/`).
