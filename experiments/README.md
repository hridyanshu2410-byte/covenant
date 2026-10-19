# Experiments: where each result of the paper comes from

Every row of the paper's release table (Table 5) and every other table of the evaluation, with the inputs and
the outputs in this repository and the command that re-checks it. All checks need only Python and the files here;
the commands that regenerate traces or validation verdicts also need the model driver, lavapipe and the Khronos
validation layer (named in each README).

| Paper result | Inputs and outputs | Re-check |
|---|---|---|
| Test suite (387 tests) | `tests/` | `python3 -m pytest -q tests` (see the top-level README) |
| Barrier mutants, 95.9% agreement with validation | `mutants/` | `bench/elide/difftest/recheck.py`, see `mutants/README.md` |
| Ray tracing and mesh shading, 15 runs | `archived-traces/` (`rtmesh/*`), `tests/test_rt_mesh.py` | `archived-traces/README.md` |
| Whole-frame differential test, 1,251 programs (Table 6) | `wholeframe/` | `wholeframe/recheck.sh`, see `wholeframe/README.md` |
| GPUBugBench, 114 cases (Table 7) | `../bench/cases/`; the release's table is `../bench/results-release.md` | `python3 bench/run.py` (needs the model driver) |
| Fresh studies 1 and 2, 57 programs (Tables 8 and 9) | `fresh-studies/` (traces, the release's checks, triaged groups of the release and of the snapshots frozen before each study) | `fresh-studies/README.md` |
| Upstream status of the 16 true causes | `fresh-studies/UPSTREAM.md` | (a ledger of upstream commits and issues) |
| Repaired programs, 15 | `repairs/` (patches, build scripts, traces) | `repairs/README.md` |
| Archived sample, engine and adjudication traces, 13 | `archived-traces/` | `archived-traces/README.md` |
| Verified barrier transformations | `../bench/elide/` (workloads, plans, traces, lavapipe and GPU results) | `../bench/elide/README.md` |

The earlier studies (blind rounds, tutorials, the three engine studies) were checked with the snapshot frozen
before each of them; their per-program results are tables of the supplement (§E, §F, §H). Of their traces, those
re-checked with the release are in `archived-traces/` and `repairs/`.
