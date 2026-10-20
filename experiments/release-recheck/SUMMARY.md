# Re-check of the archived corpora with checker f28a96b8b6fb

Produced by `experiments/recheck-release.sh` on the working tree of this repository. The checker source, and so its
fingerprint, is the same in every commit from 3612c02 on. The run was done in two parts on one machine: tests, mutants,
archived, wholeframe, fresh and repairs, then elide and bench. Validation verdicts are the stored ones; no program and
no validation layer was re-executed. Per-corpus details are in the files beside this one.

- python: Python 3.13.16; libclang: yes
- spirv-val: SPIRV-Tools v2025.1; glslangValidator: Glslang 11:15.1.0
- host: Linux 6.18, 4 cores, 15 GB, no GPU

| corpus | result | time |
|---|---|---|
| tests | 392 passed, 0 skipped, 11,632 subtests passed (`tests.txt`) | 144 s |
| mutants | 750 re-checked, 0 differ in verdict or diagnostics; both / neither / checker only / validation only = 633 / 86 / 31 / 0 (`mutants.txt`) | 7 s |
| elide | 111 barrier calls, all with the archived remove / narrow / keep decision (63 / 45 / 3); 15 narrowed calls name the sync2 sub-stages `COPY` or `CLEAR` where the archived plans name `TRANSFER` (`elide.txt`) | 2 s |
| archived | 28 traces re-checked, 0 differ, 27 guarantee-eligible (`case30-figure` has no shader module) (`archived.txt`) | 18 s |
| wholeframe | 1,251 traces, 0 errors, all eligible; hazard agreement 97.9%, hazard pairs both / checker only / validation only = 1,440 / 32 / 14 (`wholeframe-tables.md`) | 391 s |
| fresh | 114 program/mode runs re-checked and re-grouped, 0 differ in findings, eligibility or groups (`fresh.txt`) | 5,943 s |
| repairs | 304 run/mode rows, 0 with changed findings, 0 failed checks (`repairs-COMPARE.md`) | 158 s |
| bench | 114 cases, table identical to `bench/results-release.md` (`bench.md`) | 2,581 s |

Fresh-study class totals recomputed from the classification of every re-grouped group (true / technicality /
unresolved / false alarm):

- study 1, conservative: 130 / 10 / 2 / 1,026 (11.1% true); observed: 130 / 10 / 2 / 0 (91.5% true)
- study 2, conservative, 24 programs with complete grouping: 260 / 1 / 8 / 6 (94.5% of 275 true); observed: the same
- study 2, `gtk_path_walk` (grouping incomplete, 35 site groups): 22 / 0 / 12 / 1 in both modes

The `gtk_path_walk` observed mode was not classified separately; its groups equal the conservative ones, and the
comparison uses that classification.

Overall: every corpus is identical to the archived results.
