# Whole-frame differential test

This experiment compares COVENANT with Khronos synchronization validation on 1,251 small but
complete Vulkan programs, each of which renders whole frames.

`wf.cpp` holds one program with three workloads. Each workload is a frame loop with a swapchain,
frames in flight, per-frame and shared resources, host writes to mapped memory, fences, and binary
and timeline semaphores:

- **W1**, `wf offscreen`: compute writes a shared storage buffer, a draw renders into a per-frame
  offscreen image, and a blit copies that image to the swapchain image.
- **W2**, `wf timeline`: a compute submit and a graphics submit are chained by timeline semaphores,
  and the host paces on them. With `--mq`, compute runs on a second queue family, using exclusive
  buffers and queue-family ownership transfers.
- **W3**, `wf upload`: the host fills a staging ring, which is copied to a shared vertex buffer, and a
  depth-tested draw goes straight into the swapchain image.

Five configurations (W1-F2, W1-F3, W2, W3 and W2-MQ; F2/F3 means 2 or 3 frames in flight) run 6
frames at 64x64. Each has an unmutated baseline. Each also has mutants drawn from the program's own
mutation catalogue (`wf <workload> --sites`), where a mutation drops a wait, narrows a stage or access
mask, skips a present, and so on:

- every (site, kind) applied in every frame;
- every (site, kind) applied in frame 3 only;
- 150 random mutants of 1 to 3 mutations each.

That gives 5 baselines and 1,246 mutants (`gen_mutants.py`, fixed seed).

Every program runs unmodified on the model driver `libcovvk`. COVENANT checks each trace with
`host_trace.load_trace` + `timing.check`. The reference is Khronos validation with synchronization
validation: VVL `0ad67fd` on lavapipe for the single-queue configurations, and on the Khronos mock ICD
for W2-MQ, because lavapipe has one queue family. Validation is **not** re-run here. Its parsed messages
are stored in this folder (`data/raw_results.jsonl.gz`, with the program list `data/mutants.json`), and the
scorer compares against them.

## Files

| File | What |
|---|---|
| `wf.cpp`, `shaders/` | the workload program and its GLSL shaders |
| `build.sh` | builds `wf` and compiles the shaders into `$WORK/build/` |
| `gen_mutants.py` | writes the list of the 1,251 programs (`mutants.json`) from `wf --sites` |
| `runner.py` | runs every program under the model driver (Xvfb for the X11 surface) and keeps its trace |
| `check_cov.py` | checks every trace (`timing.check`; optionally `tools/check_trace.py` for guarantee eligibility) |
| `score.py` | scores the checks against the stored validation results; it uses `analyze.py` (normalisation, tables, triage of every disagreement) and `vvlparse.py` (validation-log parser) |
| `recheck.sh` | re-checks the archived traces and scores them |
| `regenerate.sh` | the whole pipeline from source: build, generate, run, check, score |

## Requirements

- Python 3.10 or newer.
- The model driver, built as described in the repository README. It needs Vulkan-Headers v1.4.365.
- To regenerate the traces, you also need a C++17 compiler, the libxcb headers, `glslangValidator`
  and `Xvfb`.
- For `recheck.sh` only: the traces archive `data/wholeframe-traces.tar.xz` (3.3 MB, in this folder), which
  holds the 1,251 recorded traces. `regenerate.sh` needs no archive.

## Run

Environment variables:

- `COVENANT_ROOT`: the checker root. The default is this repository's root.
- `VKINC`: the Vulkan-Headers include directory.
- `MD`: the model-driver build. The default is `model_driver/build`.
- `WORK`: a work directory, which you must set.
- `JOBS`: the number of parallel jobs. The default is 4.

Re-check the archived traces with this repository's checker:

```sh
ARCHIVE=experiments/wholeframe/data/wholeframe-traces.tar.xz WORK=$PWD/wf-work NO_TOOL=1 \
    bash experiments/wholeframe/recheck.sh
```

The timing-only check of all 1,251 traces takes seconds. Without `NO_TOOL=1`, every trace is also run
through `tools/check_trace.py`, which records guarantee eligibility. That takes about 1.5 s per trace.

Regenerate everything from source, using the stored validation results for scoring:

```sh
make -C model_driver VKINC=$VKINC VKH=$VKINC/vulkan/vulkan_core.h all
VKINC=$VKINC WORK=$PWD/wf-work STORED=$PWD/experiments/wholeframe/data/raw_results.jsonl.gz \
    bash experiments/wholeframe/regenerate.sh
```

## Output

Both scripts write `$WORK/out-<label>/`. The label defaults to the checker's `covenant_version`.

- `cov_<label>.jsonl`: the check of every program.
- `tables_<label>.md`, which is also printed: verdict agreement over all programs, the per-configuration
  tables and the triage of every disagreement.
- `analysis_<label>.json` and `analysis_<label>+afu.json`: the "+afu" view counts the
  `timing.acquire-first-use` warnings as hazards.
- `changes_<label>.json`: per-program changes against the stored checker results, and guarantee
  eligibility.

**Self-check.** Before scoring, `score.py` recomputes the stored tables from the stored results. It
aborts if they are not reproduced. These stored results were produced by the original prototype
checker, before and after its bug fixes (`snap`, `fix`).

## Result (as reported in the paper; checker `eb50030d8472`)

- **1,251 programs. Hazard agreement with validation (VVL `0ad67fd`): 97.9%.** This is the verdict "at
  least one hazard", with W2-MQ taken relative to its baseline. All finding classes agree in 95.4% of
  programs.
- (mutant, resource) hazard pairs:
  - 1,440 reported by both;
  - 32 by COVENANT only;
  - 14 by validation only.

  Every disagreement has a triage class. None is unexplained.
- **All 1,251 programs are guarantee-eligible:**
  - `verified_configuration: true`;
  - `checking_complete: true`;
  - no model gaps;
  - no host-domain premise pairs.
- Validation is the reference here, not ground truth. `analyze.py` sorts every disagreeing
  (program, finding) pair into one of four classes:
  - **VAL-LIMIT**: host-write races on mapped memory, which synchronization validation does not
    track;
  - **TECH**: the same race, reported on another object or under another message;
  - **DRIVER**: differences between the model driver and lavapipe, such as acquire order and
    `minImageCount`;
  - **MODEL**: documented modelling choices.
