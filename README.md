# COVENANT

COVENANT checks the synchronization of a whole Vulkan frame from **one run that
needs no GPU**. The program runs unmodified on a model driver, `libcovvk`, a
stand-in `libvulkan` that records every Vulkan call as JSON lines and traps
host writes to mapped memory. The checker turns that trace into a frame (queues,
semaphores, fences, barriers, render passes, host accesses and the presentation
engine) and reports every pair of conflicting accesses that the recorded
synchronization leaves unordered under **any** schedule it allows, not only the
schedule a real run happens to take.

This repository accompanies an anonymous paper submission (Anonymous authors).

## Contents

| Path | What |
|---|---|
| `covenant/` | the checker (Python): trace loader `host_trace.py`, timing checker `timing.py`, executable reference semantics `timing_ref.py`, report grouping and repairs `report_groups.py`, proof-licensed barrier elision `elide.py`, SPIR-V reader `spirv.py`, and the meaning checkers (`meaning*.py`, `host_*.py`, `labels.py`, `signatures/`) |
| `covenant/data/` | SPIR-V grammar files from Khronos SPIRV-Headers, with their license |
| `model_driver/` | the model driver `libcovvk` (C++), its code generator `gen.py`, the image-I/O interposer `covio.c`, and examples; see `model_driver/README.md` for the trace format |
| `tools/check_trace.py` | checks one trace of an unannotated program |
| `bench/run.py`, `bench/cases/` | GPUBugBench: 114 distilled reproductions of real-world graphics bugs, each with a buggy and a fixed variant, the runner, and this release's results (`bench/results-release.md`) |
| `bench/elide/` | barrier-elision workloads, plans, traces and measurement scripts (see `bench/elide/README.md`) |
| `tests/` | unit, property and end-to-end tests; `tests/fixtures/minimal_frames/` holds the minimal frames of the paper's appendix as standalone scripts |
| `examples/`, `run_demo.py` | a small forward renderer written against the contract IR, with seeded bugs |
| `paper/` | LaTeX sources and figures of the journal paper and its supplementary material |
| `experiments/` | the inputs and outputs of every evaluation result, with a README per experiment; `experiments/README.md` maps each table of the paper to its files and re-check command (whole-frame differential test, barrier mutants, fresh studies with the upstream ledger, source repairs, archived traces) |

## Requirements

- Linux x86-64, Python 3.10 or newer, `pytest`, `numpy`;
- a C++17 compiler (`clang++-20` is used when present, otherwise `g++`) and GNU make;
- [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) at tag **v1.4.365**
  (older system headers, such as 1.3.275, do not build the model driver);
- `glslangValidator`;
- optionally `spirv-val` (SPIRV-Tools), which the tests use to validate their SPIR-V fixtures;
- for the meaning-layer tests: the libclang Python bindings (`clang.cindex`) and `slangc`
  (set `SLANGC` to its path; default `/opt/slang/bin/slangc`). Checking a trace does not need
  libclang; only `--host-src` (reading the C++ host for struct layouts) and the meaning layer do.

```sh
git clone --depth 1 --branch v1.4.365 https://github.com/KhronosGroup/Vulkan-Headers
export VKINC=$PWD/Vulkan-Headers/include
```

## Build the model driver

```sh
make -C model_driver VKINC=$VKINC VKH=$VKINC/vulkan/vulkan_core.h all
```

This writes `model_driver/build/libcovvk.so` (plus the copies `libvulkan.so` and
`libvulkan.so.1`) and `libcovio.so`. Run `make` without `-j`: the generated
`.inc` files come from one rule with two targets. Everything under
`model_driver/build/` is a build product and is not tracked.

Run a program under it:

```sh
COVENANT_TRACE=app.jsonl COVENANT_SPIRV_DIR=spv LD_LIBRARY_PATH=model_driver/build ./app
```

## Run the tests

```sh
COVENANT_VKINC=$VKINC python3 -m pytest -q tests
```

`COVENANT_VKINC` is the Vulkan-Headers include directory; the tests that build the
model driver and the litmus programs use it, and the barrier-elision tests read flag
names from its `vulkan_core.h`. With all requirements present, all 391 tests run and
pass, none skipped. A test whose tool is missing (libclang, `slangc`, `spirv-val`,
`glslangValidator`, a C++ compiler, the headers) is skipped rather than failed;
`pytest -rs` lists each skip and its reason. Only the checker's own tests run with
Python, `pytest` and `numpy` alone.

The versions the release was tested with: Python 3.13, pytest 9.1, numpy 2.5,
Vulkan-Headers v1.4.365, glslang 15.1.0, SPIRV-Tools v2025.1 (`spirv-val`), clang 20
with its Python bindings, Slang 2026.18.

## Check a trace

```sh
python3 tools/check_trace.py <trace.jsonl> <spv dir> --json out.json \
    [--host-src file.cpp ...] [--host-arg -I... ...]
```

`<spv dir>` is the directory given as `COVENANT_SPIRV_DIR` when the trace was
recorded. The tool prints the findings and a statistics line; `--json` writes
`{"stats": {...}, "findings": [...]}`, each finding with `checker`, `code`,
`message`, `where`, `notes` and `severity`.

### Host-write capture

The model driver protects mapped memory and records which pages the host wrote
between two Vulkan calls. `COVENANT_HOST_CAPTURE` selects how the checker uses this:

- `conservative` (default): every page the trap saw written in an interval is
  charged, over every mapped binding it overlaps, whether or not its bytes changed.
  This is the mode the guarantee is stated for.
- `observed`: only the byte runs that changed are charged. A store of the value
  already there, or a write restored within one interval, is then not seen.

Other switches (all off by default): `COVENANT_HEURISTIC_DISCARD=1` lets an
execution dependency suffice before a layout transition from `UNDEFINED`
(not the specification); `COVENANT_LEGACY_SUBPASS_FLAGS=global` reads subpass
dependencies of traces without recorded dependency flags as framebuffer-global;
`COVENANT_HEURISTIC_SLOTS=1` takes two shader accesses through indices taken from
atomics (e.g. `data[atomicAdd(n, 1)]`) as disjoint, which the checker does not
establish. Each one leaves the verified configuration. Explicit arguments of
`timing.check` (`discard_relaxed=`, `slots=`) override the environment, and the
reported configuration is always the one the check ran with.

### The `eligibility` field

`stats.eligibility` says whether a result carries the guarantee (every race of the
recorded execution is reported):

| Field | Meaning |
|---|---|
| `verified_configuration` | the default configuration: spec-strict transitions from `UNDEFINED`, conservative host capture, framebuffer-local reading of unrecorded subpass flags, no exemption for accesses through atomically allocated slots |
| `checking_complete` | every queue ran to its end (no `timing.deadlock`) |
| `model_gaps` | commands or state the model does not interpret and that may hide memory accesses, including a pipeline whose shader code is missing from the SPIR-V directory or cannot be parsed ("shader code not available") and a shader that reaches descriptor memory through a pointer the analysis cannot trace to a descriptor, or applies to it an instruction outside the analysis's inventory of memory effects, such as an untyped pointer ("shader accesses not attributed"); in both cases the pipeline's storage accesses are taken as writes; must be empty |
| `host_domain_premise_pairs` | number of device-write / host-access pairs accepted only under the host-domain premise (a fence or semaphore wait without a barrier to the `HOST` domain); reported as `timing.premise-host-domain` warnings; must be 0 |
| `guarantee_eligible` | all of the above hold |

### Release source hash

`stats.covenant_version` identifies the checker source a result came from:
the first 12 hex digits of a SHA-1 over the file name and bytes of each
`covenant/*.py` (sorted by name), followed by `bench/run.py`
(`covenant_version()` in `tools/check_trace.py`). This release is
**`fa974bb2d72d`**:

```sh
python3 -c "import runpy; print(runpy.run_path('tools/check_trace.py')['covenant_version']())"
```

## Benchmark

```sh
make -C model_driver VKINC=$VKINC VKH=$VKINC/vulkan/vulkan_core.h all
python3 bench/run.py [case-dir-substring ...] [--json out.json] [--md out.md]
```

Each case builds a buggy and a fixed variant, runs both on the model driver and
checks them with every COVENANT checker; it passes when the buggy variant reports
every expected code and the fixed variant reports nothing. `COVBENCH_WORK` sets the
working directory (default: the system temporary directory).

## License

MIT, see `LICENSE`. `covenant/data/` contains files from Khronos SPIRV-Headers under
their own license (`covenant/data/SPIRV-Headers-LICENSE`).
