# Proof-licensed barrier elision

COVENANT records a whole run of a Vulkan program on the model driver (no GPU).
It then **proves** which `vkCmdPipelineBarrier(2)` calls can be removed, or which
of their stage/access masks can be narrowed, without creating a hazard under any
schedule the program's synchronization allows. The layer `VK_LAYER_COVENANT_elide`
applies that plan on a real driver, so the speedup can be measured on real hardware.

| Piece | Where |
|---|---|
| The prover | `covenant/elide.py` (`python3 -m covenant.elide trace.jsonl -o plan.json`); its docstring states what is proved and the ID scheme |
| The layer (C++) | `elide_layer/covenant_elide_layer.cpp`, manifest template, `build.sh` |
| Workloads | `particles.cpp`, `deferred.cpp`, `postchain.cpp` (+ `common.hpp`, `shaders/`, generated `shaders_spv.h`) |
| Proved plans | `plans/<workload>.plan.json`; `plans/<workload>.naive.json` is an UNSAFE control plan |
| Traces the plans were proved from | `traces/` |
| Lavapipe correctness run | `run_lavapipe.py` → `results/lavapipe.{md,json}` |
| GPU measurement | `measure.py` → `results/gpu-<host>-<time>.{md,json}` |
| Build for a GPU machine | `CMakeLists.txt` (Linux and Windows) or `build.sh` (Linux) |

## The workloads

All three are headless and offscreen (no window, no swapchain), and synchronized
the way cautious engines do it: an `ALL_COMMANDS` / `MEMORY_WRITE → MEMORY_READ|WRITE`
barrier after every pass or dispatch, and layout transitions with `ALL_COMMANDS` masks.
Each one reads its outputs back and prints an FNV-1a hash per output, the GPU time of
every frame (from timestamp queries), and the CPU submit-to-fence time.

* **particles**: 8 independent compute dispatches, one per particle system, each writing
  its own buffer, with a full barrier after each (one is spelled with sync1). After them, a
  binning pass (atomics into a histogram) and a shading pass that writes an image.
* **deferred**: 3 G-buffer render passes drawing to 3 different targets, a lighting pass
  that samples them, and a tonemap pass. There is a full barrier after every pass, plus the
  layout transitions.
* **postchain**: two independent post chains, bloom (A0→A5) and luminance (B0→B5),
  recorded interleaved, with a full barrier after every dispatch. Each barrier is redundant
  on its own, but not all of them together. Removing every individually redundant barrier
  (`plans/postchain.naive.json`) creates 30 races, and both COVENANT and syncval report them.

## Results on lavapipe

From `results/lavapipe.md` (Mesa 25.2.8 lavapipe, Khronos validation layers 1.3.275, checker as of
2026-10-05). lavapipe is a CPU driver. **Its timings say nothing about GPU speed**, so only
correctness is claimed here. Every workload reads back and hashes every output it produces
(particles: the image, the histogram and all 8 particle buffers; deferred: ldr, hdr and all 3
G-buffer targets; postchain: the output and all 12 chain levels).

| workload | barrier calls (3 frames + readback) | removed | narrowed | kept as is | structs removed / narrowed / kept | syncval hazards with the plan | output hashes |
|---|---|---|---|---|---|---|---|
| particles | 38 | 27 | 10 | 1 | 28 / 10 / 1 | 0 | identical |
| deferred | 26 | 15 | 10 | 1 | 16 / 32 / 1 | 0 | identical |
| postchain | 47 | 21 | 25 | 1 | 22 / 73 / 1 | 0 | identical |

Both matching modes were checked: strict (the traced 3 frames) and chain (8 frames, more than
were traced). Syncval also reports no hazard and no other validation message in either mode, and
the layer matched every planned call with no signature mismatch. The plans were also proved again
from traces recorded at `--scale 4` and `--scale 16` (the scales `measure.py` uses); they are
identical to the scale-1 plans, and a lavapipe run at scale 4 with the plan gives identical hashes
and no syncval hazard.

The one barrier kept as is in each workload is the readback's `TRANSFER → HOST` barrier.
The prover never touches host synchronization.

## Measuring on a GPU

Nothing here generates plans: the plans in `plans/` were proved on the model driver (no GPU)
and are applied as they are. Copy the whole repository (here checked out as `covenant/`) to the
GPU machine, because `CMakeLists.txt` uses `../../elide_layer`. A Python 3.8+ interpreter is needed; it uses no packages.

`measure.py` runs every workload at scales 1, 4 and 16 in three configurations: **base** (no
layer), **overhead** (layer loaded, plan disabled) and **plan** (layer applying the plan).
The configurations are interleaved in a shuffled order, 15 rounds × 300 frames, and the
first 30 frames of each run are dropped. For each case it reports:
- the median GPU frame time;
- the speedup, with a bootstrap 95% interval;
- whether the readback hashes of all three configurations are identical;
- whether the layer really applied the plan (signatures matched).

With `--validation` it also runs each configuration once under the Khronos validation layer
with synchronization validation. Expect 20–40 minutes in total.

### Linux (e.g. Ubuntu 24.04)

1. Use the GPU vendor's driver (for NVIDIA, the proprietary driver). Install the tools:

       sudo apt install build-essential cmake libvulkan-dev vulkan-tools vulkan-validationlayers python3

   Check with `vulkaninfo --summary` that the GPU is listed.
2. Build (from the folder that contains `covenant/`):

       cmake -S covenant/bench/elide -B build -DCMAKE_BUILD_TYPE=Release
       cmake --build build -j

   This produces `build/{particles,deferred,postchain}` and `build/layer/` (the layer `.so` and its `.json`).
3. Sanity check: the device line must name the GPU, and the layer must report its matches:

       ./build/particles --frames 5
       VK_ADD_LAYER_PATH=$PWD/build/layer VK_INSTANCE_LAYERS=VK_LAYER_COVENANT_elide \
         COVENANT_ELIDE_PLAN=$PWD/covenant/bench/elide/plans/particles.plan.json COVENANT_ELIDE_MODE=chain \
         ./build/particles --frames 5

   The second command must print the same `hashes` as the first, and a final
   `[covenant-elide] {...}` line with `"matched": 61` (12 per frame plus 1 in the readback)
   and `"sig_mismatch": 0`.
4. Prepare the machine. Plug in AC power and close other GPU applications. On a hybrid
   (Optimus) machine the workloads pick the discrete GPU by themselves; `--device "<name>"`
   (a substring of the device name) forces it. Clock locking is optional (`sudo nvidia-smi -lgc <MHz>,<MHz>`, undo with
   `sudo nvidia-smi -rgc`); many mobile GPUs refuse it, which is fine.
5. Run:

       python3 covenant/bench/elide/measure.py --bin build --layer build/layer --device "<name>" --validation

6. The results are written to `covenant/bench/elide/results/gpu-<host>-<time>.md` and `.json`.

### Windows 10/11

1. Install the following:
   - the NVIDIA Game Ready or Studio driver;
   - the **LunarG Vulkan SDK** (1.3.275 or newer; it brings the headers, the loader and the validation layer);
   - **Visual Studio 2022** (or its Build Tools) with the "Desktop development with C++" workload, which includes CMake;
   - **Python 3** from python.org.
2. In an "x64 Native Tools Command Prompt for VS 2022", go to the folder that contains `covenant\`:

       cmake -S covenant\bench\elide -B build
       cmake --build build --config Release

   This produces `build\Release\{particles,deferred,postchain}.exe` and
   `build\layer\Release\VkLayer_COVENANT_elide.dll` with `VkLayer_COVENANT_elide.json`.
3. Sanity check, the same as step 3 for Linux:

       build\Release\particles.exe --frames 5
       set VK_ADD_LAYER_PATH=%CD%\build\layer\Release
       set VK_INSTANCE_LAYERS=VK_LAYER_COVENANT_elide
       set COVENANT_ELIDE_PLAN=%CD%\covenant\bench\elide\plans\particles.plan.json
       set COVENANT_ELIDE_MODE=chain
       build\Release\particles.exe --frames 5
       set VK_INSTANCE_LAYERS=

4. Prepare the machine:
   - Windows power mode: "Best performance", on AC power.
   - NVIDIA Control Panel → Manage 3D settings → Power management mode: "Prefer maximum performance".
   - Write down whether "Hardware-accelerated GPU scheduling" is on.
5. Run it from a normal, **non-administrator** prompt. The Vulkan loader ignores layer-path
   environment variables in elevated processes.

       python covenant\bench\elide\measure.py --bin build --layer build\layer\Release --device "<name>" --validation

6. The results are written to `covenant\bench\elide\results\gpu-<host>-<time>.md` and `.json`.

Windows notes: `VK_ADD_LAYER_PATH` needs loader 1.3.234 or newer, which current drivers and
SDKs ship. With an older loader, use `VK_LAYER_PATH` instead, which replaces the search path;
`--validation` then needs the SDK's `Bin` folder added to it too. Windows Defender can slow
the first launches. The interleaved order of the configurations spreads that effect evenly.

### Reading the result

* `hashes identical = NO` or `plan applied = NO` invalidates that row. Report it rather than
  dropping it. A signature mismatch means the program recorded different barriers than in the
  trace, and the layer then forwards those barriers unchanged.
* `layer, plan off` against `base` shows what the layer costs by itself. It intercepts only
  `vkBeginCommandBuffer` and the barrier commands, so the difference should be within noise.
* Prior work (Blade, arXiv 2607.26506) recovered 29–32% of GPU time by dropping barriers between
  independent compute passes on NVIDIA, and about 7% on render passes. That is the range to compare
  `particles`/`postchain` and `deferred` against. Unlike Blade, every barrier removed here comes
  with a proof, and syncval agrees with it.

## Re-deriving the plans (Linux, lavapipe)

    python3 bench/elide/run_lavapipe.py [--work DIR]   # build, trace, prove, check on lavapipe, write results/

Or by hand, for any program that runs on the model driver:

    COVENANT_TRACE=t.jsonl COVENANT_SPIRV_DIR=spv LD_LIBRARY_PATH=model_driver/build ./app
    python3 -m covenant.elide t.jsonl --spirv-dir spv -o plan.json [--naive-control naive.json]

## Plan format (`covenant-elide-plan/1`)

* `barriers[]`: one entry per recorded barrier call, with these fields:
  - `rec`, `idx`: the strict key;
  - `chain`: the chain key;
  - `sig`: the content signature;
  - `decision`: `remove`, `narrow` or `required`;
  - `elements[]`: one per barrier struct, with `kind`/`index`, the `recorded` masks, an `action`
    (`keep`, `remove` or `narrow`) and, for `narrow`, the `new` masks as names and as hex `*_mask` values;
  - `sync1_*_stage_mask`: the per-call stages that sync1 needs;
  - `why`, per struct;
  - `not_removable_because`: the hazard the checker found when it tried to remove the barrier.
* `chain[]`: the same entries keyed by `chain`, merged over every recorded instance of that
  key, which is what makes the plan apply to any number of frames.
* `proof`: the oracle, the obligations, the baseline diagnostics, the oracle runs, and the
  source hash of the checker the plan was proved with.

Barrier identity: `rec` is the 0-based `vkBeginCommandBuffer` ordinal in the process, and `idx`
is the barrier index inside that recording. `chain` is FNV-1a 64 over the signatures of the
recording's barrier calls so far. `sig` is FNV-1a 64 of a canonical content string, documented
in `covenant/elide.py`. The layer changes a call only when the key matches **and** the
signature matches.

## Limits

* The proof covers the recorded run (the executed path). Chain matching extends it to frames
  that record the same barrier sequence, on the assumption that a frame with the same barrier
  sequence makes the same accesses. Hashing every command kind into the chain would remove
  that assumption, at the cost of intercepting every `vkCmd*`.
* The checker's resolution applies: whole buffers, images per mip level, every bound descriptor
  assumed used. Ordering across submissions through a host fence wait counts as a full
  dependency, as syncval also treats it.
* Lavapipe runs commands nearly in order, so identical hashes there are weak evidence; syncval
  is the strong check (see the naive control). The GPU run checks the hashes again.
* `timing.Signal` has no stage mask, so the oracle treats every semaphore signal as
  `ALL_COMMANDS`. The prover therefore pins every barrier in a submission batch that signals
  with a narrower stage mask (none of the three workloads uses semaphores).
* The layer matches by key and signature only. A program whose barrier sequence differs from
  the traced one (another code path, another driver feature level) gets its barriers forwarded
  unchanged; `sig_mismatch` in the layer stats shows when that happens.
