# Repair cases: from diagnosis to a clean run

COVENANT diagnosed genuine synchronization bugs in four open-source Vulkan projects. This experiment
repairs those bugs in the projects' source and checks the repaired programs again. For every case, the
workflow is the same:

1. Start from the original program and the bug COVENANT diagnosed in it.
2. Apply a source patch (`patches/<CASE>.diff`).
3. Rebuild with the same flags.
4. Re-run with the same command under the model driver, ending after 20 presents.
5. Check the trace in both host-capture modes: conservative (the proved configuration) and observed.

Validation (lavapipe + VVL `0ad67fd`, synchronization validation) also runs on the repaired programs.

A repair is **clean for its case** when two conditions hold in the conservative check of the patched
program:

- no diagnostic is attributed to the case's cause;
- no diagnostic code appears that the original program did not have.

`scripts/make_case.py` attributes every diagnostic mechanically, by code and by resource or message.

## The 11 cases (nine causes)

| Case | Project, revision | Cause | Patch |
|---|---|---|---|
| VKMARK-PSR | vkmark `ab6e6f3` | one render-finished semaphore per scene, signalled again while a present may still wait on it | one per swapchain image, ported from vkmark pull request #89 (+42/-35, 14 files) |
| VKMARK-ACQ | vkmark `ab6e6f3` | the acquire semaphore is reused while the submission waiting on it may be pending | a semaphore and a fence per frame slot (+47/-18, 10 files) |
| VKMARK-UBO | vkmark `ab6e6f3` | the host rewrites the uniform buffer while the previous frame may still read it | one fence per scene, waited before the rewrite (+45/-5, 10 files) |
| VKMARK-CLR | vkmark `ab6e6f3` | the clear scene's first barrier does not chain to the acquire wait | widen `srcStageMask` (+1/-1) |
| VKMARK-DEPTH | vkmark `ab6e6f3` | the render-pass dependency covers color only; the shared depth image races between frames | add the depth stages and accesses (+10/-4) |
| VKMARK-ALL | vkmark `ab6e6f3` | all five vkmark causes | the five repairs together (+135/-58, 18 files) |
| GST-WAIT | GStreamer 1.24.2 (gst-plugins-bad) | `vulkansink` waits for the acquire semaphore at BOTTOM_OF_PIPE | wait at TRANSFER (+1/-1) |
| GST-IA | GStreamer 1.24.2 (gst-plugins-bad) | the full-screen quad makes sampled images visible to input-attachment reads only | `dstAccessMask = SHADER_READ` (+1/-1) |
| GST-ALL | GStreamer 1.24.2 (gst-plugins-bad) | both GStreamer causes | both lines (+2/-2) |
| IMGUI-PRE | Dear ImGui `9d9a621f` | the Vulkan helper transitions swapchain images before they are acquired | remove that submission (-75) |
| SDL-1 | SDL3 `e4af6f8e` | the Vulkan renderer rewrites vertex buffers that frames in flight still read | one set of vertex buffers per frame in flight (+38/-18) |

Each `.diff` begins with a header that gives:

- the upstream repository, the revision and the path the patch is relative to;
- the changed files;
- the cause and the repair.

`git apply` skips this header. The patches contain only the patch hunks; no other upstream source is
included.

Programs:

- **vkmark**: the scenes `clear`, `cube`, `desktop`, `effect2d`, `shading` (phong), `texture` and
  `vertex`.
- **GStreamer**: five `gst-launch-1.0` pipelines ending in `vulkansink`:
  - `sink`;
  - `colorconvert`;
  - `overlay`;
  - `viewconvert`;
  - `shaderspv`.
- **Dear ImGui**: `example_glfw_vulkan` and `example_sdl2_vulkan`.
- **SDL3**: `testsprite` with `SDL_RENDER_DRIVER=vulkan`.

`scripts/progs.sh` gives the exact commands.

## Requirements

- The model driver and the checker, as described in the repository README.
- Git and network access, to fetch the four projects.
- Build tools for the four projects:
  - `meson`, `ninja` and `cmake`;
  - `glslc`;
  - `libxcb-icccm4-dev` and `libxkbcommon-x11-dev`;
  - `libgstreamer1.0-dev` and `libgstreamer-plugins-base1.0-dev` 1.24.2;
  - the GLFW and SDL2 development packages, for the ImGui examples;
  - `clang-20`, for SDL. Set `CC` and `CXX` to use another compiler.
- `xvfb-run` and `bc`.
- For validation:
  - lavapipe, from Mesa;
  - a build of Vulkan-ValidationLayers (`0ad67fd` in the paper);
  - a C compiler, for the present-limit shim `scripts/presentlimit.c`.

## Run

Environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `COVENANT_ROOT` | the checker root | this repository's root |
| `MD` | the model-driver build | `$COVENANT_ROOT/model_driver/build` |
| `WORK` | the work directory; you must set it | none |
| `VVL` | the directory of the VVL build that holds `VkLayer_khronos_validation.json` | none |
| `LVP_ICD` | the lavapipe ICD manifest | `/usr/share/vulkan/icd.d/lvp_icd.json` |
| `GST_REPO` | the GStreamer repository | the freedesktop.org GitLab |

All cases (fetch, patch, build, trace, check, validate, summarize):

```sh
WORK=$PWD/rep-work VVL=<vvl build>/layers bash experiments/repairs/scripts/run_all.sh
WORK=$PWD/rep-work NOSV=1 bash experiments/repairs/scripts/run_all.sh VKMARK-ALL GST-ALL   # no validation
```

`run_all.sh` calls the single steps, which you can also run by hand:

- `setup_sources.sh`: fetch the sources and apply the patches;
- `build_*.sh`: build;
- `trace.sh <out> <prog> <tree> 20`: trace under the model driver;
- `check.sh <out> conservative|observed`: run `tools/check_trace.py` and `covenant.report_groups`;
- `sv.sh <out> <prog> <tree> 20`: run validation.

To re-check the archived traces (`data/traces.tar.xz`, 152 runs, 4.8 MB, in this folder) with
this repository's checker, and compare with the archived check results:

```sh
ARCHIVE=experiments/repairs/data/traces.tar.xz WORK=$PWD/rep-work JOBS=2 ONLY='^(vkmark/all|gst/both|imgui/fix|sdl/fix)/' \
    bash experiments/repairs/scripts/recheck.sh
```

The `ONLY` pattern selects the 15 repaired programs. Leave it out to re-check all 152 runs.

## Output

`run_all.sh` writes:

- `$WORK/runs/<project>/<variant>/<prog>/`: the trace, and `conservative/` and `observed/` with
  `check.json` and `groups.json`;
- `sv_<prog>/vvl.txt`: the validation log, for the repaired variants;
- per case:
  - `$WORK/cases/<CASE>/summary.md`: diagnostics and groups before and after the repair, in both
    modes, the attribution of every diagnostic to a cause, and the premise checks;
  - `data/runs.json` and `data/verdicts.json`.

`recheck.sh` writes `COMPARE.md` and `compare.json`.

## Result (as reported in the paper; checker `eb50030d8472`)

- **11 source-patch cases cover nine causes.** Every single-cause repair is clean for its own cause and
  adds no new diagnostic code.
- **With all of a project's patches applied, 15 programs report no diagnostic in either capture
  mode.** They are the 7 vkmark scenes (VKMARK-ALL), the 5 GStreamer pipelines (GST-ALL), the 2 Dear
  ImGui examples (IMGUI-PRE) and SDL3 `testsprite` (SDL-1). Validation reports no synchronization
  message on them.
- **Premises.** Every run meets the premises: no model warning, no `timing.unmodelled*` finding and no
  unimplemented command.
- Before the repairs, the conservative check reports these numbers of diagnostics / groups. Each count
  is a sum over the case's programs:

  | Project | Diagnostics / groups |
  |---|---|
  | vkmark | 1124 / 38 |
  | GStreamer | 495 / 33 |
  | Dear ImGui | 40 / 40 |
  | SDL3 | 166 / 2 |

**Note on `gst_colorconvert`.** In some runs, the structure checker reports one warning,
`structure.host-layout`. This is not a synchronization diagnostic. It comes from uninitialized padding
that GStreamer copies into a uniform buffer. It appears or not from run to run, in the original and in
the repaired build alike, and the repair is not meant to change it.
