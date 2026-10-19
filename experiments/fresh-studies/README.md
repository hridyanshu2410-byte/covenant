# Fresh studies: traces, release checks and classifications

The two studies of previously unchecked programs (paper §7.5, Supplement §I):
study 1 has 32 programs (Qt Quick 3D, wgpu, Dear ImGui, vkcube) and study 2 has 25
(GTK 4, vkmark, mpv/libplacebo, GStreamer), each traced for 20 presents under the
model driver.

| File | Contents |
|---|---|
| `data/traces.tar.xz` | `f1/runs/<program>/` and `f2/runs/<program>/`: `trace.jsonl` and the `spv/` directory of each of the 57 programs |
| `data/release-checks.tar.xz` | `study1/` and `study2/`: `check.json` of this repository's checker on every program, in both capture modes (`conservative`, `observed`) |
| `data/classes-release-study1.json`, `data/classes-release-study2.json` | the triaged report groups of the released checker, keyed `<program>/<mode>` |
| `data/classes-frozen-study1.json`, `data/classes-frozen-study2.json` | the triaged report groups of the checker frozen before each study (the fresh results) |

Each classified group has its `id` (see below), its
`family` (`RACE` or `PROTOCOL`), its `class` (`TRUE`, `TECH`, `UNRESOLVED` or `FA`), the
`cause` it belongs to, its member diagnostics and the repair it proposes (`what`).
The triage rules are in Supplement §I.

## Re-check

```sh
mkdir -p fs && tar -xJf experiments/fresh-studies/data/traces.tar.xz -C fs
d=fs/f2/runs/vkmark_cube
COVENANT_HOST_CAPTURE=conservative python3 tools/check_trace.py $d/trace.jsonl $d/spv --json out.json
COVENANT_HOST_CAPTURE=conservative python3 -m covenant.report_groups $d/trace.jsonl --spirv-dir $d/spv --json groups.json
```

`out.json` has the same findings as the corresponding `check.json` of `release-checks.tar.xz`. In the
classification files, `R<n>` is the n-th entry of the `repairs` list of `groups.json` and `G<n>` the n-th
entry of its `other_groups` list. Every
program checks in seconds to a few minutes, except `gtk_path_walk`, which needs about
8 GB of memory to group (its grouping stops at the budget and is reported as incomplete).
