# Archived traces: samples, engines, adjudication, ray tracing and mesh shading

The recorded traces behind the last row and the ray-tracing row of the paper's release table (Table 5),
each with its SPIR-V directory and the release's result on it (`release-check.json`).

| Trace | What it is |
|---|---|
| `sdl1-teaser` | the SDL3 host-write race of the paper's teaser figure (bug SDL-1) |
| `case30-figure` | benchmark case 30 as recorded for a figure (render-finished semaphores indexed by frame in flight); its shader module was not kept, so the release reports a model gap ("shader code not available") and the result is outside the guarantee's domain, with the same findings |
| `sw13/raytracing` | the ray-tracing program of one sample collection |
| `vsg_rtt` | VulkanSceneGraph's render-to-texture example |
| `difftest/{deferred,particles,postchain}` | the three barrier-elision workloads, unmutated (the mutants of `../mutants/` are derived from them) |
| `adjudication/vsg5-*`, `adjudication/depthpeel-*` | the VSG-5 and `vsg_depthpeel` reproductions as written and in variants that add or remove one dependency (Supplement §D) |
| `rtmesh/*` | the 15 runs of the ray-tracing and mesh-shading test (13 programs; `lt-pair` and `lt-copy` test rule R-LT) |

`data/archived-traces.tar.xz` holds `archived/<trace>/trace.jsonl`, `spv/` and `release-check.json`. Some
traces have no pipelines, so their `spv/` is empty.

## Re-check

```sh
mkdir -p at && tar -xJf experiments/archived-traces/data/archived-traces.tar.xz -C at
for d in $(find at/archived -name trace.jsonl -printf '%h\n'); do
  python3 tools/check_trace.py $d/trace.jsonl $d/spv --json $d/check.json > /dev/null
done
```

Each `check.json` has the same findings as the `release-check.json` beside it. Every trace is
guarantee-eligible except `case30-figure`.
