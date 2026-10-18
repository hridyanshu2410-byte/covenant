# COVENANT model driver (`libcovvk.so`)

A stand-in `libvulkan` that needs no GPU. An ordinary Vulkan C/C++ program
linked against it (or loading it through `vkGetInstanceProcAddr`, vulkan.hpp's
dynamic dispatcher or volk) runs to completion and leaves a JSON-lines trace
of every call. `covenant/host_trace.py` turns that trace into a `HostModel`
and a `covenant.timing.Frame`, so the timing checker can reason about
every schedule the recorded synchronization allows, not just the one a real run takes.

## Build

    make            # build/libcovvk.so, plus copies build/libvulkan.so{,.1}
    make example    # build/triangle and its SPIR-V (glslangValidator)

The build uses clang++-20 if installed, otherwise g++. `gen.py` reads the
`vulkan_core.h` given as `VKH` (default `/usr/include/vulkan/vulkan_core.h`;
the tests use Vulkan-Headers v1.4.365, see the end of this file) and generates the enum name
tables, a stub for every command not written by hand, and the
`vkGetInstanceProcAddr` table.

## Run a program under it

    # linked directly
    c++ app.cpp -Lmodel_driver/build -lcovvk -Wl,-rpath,$PWD/model_driver/build
    COVENANT_TRACE=app.jsonl ./a.out
    # an existing binary linked against libvulkan.so.1, or one that dlopens it
    COVENANT_TRACE=app.jsonl LD_LIBRARY_PATH=model_driver/build ./app

On a file system that does not keep the exec bit, run a binary built there
through the dynamic loader: `/lib64/ld-linux-x86-64.so.2 ./build/triangle ...`.

Environment:
- `COVENANT_TRACE`: the trace path (default `covenant-trace.jsonl`).
- `COVENANT_SPIRV_DIR`: if set, every shader module is also written there as `<sha1>.spv`.
- `COVENANT_SURFACE_EXTENT=WxH`: the headless surface's current extent (default 800x600).
- `COVENANT_NO_HOST_TRACKING=1`: turns off detection of mapped-memory writes.
- `COVENANT_NO_READBACK_FILL=1`: leaves readback buffers zeroed (see `covenant_readback_fill`).

### Image-I/O interposer (`build/libcovio.so`, `make io`)

`covio.c` records what a program hands to image writers, for the meaning
checker's host pixel path (`covenant/witness.py`): `fopen`/`fwrite`/`fclose`
(first 64 KiB of each write), libpng (`png_set_IHDR`, `png_set_gAMA`,
`png_set_sRGB`, `png_write_row`, `png_write_image`, `png_image_write_to_file`),
lodepng (`lodepng_encode32_file`, `lodepng_encode24_file`), and
`stbi_write_png` / `stbi_write_hdr` / tinyexr `SaveEXR` when they come from a
shared library (a header-only copy compiled into the program cannot be
interposed; its output is still seen through `fwrite`). Load it with
`LD_PRELOAD=model_driver/build/libcovio.so` and set `COVENANT_IO_TRACE` to the
JSON-lines output path; writes to `COVENANT_TRACE` and under
`COVENANT_SPIRV_DIR` are ignored. `bench/run.py` does this for every case.

```python
from covenant.host_trace import load_trace
from covenant import timing
m = load_trace("app.jsonl")          # HostModel: images, views, buffers, swapchains, pipelines, draws, frame
print(timing.check(m.frame))
```

## The fake device

The driver exposes one discrete GPU with Vulkan 1.3.275 and every feature
reported as supported (dynamic rendering, synchronization2, timeline
semaphores and so on). It has two queue families:
- family 0: graphics, compute and transfer, 2 queues;
- family 1: compute and transfer, 1 queue.

The loader names these queues `graphics0`, `graphics1` and `compute0`. There
are 4 memory types: device-local; host-visible and coherent; host-visible,
coherent and cached; and device-local plus host-visible.

Supported extensions:
- instance: `VK_KHR_surface`, `VK_EXT_headless_surface`, `VK_EXT_debug_utils`;
- device: `VK_KHR_swapchain`, plus the promoted 1.3 extensions, plus ray tracing and mesh shading:
  `VK_KHR_acceleration_structure`, `VK_KHR_ray_tracing_pipeline`, `VK_KHR_ray_query`,
  `VK_KHR_deferred_host_operations`, `VK_KHR_pipeline_library`, `VK_KHR_ray_tracing_maintenance1`,
  `VK_KHR_ray_tracing_position_fetch`, `VK_EXT_ray_tracing_invocation_reorder`, `VK_EXT_mesh_shader`,
  `VK_KHR_draw_indirect_count`, `VK_KHR_device_group` (see "Ray tracing, mesh shading and events").

Surface capabilities report `minImageCount = 2`. A swapchain gets
`imageCount = max(minImageCount, 3)`. `vkAcquireNextImageKHR` hands out images
round-robin and never blocks. Handles are unique, non-null and stable across
runs, for example `buffer#3`.

Host writes are detected as follows. `vkMapMemory` returns a private mapping
that is protected read-only. The first write to a page raises SIGSEGV, and the
handler marks the page dirty and makes it writable. On the next Vulkan call
the driver emits one `host_write` record per memory object and protects the
pages again. Faults outside these mappings are passed on to the previous handler.
(The pages are re-protected *before* the diff against the shadow copy, so a
write by another thread while the record is being built is either in this record
or faults into the next one; re-protecting after the diff would let such a write
surface one or more records later, attributed to a later interval.)

Each `host_write` record also carries
`fault_tids` (the threads whose store faulted a page of this memory in the
interval, at most 4, then `fault_tids_overflow: true`), `call_tids` (the threads
of the Vulkan calls that open and close the interval) and `threads` (threads in
the process at the flush, from `/proc/self/task`). At every flush the driver
checks that its SIGSEGV handler is still installed; if the application replaced
it, one `covenant_capture_warning` record is emitted (the capture is then
incomplete). The checker charges every written page by default
(`COVENANT_HOST_CAPTURE=conservative`); `writes` is used only in the optional
`observed` mode. `make hostcapture` builds `examples/hostcapture.cpp`, the probe
the capture tests run (same-value store, write-restore, page neighbour, second
thread).

Harness knob: `COVENANT_MAX_PRESENTS=N` closes the trace and ends the process
(`_exit(0)`) right after the N-th `vkQueuePresentKHR` is recorded, for programs that
have no frame-count option. Unset, nothing changes.

## Trace format (one JSON object per line)

Every record has `seq` (increasing) and `fn` (the Vulkan command, or
`host_write`, `covenant_trace_begin`, `covenant_trace_end`). Handles are
strings such as `"image#2"`, and a null handle is `null`. Enums appear as
`VK_*` names and flags as lists of names. Stage and access masks always use
the sync2 names, so sync1 bits are written the same way.

Main fields, by call:

| Call | Fields recorded |
| --- | --- |
| `vkCreateImage` | `image`, `format`, `extent`, `mip_levels`, `array_layers`, `usage`, `tiling`, `initial_layout` |
| `vkCreateImageView` | `view`, `image`, `format`, `view_type`, `aspect`, `base_mip`, `level_count`, `base_layer`, `layer_count` |
| `vkCreateBuffer` | `buffer`, `size`, `usage` |
| `vkAllocateMemory` | `memory`, `size`, `type_index`, `properties` |
| `vkBindBufferMemory` / `vkBindImageMemory` | `buffer` or `image`, `memory`, `offset`, `size` |
| `vkCreateSwapchainKHR` | `swapchain`, `format`, `color_space`, `extent`, `min_image_count` (requested), `image_count`, `surface_min_image_count`, `usage`, `present_mode`, `images` |
| `vkCreateShaderModule` | `module`, `sha1` (of the SPIR-V), `code_size` |
| `vkCreateDescriptorSetLayout` | `layout`, `bindings`: list of `{binding, type, count, stages}` |
| `vkCreatePipelineLayout` | `set_layouts`, `push_constants` |
| `vkAllocateDescriptorSets` | `sets`, `set_layouts` |
| `vkUpdateDescriptorSets` | `writes`: list of `{set, binding, array_element, type, elements}`, where each element is `{view, sampler, layout}`, `{buffer, offset, range}` or `{texel_view}`; also `copies` |
| `vkCreateGraphicsPipelines` | `pipelines`: list of `{pipeline, layout, render_pass, subpass, stages, color_formats, depth_format, stencil_format, vertex_bindings, vertex_attributes, topology, depth_test, depth_write, depth_compare, blend, viewports, dynamic_states}`; each stage is `{stage, module, sha1, entry}`; each `blend` entry is `{enable, write_mask, src_color, dst_color, color_op, src_alpha, dst_alpha, alpha_op}` (VkBlendFactor / VkBlendOp names); `viewports` (static viewports as `[x, y, width, height, minDepth, maxDepth]`) is present only when the viewport is not dynamic state |
| `vkCreateComputePipelines` | the same, without the graphics state |
| `vkCreateRenderPass` (also used for `vkCreateRenderPass2`) | `attachments`, `subpasses`, `dependencies` |
| `vkCreateFramebuffer` | `framebuffer`, `render_pass`, `attachments` |
| `vkCmd*` | always `cb` |
| `vkCmdBeginRendering` | `color`, `depth`, `stencil`: each `{view, layout, load, store, resolve_view, resolve_layout, clear?}` |
| `vkCmdBeginRenderPass` | `render_pass`, `framebuffer`, `clear_values` |
| `vkCmdPipelineBarrier` / `vkCmdPipelineBarrier2` | `memory_barriers`, `buffer_barriers`, `image_barriers`; each has `src_stage`, `src_access`, `dst_stage`, `dst_access`, `src_family`, `dst_family`, and for images `image`, `old_layout`, `new_layout`, `base_mip`, `level_count`, ... |
| draw, dispatch, copy, blit, clear, bind calls | their arguments, plus the subresources touched |
| `vkQueueSubmit` (also used for `vkQueueSubmit2`) | `queue`, `submits`: list of `{wait: [{semaphore, value, stage}], command_buffers, signal: [{semaphore, value}]}`, `fence` |
| `vkCreateSemaphore` | `type` (`binary` or `timeline`), `initial_value` |
| `vkCreateFence` | `signaled` |
| `vkWaitForFences` | `fences`, `wait_all`, `timeout` (`"infinite"` or nanoseconds) |
| `vkResetFences` | `fences` |
| `vkAcquireNextImageKHR` | `swapchain`, `semaphore`, `fence`, `timeout`, `image_index` |
| `vkQueuePresentKHR` | `queue`, `wait`, `swapchains`: list of `{swapchain, image_index}` |
| `vkSetDebugUtilsObjectNameEXT` | `object`, `object_type`, `name` |
| `host_write` | `memory`, `offset` (first dirty page), `offsets`, `page_size`, `fault_tids`, `fault_tids_overflow`, `call_tids`, `threads`; `writes`: `[[byte offset, hex]]`, the bytes that changed since the last flush (mapped memory starts zeroed, so this reconstructs the full contents; at most 256 KiB of content per record. Past that budget runs are `[byte offset, "", length]` without content and the record has `content_truncated: true`; only past 2^20 runs is it cut, with `writes_truncated: true` and `truncated_at` (the first byte offset not diffed). Older traces may carry only `writes_truncated: true`, cut at 256 KiB) |
| `vkCmdCopyImageToBuffer` | `src`, `dst`, `regions`: `{buffer_offset, row_length, image_height, extent, image}` |
| `covenant_readback_fill` | at submit, an image -> host-visible buffer copy has no GPU to produce its data, so the destination gets a probe ramp in the image's format (R8G8B8A8 / B8G8R8A8 UNORM or SRGB, RGBA16F, RGBA32F): the texel in column c holds `RAMP[c % 48]` in R, G and B (alpha 1), RAMP being `covenant/transfer.py`'s ramp (16 log-spaced values in [1e-3, 0.1) then 32 evenly spaced in [0.1, 1]). Fields: `cb`, `image`, `buffer`, `memory`, `offset`, `format`, `texel_size`, `extent`, `row_length`, `ramp_period`. `COVENANT_NO_READBACK_FILL=1` turns it off |
| any call that is only stubbed | `{"warning": "unimplemented"}`; a `vkCmd*` stub also records `cb`, so the loader knows whether and where it was submitted |

`vkCmdSetViewport` records `viewports` in the same form; `vkCmdSetViewportWithCount(EXT)` is recorded as `vkCmdSetViewport` with `with_count: true`.

The `vkCmdSet*` dynamic-state calls in `gen.py`'s `HARMLESS_DYNAMIC_STATE` (state that cannot add a
memory access beyond what the loader already assumes) record `cb` and no warning. Every other stubbed
`vkCmd*` (for example `vkCmdSetRenderingAttachmentLocationsKHR`, `vkCmdSetDescriptorBufferOffsetsEXT`)
records `cb` and the warning.

## Ray tracing, mesh shading and events

Recorded (all with `cb` where they are commands):
- `vkCreate/DestroyAccelerationStructureKHR` (`acceleration_structure`, `buffer`, `offset`, `size`, `type`,
  `address`),
  `vkGetAccelerationStructureDeviceAddressKHR`, `vkGetBufferDeviceAddress*`.
- `vkCmdBuildAccelerationStructuresKHR` / `...IndirectKHR`: `builds`, each with `type`, `mode`, `flags`,
  `src`, `dst`, `scratch` (address range), `geometries` (input addresses) and `primitive_count` (or the
  `indirect` range); `vkCmdCopyAccelerationStructure*KHR`,
  `vkCmdCopyAccelerationStructureToMemoryKHR`, `vkCmdCopyMemoryToAccelerationStructureKHR`,
  `vkCmdWriteAccelerationStructuresPropertiesKHR`.
- Host builds and copies (`vkBuildAccelerationStructuresKHR`, `vkCopyAccelerationStructureKHR`, ...).
- `covenant_tlas_refs`: at submit, for each TLAS build whose instance buffer is host-visible, the BLAS
  addresses its instances point at (so a trace or ray query can read the BLASes through the TLAS).
- `vkCreateRayTracingPipelinesKHR` (stages, groups, libraries), `vkGetRayTracingShaderGroupHandlesKHR`,
  `vkCmdTraceRaysKHR` / `...IndirectKHR` / `...Indirect2KHR` (SBT regions as addresses).
- `vkCmdDrawMeshTasksEXT` / `...IndirectEXT` / `...IndirectCountEXT`, `vkCmdDrawIndirectCount`,
  `vkCmdDrawIndexedIndirectCount`, `vkCmdDispatchBase`.
- Events: `vkCmdSetEvent(2)`, `vkCmdResetEvent(2)`, `vkCmdWaitEvents(2)` (same barrier fields as
  `vkCmdPipelineBarrier(2)`, plus `events`), and the host `vkSetEvent` / `vkResetEvent` /
  `vkGetEventStatus`.

`examples/triangle.cpp --events` uses `vkCmdSetEvent2` + `vkCmdWaitEvents2` for the present
transition; `--bug-event-early` sets the event before the draw (the checker must report it).

Programs that load both `libvulkan.so` and `libvulkan.so.1` (the Khronos Vulkan-Samples framework,
volk) need them to be the same file: the driver aborts on a second copy. The build makes copies (some file systems support
neither symlinks nor hard links); on a local file system run
`ln -f libcovvk.so libvulkan.so.1; ln -f libcovvk.so libvulkan.so` in `build/` to make them one file.

Building against an older system header (for example 1.3.275) fails on an unrelated command
(`vkCmdPushDescriptorSet2`); build with Vulkan-Headers v1.4.365:
`make VKH=<vulkan-headers>/include/vulkan/vulkan_core.h VKINC=<vulkan-headers>/include all`.

## Known limits

- Only the executed path is covered, because host control flow is observed by running the program.
- Descriptors are resolved when the command buffer is submitted. Update-after-bind is approximated. Descriptor update templates, push descriptors, descriptor buffers and buffer device address accesses made by shaders are not tracked: they are recorded as unimplemented or ignored, and the checker lists them in `model.unsupported` (a stubbed command that is submitted becomes a `timing.unmodelled-command` warning).
- The loader tracks image layouts per mip level (`<image>.mip<k>`) but not per array layer.
- Implicit render-pass transitions become barriers that cover everything; subpass dependencies are not checked.
- Draws treat storage resources as read and dispatches treat them as written. With `COVENANT_SPIRV_DIR` set, the SPIR-V `NonWritable` decorations refine this. Every binding of the bound sets is assumed to be used, without static-use analysis.
- Host reads of mapped memory are not seen, only writes. Writes made after the last Vulkan call are flushed at exit.
- There are no window-system surfaces (xcb, xlib, wayland); programs must use `VK_EXT_headless_surface`.
- The driver does not act as a loader ICD: it replaces `libvulkan` and does not plug into it.
