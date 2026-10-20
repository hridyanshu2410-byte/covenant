| # | Case | Category | Expected | Buggy variant reports | Fixed variant | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Texture decoded twice (sRGB view plus manual decode) | meaning | meaning.color-mismatch | meaning.color-mismatch | clean | detected |
| 2 | Video background sampled without sRGB decode | meaning | meaning.color-mismatch | meaning.color-mismatch | clean | detected |
| 3 | Baked linear lightmap read through an sRGB view | meaning | meaning.color-mismatch | meaning.color-mismatch | clean | detected |
| 4 | Grayscale sRGB image uploaded with a linear format | meaning | meaning.color-mismatch | meaning.color-mismatch | clean | detected |
| 5 | Vertex normals supplied in view space, used as object space | meaning | meaning.frame-mismatch | meaning.frame-mismatch | clean | detected |
| 6 | Bake pass keeps forward-Z depth test under a reverse-Z projection | meaning | meaning.convention-mismatch | meaning.convention-mismatch | clean | detected |
| 7 | Shadow bias sign left from forward-Z under reverse-Z | meaning | (none: known miss) | - | clean | miss (expected) |
| 8 | Host struct ignores vec3 alignment of a uniform block | structure | structure.host-layout | structure.host-layout | clean | detected |
| 9 | Previous material's descriptor set bound for interleaved draws | structure | (none: known miss) | - | clean | miss (expected) |
| 10 | Descriptor set 1 left unbound on one draw path | structure | structure.missing-binding | structure.missing-binding | clean | detected |
| 11 | Bindless material heap not bound for one pass | structure | structure.missing-binding | structure.missing-binding | clean | detected |
| 12 | OIT fragment list resolved without a barrier after the append pass | timing | timing.missing-barrier | timing.missing-barrier | clean | detected |
| 13 | Camera uniform buffer shared across frames in flight | timing | timing.cross-queue-race | timing.cross-queue-race | clean | detected |
| 14 | Mip generation barrier uses the read access mask instead of the write | timing | timing.barrier-scope | timing.barrier-scope | clean | detected |
| 15 | Compute-written vertex buffer read by a draw with the wrong barrier scope | timing | timing.barrier-scope | timing.barrier-scope | clean | detected |
| 16 | Acquired swapchain images not given back on a skipped frame (freeze) | timing | timing.swapchain-overacquire | timing.swapchain-overacquire | clean | detected |
| 17 | Render-finished semaphore indexed by frame in flight instead of by swapchain image | timing | timing.present-semaphore-reuse | timing.present-semaphore-reuse | clean | detected |
| 18 | Two contexts on different queues use the same texture with no GPU-side ordering | timing | timing.cross-queue-race | timing.barrier-scope, timing.command-buffer-in-use, timing.cross-queue-race | clean | detected |
| 19 | Queue family ownership acquire on the compute queue without a matching release | timing | timing.ownership-unmatched-acquire | timing.ownership-unmatched-acquire | clean | detected |
| 20 | Upload-queue semaphore signal scope ALL_TRANSFER does not cover the release barrier's layout transition | timing | timing.signal-scope | timing.signal-scope, timing.signal-stage-scope | clean | detected |
| 21 | Reversed-depth pass starts from depth 1.0 (the forward-Z far value) | meaning | meaning.convention-mismatch | meaning.convention-mismatch | clean | detected |
| 22 | GLM_FORCE_DEPTH_ZERO_TO_ONE defined after the GLM include, so projections use OpenGL's [-1,1] depth range | meaning | meaning.convention-mismatch | meaning.convention-mismatch | clean | detected |
| 23 | Composite shader gamma-encodes with a uniform gamma of 2.2 before an sRGB surface encodes again | meaning | meaning.color-mismatch | meaning.color-mismatch | clean | detected |
| 24 | Albedo textures allocated as RGBA8_UNORM instead of RGBA8_SRGB | meaning | meaning.color-mismatch | meaning.color-mismatch | clean | detected |
| 25 | Host and shader declare SkyboxUniforms fields in different order (f32 before mat4) | structure | structure.host-layout | structure.host-layout | clean | detected |
| 26 | One vkCmdPushConstants with VERTEX&#124;FRAGMENT over ranges declared per stage | structure | structure.push-constant-range | structure.push-constant-range | clean | detected |
| 27 | Texture upload barrier targets only the fragment stage; a vertex shader samples the texture | timing | timing.barrier-scope | timing.barrier-scope | clean | detected |
| 28 | Depth made read-only for attachment reads only; soft particles also sample it in the fragment shader | timing | timing.barrier-scope | timing.barrier-scope | clean | detected |
| 29 | Second render pass to the same target has no barrier after the first | timing | timing.missing-barrier | timing.missing-barrier | clean | detected |
| 30 | Render-finished semaphores allocated and indexed per frame in flight, not per swapchain image | timing | timing.present-semaphore-reuse | timing.present-semaphore-reuse | clean | detected |
| 31 | Storage image written on the compute family and sampled on the graphics family with EXCLUSIVE sharing and no ownership transfer | timing | timing.ownership-missing | timing.ownership-missing | clean | detected |
| 32 | Swapchain acquire barrier has source stage NONE, so its layout transition does not chain off the image-available semaphore wait | timing | timing.wait-stage-scope | timing.acquire-first-use, timing.wait-stage-scope | clean | detected |
| 33 | View-position reconstruction maps depth to [-1,1] (WebGL) although the reversed-depth projection writes [0,1] | meaning | meaning.convention-mismatch | meaning.convention-mismatch | clean | detected |
| 34 | GUI material's _GammaOutput flag skips the sRGB decode of the GUI texture before an sRGB framebuffer encodes again | meaning | meaning.color-mismatch | meaning.color-mismatch | clean | detected |
| 35 | Scene-graph matrix stack pushes projection/modelview to a fullscreen pipeline whose layout only declares a fragment push-constant block | structure | structure.push-constant-range | structure.push-constant-range | clean | detected |
| 36 | Per-frame depth layout transition uses TOP_OF_PIPE and no access, unordered against the previous frame's depth writes | timing | timing.barrier-scope | timing.barrier-scope | clean | detected |
| 37 | Explicit f32 padding slot after nine scalars does not align vec4 tint_color: shader reads it at 48, host writes it at 40 | structure | structure.host-layout | structure.host-layout | clean | detected |
| 38 | Render-target viewer frees a texture's descriptor set on resize while the previous frame's command buffer still uses it | timing | timing.free-while-in-use | timing.free-while-in-use | clean | detected |
| 39 | Texture Skia renders into is sampled by the sprite batch while still in COLOR_ATTACHMENT_OPTIMAL | timing | timing.wrong-layout | timing.wrong-layout | clean | detected |
| 40 | Compositor's Normal blend multiplies a premultiplied layer's RGB by its alpha again | meaning | meaning.convention-mismatch | meaning.convention-mismatch | clean | detected |
| 41 | Culling frustum passes the field of view in degrees to tan(); the projection converts it to radians | meaning | meaning.unit-mismatch | meaning.unit-mismatch | clean | detected |
| 42 | Prepass texture layout entries zipped onto bindings 17-20, so with the normal prepass off motion vectors land on the normal slot | structure | structure.binding-not-in-layout | structure.binding-not-in-layout | clean | detected |
| 43 | Frame pass's outgoing dependency covers fragment-shader reads only; the readback copy races it, and the next frame's pass races the copy | timing | timing.barrier-scope | timing.barrier-scope | clean | detected |
| 44 | Descriptor cache skips rebinding identical buffer writes for a pipeline whose layout differs only in push constants | structure | structure.incompatible-set-layout | structure.incompatible-set-layout | clean | detected |
| 45 | Offscreen render passes declare only an incoming dependency; the final-layout transition and attachment writes are not ordered before the next pass samples them | timing | timing.barrier-scope | timing.barrier-scope | clean | detected |
| 46 | Offscreen item atlas transitioned to COLOR_ATTACHMENT_OPTIMAL but rendered in GENERAL, and the render transition does not wait for the previous batch's attachment writes | timing | timing.wrong-layout, timing.barrier-scope | timing.barrier-scope, timing.stale-barrier, timing.wrong-layout | clean | detected |
| 47 | Image-acquired semaphore of a frame slot is re-signalled by vkAcquireNextImageKHR before waiting for the slot's fence | timing | (none: known miss) | timing.acquire-semaphore-in-use | clean | missed |
| 48 | Barriers after copies give TRANSFER_WRITE as the source access with shader stages or BOTTOM_OF_PIPE as the source stage | timing | timing.barrier-scope | timing.barrier-scope | clean | detected |
| 49 | Window-space view cache overflow frees every cached descriptor set while earlier ring command buffers still use them | timing | timing.free-while-in-use | timing.free-while-in-use | clean | detected |
| 50 | Layout transition of a D24_UNORM_S8_UINT attachment names only the stencil aspect | timing | (none: known miss) | - | clean | missed |
| 51 | HDR10 output pass applies exposure, tone map and PQ to an intermediate that holds sRGB-encoded BT.709 values | meaning | meaning.color-mismatch | meaning.color-mismatch | clean | detected |
| 52 | BT.709 -> BT.2020 matrix in the HDR ImGui shader is applied transposed on Vulkan | meaning | meaning.convention-mismatch | meaning.convention-mismatch | clean | detected |
| 53 | Sprites drawn into a transparent render target with SRC_ALPHA blending for alpha, then layered with SRC_ALPHA again | meaning | meaning.convention-mismatch | meaning.convention-mismatch | clean | detected |
| 54 | Precompiled SPIR-V flips position.y and the Vulkan backend also sets a negative-height viewport | meaning | meaning.convention-mismatch | meaning.convention-mismatch | clean | detected |
| 55 | Route projection puts east on +x and north on +z in a right-handed world, mirroring the map | meaning | meaning.convention-mismatch | - | clean | missed |
| 56 | HDR 2D viewport (linear RGBA16F) read back and quantized to 8-bit PNG without sRGB encoding | meaning | meaning.color-mismatch | meaning.color-mismatch | clean | detected |
| 57 | Hand-written float offsets ignore the 16-byte alignment of vec4 slicePlane after a new f32 field | structure | structure.host-layout | structure.host-layout | clean | detected |
| 58 | Swapchain framebuffers created with height = width | structure | (none: known miss) | structure.extent-mismatch | clean | missed |
| 59 | Fragment-stage push constant update at offset 64 into a layout whose only range is vertex-stage | structure | structure.push-constant-range | structure.push-constant-range | clean | detected |
| 60 | Swapchain image rendered and presented while still in VK_IMAGE_LAYOUT_UNDEFINED | timing | timing.wrong-layout | timing.wrong-layout | clean | detected |
| 61 | Present-wait semaphore indexed by the frame number instead of the acquired swapchain image | timing | timing.present-semaphore-reuse | timing.signal-stage-scope | timing.signal-stage-scope | missed |
| 62 | Secondary-viewport swapchain transition uses srcStage TOP_OF_PIPE, not the acquire semaphore's wait stage | timing | timing.wait-stage-scope | timing.acquire-first-use, timing.wait-stage-scope | timing.acquire-first-use | detected, fixed variant not clean |
| 63 | Light-pass barrier gives the depth texture the wrong before-state, and G-buffer resources are never moved back | timing | timing.wrong-layout | timing.barrier-scope, timing.missing-barrier, timing.stale-barrier, timing.wrong-layout | clean | detected |
| 64 | Render pass colour dependency with srcAccessMask 0: the shared multisampled target's clear races the previous frame's writes | timing | timing.barrier-scope | timing.barrier-scope | clean | detected |
| 65 | OPENGL_TO_WGPU_MATRIX written transposed: clip depth is not remapped from [-w, w] to [0, w] | meaning | meaning.convention-mismatch | meaning.convention-mismatch, timing.cross-queue-race | timing.cross-queue-race | detected, fixed variant not clean |
| 66 | ImGui orthographic projection assumes a lower-left clip origin on an upper-left (y-down) clip space | meaning | meaning.convention-mismatch | meaning.convention-mismatch, timing.cross-queue-race | timing.cross-queue-race | detected, fixed variant not clean |
| 67 | ImGui renders SDR sRGB colours with an 8-bit UNORM pipeline into an HDR10 (PQ, BT.2020) back buffer | meaning | meaning.color-mismatch | meaning.color-mismatch | clean | detected |
| 68 | Linear glTF baseColorFactor stored as an sRGB colour, then linearized again in the PBR shader | meaning | meaning.color-mismatch | meaning.color-mismatch, timing.cross-queue-race | timing.cross-queue-race | detected, fixed variant not clean |
| 69 | sRGB-encoded camera frames sampled through a UNORM view into a linear rgba16Float document | meaning | meaning.color-mismatch | timing.cross-queue-race | timing.cross-queue-race | missed |
| 70 | Unflipped glTF normal texture with a UV-derived tangent frame and normalScale left at (1, 1) | meaning | meaning.convention-mismatch | - | clean | missed |
| 71 | Cubemap faces loaded from URLs get the env-atlas texture state and are decoded as RGBP instead of sRGB | meaning | meaning.color-mismatch | meaning.frame-mismatch | meaning.frame-mismatch | missed |
| 72 | Packed vertex buffers draw from the mesh's first vertex, but morph targets are still looked up by the raw vertex index | structure | (none: known miss) | - | clean | missed |
| 73 | Render pipeline cache key ignores the vertex buffer layout: a stride-16 pipeline is reused for a stride-12 geometry | structure | (none: known miss) | meaning.frame-mismatch | meaning.frame-mismatch | missed |
| 74 | Shader samples a combined image sampler at set 0 binding 1 that the pipeline layout does not declare | structure | structure.binding-not-in-layout | structure.binding-not-in-layout, structure.missing-binding | clean | detected |
| 75 | Chained compute dispatches: the per-buffer dependency barrier's source stages omit COMPUTE_SHADER | timing | timing.barrier-scope | timing.barrier-scope | clean | detected |
| 76 | GPU-driven draw: compute-written indirect arguments and instance data are covered only by a shader-storage barrier | timing | timing.barrier-scope | timing.barrier-scope | clean | detected |
| 77 | Blit copies into a destination texture left in its sampled state (no transition to COPY_DEST) | timing | timing.wrong-layout | timing.missing-barrier, timing.wrong-layout | clean | detected |
| 78 | CPU reads a copy-written host-visible buffer after a fence wait with no TRANSFER -> HOST barrier | timing | timing.missing-barrier | - | clean | missed |
| 79 | Replay restores a buffer through a staging buffer: the copy out of srcBuf is not ordered after the copy that initialised it | timing | timing.missing-barrier | timing.missing-barrier | clean | detected |
| 80 | Texture-cache state tracker skips the barrier between two storage-image dispatches because layout and access mask are unchanged | timing | timing.missing-barrier | timing.missing-barrier | clean | detected |
| 81 | SDR slides written unscaled to an scRGB (Rgba16Float, 1.0 = 80 nits) HDR swapchain | meaning | meaning.color-mismatch | meaning.color-mismatch | clean | detected |
| 82 | CPU vertex decode computes the bitangent sign with the triple-product operands swapped | meaning | meaning.convention-mismatch | - | clean | missed |
| 83 | Emulated sRGB encode applied to premultiplied colour before a premultiplied-alpha blend (FL0) | meaning | meaning.convention-mismatch | meaning.convention-mismatch | clean | detected |
| 84 | HDR10 PGS subtitle palette (PQ, BT.2020) uploaded unconverted to an SDR (sRGB, BT.709) GUI | meaning | meaning.color-mismatch | - | clean | missed |
| 85 | WebGL shader addresses a screen-sized texture by gl_FragCoord, assuming a bottom-left origin, on a top-left-origin backend | meaning | meaning.convention-mismatch | - | clean | missed |
| 86 | glTF importer feeds spot cone angles in degrees to a luminance conversion that expects radians | meaning | meaning.unit-mismatch | meaning.unit-mismatch | clean | detected |
| 87 | Gaussian-splat footprint uses the x-axis focal length for both axes; viewport and projection aspect differ | meaning | (none: known miss) | - | clean | missed |
| 88 | GPU YUV->RGB shader hard-codes BT.601 for a BT.709 video stream | meaning | meaning.color-mismatch | - | clean | missed |
| 89 | Tonemap sRGB-conversion flag set on bit 7 while the shader tests bit 6 | meaning | meaning.color-mismatch | meaning.color-mismatch | clean | detected |
| 90 | Material params grew to 272 bytes but the per-material stride stayed 256 | structure | structure.host-layout | - | clean | missed |
| 91 | Fragment-shader constants placed at the unaligned end of the vertex constants | structure | (none: known miss) | - | clean | missed |
| 92 | First frame draws the menu with SDR-format pipelines inside the HDR10 swapchain pass | structure | (none: known miss) | meaning.color-mismatch | clean | missed |
| 93 | Location-indicator shader reads its drawable UBO from the global paint params binding | structure | structure.host-layout | - | clean | missed |
| 94 | SDFGI render-pass uniform set provides the material feedback buffer at binding 37 instead of 38 | structure | structure.binding-not-in-layout | structure.binding-not-in-layout, structure.missing-binding | clean | detected |
| 95 | Per-frame UNDEFINED -> depth-attachment transition of the shared depth image uses TOP_OF_PIPE / no access as its source scope with two frames in flight | timing | timing.barrier-scope | timing.barrier-scope | clean | detected |
| 96 | Concat's vkCmdCopyBuffer regions sit between compute->compute barriers that order neither the copies' reads nor their writes | timing | timing.barrier-scope | timing.barrier-scope | clean | detected |
| 97 | Async tensor copy on the transfer queue is not ordered after the producer graph the consumer backend waited for (event wait applied to the compute context only) | timing | timing.cross-queue-race | timing.cross-queue-race | clean | detected |
| 98 | Direct-rendering frame buffer returned to the decoder pool while its async upload copy is still pending | timing | timing.free-while-in-use | timing.cross-queue-race | clean | missed |
| 99 | Camera field of view given in degrees (45.0) to a perspective function that takes radians | meaning | meaning.unit-mismatch | meaning.unit-mismatch | clean | detected |
| 100 | Oklab gradient path: linear_to_srgb and srgb_to_linear have their exponents swapped | meaning | meaning.color-mismatch | - | clean | missed |
| 101 | Cursor-cell text colour loaded sRGB-encoded in the vertex shader, then re-encoded by the fragment shader that expects linear input | meaning | meaning.color-mismatch | meaning.color-mismatch | clean | detected |
| 102 | glTF positions loaded with Y negated (a mirror, not an up-axis change), compensated by clockwise front faces and a -90 degree rotation | meaning | meaning.convention-mismatch | - | clean | missed |
| 103 | Fill pattern sampled with a north-up pattern coordinate as a top-down atlas coordinate (pattern drawn upside down) | meaning | meaning.convention-mismatch | - | clean | missed |
| 104 | SVG rasterised to premultiplied RGBA uploaded unconverted into the BGRA (straight-alpha) sprite atlas | meaning | meaning.color-mismatch | - | clean | missed |
| 105 | Same render target rebound with the other flipY reuses the open pass and keeps pipelines keyed to the old front face | meaning | meaning.convention-mismatch | meaning.convention-mismatch | clean | detected |
| 106 | Screenshot rendered in the RGBA swapchain format copied byte-for-byte into a BGRA (QImage RGB32) buffer | meaning | meaning.color-mismatch | - | clean | missed |
| 107 | Limited-range conversion adds its +16/255 offset to premultiplied-alpha overlay texels, which the ONE / ONE_MINUS_SRC_ALPHA blend adds to the video | meaning | meaning.color-mismatch | meaning.convention-mismatch | clean | missed |
| 108 | 10-bit BT.2020 limited-range video converted with the 8-bit range offsets and matrix scale (16/255, 128/255, 255/219) | meaning | meaning.unit-mismatch | meaning.unit-mismatch | clean | detected |
| 109 | PQ source mapped to an SDR destination with a gamut matrix that ignores the different luminance of 1.0 (10000 nits vs 203 nits) | meaning | meaning.unit-mismatch | meaning.color-mismatch | clean | missed |
| 110 | Custom-shader cursor uniform adds the glyph's Y bearing (measured up from the cell bottom) to a y-down top-of-cell coordinate | meaning | meaning.convention-mismatch | - | clean | missed |
| 111 | Host uniform struct stores selection background before foreground while the shader's std140 block declares foreground first | structure | structure.host-layout | structure.host-layout | clean | detected |
| 112 | Per-light CPU record carries a 12-byte leading pad (92 bytes) while the shader's light array has an 80-byte stride starting at offset 16 | structure | structure.host-layout | structure.host-layout | clean | detected |
| 113 | Byte-packed C++ per-view constants put ViewInfo at 308 while the shader's ViewData block starts it on the next 16-byte register (320) | structure | structure.host-layout | structure.host-layout | clean | detected |
| 114 | UBO sync polyfill takes its offset from the wrong argument, so every uniform group is written at offset 0 of the shared batch buffer | structure | structure.host-layout | - | clean | missed |
