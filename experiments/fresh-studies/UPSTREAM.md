# Upstream status of the true causes of the two fresh studies

For each true cause: the tested revision, whether the pattern is still present upstream (revisions checked
on the day of the studies), the upstream commit that changes the pattern, and the issue it names. "Days" is the
time from that commit (or, for a cause still present, from the opening of the issue or PR) to the studies.
Commits were found with `git log -S/-G` on the lines that carry each cause, and every diff was read; a commit
is listed only when its diff changes the exact pattern. Trackers on bugreports.qt.io, gitlab.freedesktop.org,
gitlab.gnome.org and code.videolan.org could not be searched; for those only the IDs named in commit messages
are given. Nothing was reported upstream as part of this work.

| cause | project | tested revision | status | commit changing the pattern | issue / PR | days |
| --- | --- | --- | --- | --- | --- | --- |
| QT-ACQ | qtbase (QRhi Vulkan) | Qt 6.4.2 (Ubuntu pkg; tag `v6.4.2` = `e3e40c44d3f`, 2022-12-12) | **fixed** (dev `7faa468198d`) | `24d62ffd720` 2025-01-27 "rhi: vulkan: Simplify frame sync" (author date 2025-01-15) | QTBUG-132808 (`Fixes:` line) | 627 |
| QT-PSR | qtbase (QRhi Vulkan) | Qt 6.4.2 (`e3e40c44d3f`) | **fixed** (dev `7faa468198d`) | `eab1295c882` 2025-06-25 "rhi: vulkan: Avoid VUID-vkQueueSubmit-pSignalSemaphores-00067" | QTBUG-137986 | 478 |
| QT-VS | qtbase (QRhi) | Qt 6.4.2 (`e3e40c44d3f`) | **present** (dev `7faa468198d`, `qrhi.cpp:13766-13782`: "pick the earlier stage") | none | none found | n/a |
| VKCUBE-PSR | Vulkan-Tools (vkcube) | 1.3.275 (Ubuntu pkg; tag `v1.3.275` = `2c0a644d`, 2024-01-05) | **fixed** (main `a967170a`, 2026-10-11) | `a8f20787` 2025-04-11 "cube: Re-record command buffers each frame" (author date 2025-04-08) | none named | 553 |
| IMGUI-PRE | Dear ImGui (Vulkan backend helper) | `9d9a621f` (2026-10-15) | **present** (master `4be0b47ec`, 2026-10-16; `imgui_impl_vulkan.cpp:1960-1975` unchanged) | none (pattern introduced by `bbc89b639`, 2025-05-07, "fixed validation errors during window detach in multi-viewport mode", #8600/#8176) | issue **#9496** "Vulkan Image Transition Before Acquisition" (open, opened 2026-08-01). PR **#9553** "Backends: Vulkan: don't transition swapchain images before acquire" (open, 2026-09-23). Its diff deletes exactly the UNDEFINED -> PRESENT_SRC loop. | 76 (issue); 23 (PR) |
| VKMARK-PSR | vkmark | `ab6e6f3` (2023-04-12; Ubuntu 2017.08+git20220909 is source-identical to it) | **present** (master `83a128d`, 2025-09-09, no later commits) | none on master. PR #89 commit `a20e3f3` "scenes: allocate one submit semaphore per image" (2026-10-08) changes the exact pattern but is not merged. | issue **#88** "Incorrect semaphore usage" (open, 2026-10-06). PR **#89** "Fix VVL errors and improve performance" (open, 2026-10-08) | 10 (issue); 8 (PR) |
| VKMARK-ACQ | vkmark | `ab6e6f3` | **fixed** (master `83a128d`) | `956e145` 2025-01-08 "ws: Resolve swapchain related validation errors" | none named | 646 |
| VKMARK-UBO | vkmark | `ab6e6f3` | **fixed** (master `83a128d`) | `fd2d798` 2025-02-25 "scenes: Use dedicated uniform buffer per image when needed" | `Fixes #69`: "Multiple frames in flight are using the same uniform buffer, causing duplicate/skipped/misordered frames" (opened 2025-02-05, closed 2025-02-25) | 598 (fix); 618 (issue) |
| VKMARK-CLR | vkmark | `ab6e6f3` | **present** (master `83a128d`: `clear_scene.cpp:143-144,167`) | none. PR #89 keeps the TRANSFER-src barrier and the COLOR_ATTACHMENT_OUTPUT wait. | none found (GitHub search) | n/a |
| VKMARK-DEPTH | vkmark | `ab6e6f3` | **present** (master `83a128d`: `render_pass_builder.cpp:95-103`) | none (not touched by PR #89) | none found (GitHub search) | n/a |
| GST-WAIT | GStreamer (gst-plugins-bad Vulkan, vulkansink) | 1.24.2 (`2d82731515`, 2024-04-09) | **present** (main `d77a8cbc3f`, 2026-10-16: `gstvkswapper.c:1179-1181`, wait at `BOTTOM_OF_PIPE`) | none | not searchable | n/a |
| GST-IA | GStreamer (gst-plugins-bad Vulkan, fullscreenquad) | 1.24.2 (`2d82731515`) | **present** (main `d77a8cbc3f`: `gstvkfullscreenquad.c:1501-1504`) | none | not searchable | n/a |
| PLACEBO-EXT | libplacebo (used by mpv 0.37) | v6.338.2 (`64c19545`, 2024-01-22) | **fixed** (master `0df80ae5`, 2026-10-16) | `fd01c75c` 2025-03-14 "vulkan/gpu_tex: properly time layout transitions with external dependencies" (author date 2025-03-10) | `Fixes: code.videolan.org/videolan/libplacebo/-/issues/333` | 581 |
| GTK-SEM | GTK 4 (GDK Vulkan) | 4.14.5 (`320f27a202`, 2024-08-16) | **fixed** on main (`c18dbd2278`, 2026-10-16). Never fixed on `gtk-4-14`: that branch still has `draw_semaphore`. | `b08a17064e` 2024-07-14 "vulkan: Remove draw semaphore" (author date 2024-07-13) | none named | 824 |
| GTK-ACQ | GTK 4 (GSK Vulkan) | 4.14.5 (`320f27a202`) | **fixed** on main (`c18dbd2278`); not on `gtk-4-14` | `4966f8cdf8` 2024-07-17 "vulkan: Add an acquire semaphore to frames" | none named | 821 |
| GTK-PRS | GTK 4 (GDK/GSK Vulkan) | 4.14.5 (`320f27a202`) | **fixed** on main (`c18dbd2278`); not on `gtk-4-14` | `fc6581cf3b` 2025-11-12 "vulkan: Rework swapchain present implementation" | `Fixes:` GNOME/gtk#7866 and mesa/mesa#14087. Part of GNOME/gtk!9128. | 338 |
