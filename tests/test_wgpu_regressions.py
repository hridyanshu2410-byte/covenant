"""Regression tests on real wgpu traces.

Counter query  vkGetSemaphoreCounterValue is a host observation of a timeline value (wgpu learns
               completion only this way); without it every host write and command-buffer reuse
               reads as racing.
Whole image    a layout map entry for a whole image is bounded by the image's extent, so an image
               transitioned layer by layer does not still read as partly in its initial layout
               (wgpu_shadow wrong-layout, wgpu_skybox stale-barrier).
X8_D24         X8_D24_UNORM_PACK32 is a depth format, not a color format (wgpu_skybox barrier-scope).

Coverage: vkCmdClearAttachments and vkCmdCopyQueryPoolResults are modelled (not only recorded, which
would leave a program using them outside the premises), tested on the model-driver triangle example.

Fixtures (tests/fixtures/wgpu): traces of the wgpu examples cube, shadow and skybox, on which a checker
without these three rules reports 43, 193 and 63 report groups (observed capture), all false alarms.
"""
import gzip
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from covenant import timing  # noqa: E402
from covenant.diagnostics import Report  # noqa: E402
from covenant.host_trace import load_trace  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "wgpu"


def _check(name):
    tmp = Path(tempfile.mkdtemp(prefix="cov-wgpu-"))
    try:
        tr = tmp / "trace.jsonl"
        tr.write_bytes(gzip.decompress((FIX / f"{name}.trace.jsonl.gz").read_bytes()))
        old = os.environ.get("COVENANT_HOST_CAPTURE")
        os.environ["COVENANT_HOST_CAPTURE"] = "observed"
        try:
            m = load_trace(str(tr), spirv_dir=str(FIX / f"{name}.spv"))
        finally:
            if old is None:
                os.environ.pop("COVENANT_HOST_CAPTURE", None)
            else:
                os.environ["COVENANT_HOST_CAPTURE"] = old
        r = timing.check(m.frame, Report(), window=False)
        return [d for d in r.items if getattr(d, "code", "").startswith("timing.")
                and d.code not in ("timing.unmodelled-command", "timing.unmodelled")]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class WgpuTraces(unittest.TestCase):
    def test_counter_query_is_a_host_wait(self):
        self.assertEqual([d.code for d in _check("wgpu_cube")], [])

    def test_layer_by_layer_transitions_cover_the_image(self):
        self.assertEqual([d.code for d in _check("wgpu_shadow")], [])

    def test_x8_d24_is_depth(self):                       # whole-image layouts and X8_D24 as depth
        self.assertEqual([d.code for d in _check("wgpu_skybox")], [])

    def test_whole_region_uses_extent(self):
        lay = timing._Layouts({"img": "undefined"}, {"img": ((0, 10),)})
        lay.set("img", ((0, 2),), "a")
        lay.set("img", ((2, 10),), "a")
        self.assertEqual(lay.get("img", None), ["a"])
        lay2 = timing._Layouts({"img": "undefined"})             # no extent known: unbounded interval
        lay2.set("img", ((0, 2),), "a")
        self.assertIn("undefined", lay2.get("img", None))


import test_host_trace as _th  # noqa: E402


class ClearAndQueryCopyCoverage(_th.ModelDriverTriangle):
    def test_correct_program_has_no_timing_diagnostics(self):   # inherited tests are not rerun here
        pass

    test_missing_fence_wait_races_on_uniform_buffer = test_host_model_contents = \
        test_spirv_dump_and_reflection = test_correct_program_has_no_timing_diagnostics

    def _unmodelled(self, m):
        return [w for w in m.warnings if "not modelled" in str(w)]

    def test_clear_attachments_is_a_color_write(self):
        m = self.load("clear", "--clear-only")
        r = timing.check(m.frame)
        self.assertTrue(r.ok, str(r))
        self.assertEqual(self._unmodelled(m), [])
        uses = [op for op in m.frame.queues["graphics0"] if isinstance(op, timing.Use) and "(clear)" in op.where]
        self.assertEqual(len(uses), 6)
        self.assertTrue(all((u.stage, u.flag) == ("COLOR_ATTACHMENT_OUTPUT", "COLOR_ATTACHMENT_WRITE") for u in uses))

    def test_clear_attachments_outside_the_event_scope_races(self):
        m = self.load("clear-bug", "--clear-only", "--bug-event-early")
        r = timing.check(m.frame)
        self.assertTrue(any("(clear)" in str(d) for d in r.items), str(r))

    def test_query_copy_is_modelled(self):
        m = self.load("query", "--query-copy")
        r = timing.check(m.frame)
        self.assertTrue(r.ok, str(r))
        self.assertEqual(self._unmodelled(m), [])

    def test_query_copy_into_vertices_races_with_next_draw(self):
        m = self.load("query-bug", "--bug-query-copy-vertices")
        r = timing.check(m.frame)
        self.assertTrue(any("vertices" in str(d) and "vkCmdCopyQueryPoolResults" in str(d)
                            for d in r.items), str(r))


if __name__ == "__main__":
    unittest.main()
