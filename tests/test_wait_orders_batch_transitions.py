"""A semaphore wait whose stage mask holds every stage (ALL_COMMANDS, or
TOP_OF_PIPE, which in a second synchronization scope is ALL_COMMANDS with no
accesses) orders the layout transitions of its batch.

synchronization.adoc:441-442: "VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT specifies
all operations performed by all commands supported on the queue it is used
with"; cmdbuffers.adoc:2387-2393: the second synchronization scope of a wait
"includes every command in the same batch ... limited by pWaitDstStageMask"."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from covenant import timing, timing_ref  # noqa: E402
from covenant.diagnostics import Report  # noqa: E402
from covenant.host_trace import load_trace  # noqa: E402
from test_descriptor_ranges_and_first_acquire import _gfx_trace  # noqa: E402
from test_timing_fixes import Trace, codes  # noqa: E402


def _trace(d, wait_stage, src="VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT", present_first=True):
    p = _gfx_trace(os.path.join(d, "t.jsonl"), True, True, first_use_src=src, present_first=present_first)
    recs = [json.loads(x) for x in open(p)]
    for r in recs:
        if r["fn"] == "vkQueueSubmit":
            r["submits"][0]["wait"][0]["stage"] = [wait_stage]
    with open(p, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    return load_trace(p)


class FullScopeWaitOrdersTransitions(unittest.TestCase):
    def _codes(self, wait_stage, **kw):
        with tempfile.TemporaryDirectory() as d:
            m = _trace(d, wait_stage, **kw)
            return [x.code for x in timing.check(m.frame, Report(), window=False).items], m

    def test_all_commands_wait_orders_transition(self):
        codes, _ = self._codes("VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT")
        self.assertNotIn("timing.wait-stage-scope", codes)
        self.assertNotIn("timing.acquire-first-use", codes)

    def test_top_of_pipe_wait_orders_transition(self):
        # TOP_OF_PIPE in a second scope = ALL_COMMANDS with access 0 (synchronization.adoc:521-524)
        codes, _ = self._codes("VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT")
        self.assertNotIn("timing.wait-stage-scope", codes)
        self.assertNotIn("timing.acquire-first-use", codes)

    def test_narrow_wait_still_needs_the_chain(self):
        codes, _ = self._codes("VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT")
        self.assertIn("timing.wait-stage-scope", codes)
        self.assertIn("timing.acquire-first-use", codes)

    def test_all_graphics_is_not_every_stage(self):
        codes, _ = self._codes("VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT")
        self.assertIn("timing.wait-stage-scope", codes)

    def test_narrow_wait_with_chained_barrier_is_clean(self):
        codes, _ = self._codes("VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT",
                               src="VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT")
        self.assertNotIn("timing.wait-stage-scope", codes)
        self.assertNotIn("timing.acquire-first-use", codes)

    def test_reference_semantics_agree(self):
        # the reference semantics changed with the checker: exact agreement on both sides
        for stage, racy in (("VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT", False),
                            ("VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT", False),
                            ("VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT", True)):
            with tempfile.TemporaryDirectory() as d:
                fr = _trace(d, stage).frame
                spec = {frozenset((a.key, b.key)) for a, b in timing_ref.Reference(fr, False).races()}
                hz = []
                timing.check(fr, Report(), hazards=hz, exhaustive=True, discard_relaxed=False, window=False)
                full = {frozenset((a, b)) for a, b, _ in hz}
                self.assertEqual(spec, full, stage)
                self.assertEqual(bool(spec), racy, stage)


class ReleaseUnderFullSignal(unittest.TestCase):
    """BOTTOM_OF_PIPE in a semaphore signal's (first) scope is ALL_COMMANDS
    with no accesses (synchronization.adoc:525-528), so it covers the release
    half of an ownership transfer, whose dst stages are ignored."""

    def _codes(self, stage):
        t = Trace()
        t.image("image#1", sharing="VK_SHARING_MODE_EXCLUSIVE")
        rel = {"image": "image#1", "old_layout": "VK_IMAGE_LAYOUT_GENERAL", "new_layout": "VK_IMAGE_LAYOUT_GENERAL",
               "src_stage": ["VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT"], "dst_stage": [], "src_access": [],
               "dst_access": [], "src_family": 0, "dst_family": 1, "aspect": ["VK_IMAGE_ASPECT_COLOR_BIT"],
               "base_mip": 0, "level_count": 1, "base_layer": 0, "layer_count": 1}
        t.add("vkCreateSemaphore", semaphore="sem#1", type="binary")
        t.submit([("vkCmdPipelineBarrier2", dict(memory_barriers=[], buffer_barriers=[], image_barriers=[rel]))],
                 signal=[{"semaphore": "sem#1", "value": None, "stage": [stage]}])
        return codes(t.model().frame)

    def test_bottom_of_pipe_signal_covers_release(self):
        self.assertNotIn("timing.signal-scope", self._codes("VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT"))
        self.assertNotIn("timing.signal-scope", self._codes("VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT"))

    def test_narrow_signal_does_not(self):
        self.assertIn("timing.signal-scope", self._codes("VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT"))


if __name__ == "__main__":
    unittest.main()
