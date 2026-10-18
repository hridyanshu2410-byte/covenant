"""Render-pass final-layout transitions and repair verification.

1. A render pass's final-layout transition starts from the attachment's
   current layout, not from UNDEFINED (old = None).
2. Repair verification: report_groups verifies a repair in
   pairwise mode and by event identity, so a repair that removes the target
   pair but adds another racing pair at a site that already had a report is
   rejected. Repairs only add synchronization (wider masks, inserted barriers
   without a layout change), so by repair monotonicity they never add a race;
   the identity check makes the checker confirm it on every repaired frame."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from covenant import report_groups, timing, timing_ref  # noqa: E402
from covenant.diagnostics import Report  # noqa: E402
from covenant.host_trace import load_trace  # noqa: E402
from covenant.timing import Barrier, Frame, Use  # noqa: E402

F = frozenset


def _frame(*queues):
    fr = Frame()
    for name, ops in queues:
        fr.q(name).extend(ops)
    return fr


def _bar(res, ss, sa, ds, da, where="vkCmdPipelineBarrier @seq 1 (command_buffer#1)", old=None, new=None):
    return Barrier(res, old, new, where, src_stages=F(ss), src_access=F(sa), dst_stages=F(ds), dst_access=F(da))


# ------------------------------------------------------------------ 1. final-layout transitions
HAVE_BENCH = bool(__import__("shutil").which(os.environ.get("CXX", "clang++-20")))


@unittest.skipUnless(HAVE_BENCH, "needs the bench toolchain (clang++-20, slangc)")
class FinalLayoutStartsFromCurrentLayout(unittest.TestCase):
    def test_bench_case_45_final_layout_transition_has_old_layout(self):
        sys.path.insert(0, str(ROOT / "bench"))
        import run as bench_run
        case = ROOT / "bench" / "cases" / "45-offscreen-pass-no-outgoing-dependency"
        out, _, trace, _ = bench_run.build_variant(case, True)
        m = load_trace(str(trace), spirv_dir=str(out / "spv-dump"))
        finals = [o for ops in m.frame.queues.values() for o in ops
                  if isinstance(o, Barrier) and "render pass final layout" in o.where]
        self.assertTrue(finals, "case 45 has render-pass final-layout transitions")
        for b in finals:
            self.assertIsNotNone(b.old, b.where)
            self.assertNotEqual(b.old, b.new)


class FinalLayoutHazardIsNotFromUndefined(unittest.TestCase):
    """A write only execution-ordered before a transition from a defined layout is a
    hazard in both modes; the heuristic R1 (for transitions from UNDEFINED) no longer
    applies to final-layout transitions."""

    def _fr(self, old):
        w = Use("img", "write", "vkCmdDraw @seq 2 (command_buffer#1)", stage="COLOR_ATTACHMENT_OUTPUT",
                flag="COLOR_ATTACHMENT_WRITE")
        t = Barrier("img", old, "present", "vkCmdEndRenderPass @seq 3 (command_buffer#1) (render pass final layout)",
                    src_stages=F({"COLOR_ATTACHMENT_OUTPUT"}), src_access=F(),
                    dst_stages=F({"BOTTOM_OF_PIPE"}), dst_access=F())
        return _frame(("graphics0", [w, t]))

    def test_defined_old_layout_is_a_hazard_under_the_heuristic_too(self):
        for relaxed in (False, True):
            hz = []
            timing.check(self._fr("color_attachment"), Report(), hazards=hz, window=False,
                         discard_relaxed=relaxed, exhaustive=True)
            self.assertTrue(hz, relaxed)
        hz = []
        timing.check(self._fr(None), Report(), hazards=hz, window=False, discard_relaxed=True, exhaustive=True)
        self.assertFalse(hz)     # with old = None (a wrong model), R1 would hide it


# ------------------------------------------------------------------ 2. repair verification
class RepairVerification(unittest.TestCase):
    def _missing_barrier_frame(self):
        cb = "(command_buffer#1)"
        w = Use("buf", "write", f"vkCmdCopyBuffer @seq 2 {cb}", stage="COPY", flag="TRANSFER_WRITE")
        r = Use("buf", "read", f"vkCmdDispatch @seq 3 {cb}", stage="COMPUTE_SHADER", flag="SHADER_STORAGE_READ")
        return _frame(("graphics0", [w, r]))

    def test_default_is_pairwise_by_identity(self):
        self.assertTrue(report_groups.PAIRWISE_VERIFY)
        res = report_groups.group(self._missing_barrier_frame())
        self.assertEqual(res["repair_verify"], "pairwise, event identity")
        self.assertEqual(res["repair_groups"], 1, res)
        self.assertEqual(res["new_pair_rejects"], 0)

    def test_sites_rule_still_available(self):
        with mock.patch.object(report_groups, "PAIRWISE_VERIFY", False):
            res = report_groups.group(self._missing_barrier_frame())
        self.assertEqual(res["repair_verify"], "epoch, site keys")
        self.assertEqual(res["repair_groups"], 1, res)

    def test_repairs_are_monotone_on_random_frames(self):
        """Repair monotonicity: applying any synthesized candidate
        (widened masks, an inserted barrier without a layout change) never adds a
        racing pair under the reference semantics, by event identity."""
        try:
            from timing_gen import random_frame
        except ImportError:     # pragma: no cover
            self.skipTest("timing_gen.random_frame not available")
        checked = 0
        for seed in range(60):
            fr = random_frame(seed)
            g = report_groups.Grouper(fr)
            g.groups()
            base = timing_ref.race_keys(fr)
            for sg in g.site_groups:
                if not sg.hazards:
                    continue
                for rep in g._candidates(sg.hazards[0]):
                    f2 = g._apply(rep)
                    ident = g._ident
                    # map each reference race of f2 back to original identities
                    after = timing_ref.race_keys(f2)
                    back = {_remap(k, f2, fr, ident) for k in after}
                    self.assertLessEqual(back, {_norm(k) for k in base}, (seed, rep.describe()))
                    checked += 1
        self.assertGreater(checked, 0)


def _norm(k):
    return k


def _remap(k, f2, fr, ident):
    """Reference race keys are ((queue, pos), (queue, pos), ...) tuples of the frame
    they were computed on; translate positions of f2 to positions of fr."""
    out = []
    for part in k:
        if isinstance(part, tuple) and len(part) == 2 and part[0] in f2.queues and isinstance(part[1], int):
            op = f2.queues[part[0]][part[1] - 1]
            out.append(ident.get(id(op), part))
        else:
            out.append(part)
    return tuple(out) if not isinstance(k, frozenset) else frozenset(out)


if __name__ == "__main__":
    unittest.main()
