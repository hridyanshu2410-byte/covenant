"""Reference semantics (covenant/timing_ref.py) vs the
checker (covenant/timing.py).

Theorem 1 (soundness): every race of the reference is reported by the
exhaustive (pairwise) checker.  The epoch checker meets criterion E: every
racing pair has at least one of its two events flagged, and every pair it
reports is also reported pairwise.  Without region overlap the checker is
also complete (no stricter pairs), so the tests check equality there.

The unit tests at the end are checker bugs the reference found during
development (bugs A-H); each one is a small frame on which an earlier checker
was silent.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from covenant import timing, timing_ref  # noqa: E402
from covenant.diagnostics import Report  # noqa: E402
from covenant.timing import Barrier, Frame, Signal, Use, Wait  # noqa: E402
from timing_gen import random_frame  # noqa: E402

F = frozenset
LT, LTW = F({"LAYOUT_TRANSITION"}), F({"LAYOUT_TRANSITION_WRITE"})
CA, CAW = "COLOR_ATTACHMENT_OUTPUT", "COLOR_ATTACHMENT_WRITE"
FS = "FRAGMENT_SHADER"


def spec_pairs(fr, relaxed, kill=False):
    return {frozenset((a.key, b.key)) for a, b in timing_ref.Reference(fr, relaxed, kill).races()}


def checker_pairs(fr, relaxed, exhaustive, kill=False):
    hz = []
    timing.check(fr, Report(), hazards=hz, exhaustive=exhaustive, discard_relaxed=relaxed, discard_kill=kill)
    return {frozenset((a, b)) for a, b, _ in hz}


class RandomFrames(unittest.TestCase):
    SEEDS = int(os.environ.get("COVENANT_REF_SEEDS", "150"))

    def run_opts(self, exact=True, relaxed=False, kill=False, **opts):
        for seed in range(self.SEEDS):
            fr = random_frame(seed, **opts)
            spec = spec_pairs(fr, relaxed, kill)
            full = checker_pairs(fr, relaxed, True, kill)
            ep = checker_pairs(fr, relaxed, False, kill)
            with self.subTest(seed=seed, opts=opts):
                self.assertFalse(spec - full, "Theorem 1: race the checker misses")
                if exact:
                    self.assertFalse(full - spec, "checker stricter than the reference")
                self.assertFalse(ep - full, "epoch pair not reported pairwise")
                flagged = {k for p in ep for k in p}
                self.assertFalse([p for p in spec if not p & flagged], "criterion E")

    def test_default(self):
        # the default (verified) configuration is spec-strict
        self.run_opts()

    def test_heuristic_discard(self):
        # the opt-in heuristic R1 (COVENANT_HEURISTIC_DISCARD=1): checker and
        # reference relax the same rule, so they must still agree
        self.run_opts(relaxed=True)

    def test_discard_kill(self):
        self.run_opts(relaxed=True, kill=True)
        self.run_opts(relaxed=True, kill=True, exact=False, regions=True)

    def test_signal_stage_masks(self):
        # sync2 signal stage masks on every hop, relays through a third queue
        # and through the host
        self.run_opts(sig_masks=True)
        self.run_opts(sig_masks=True, queues=("host", "q0", "q1", "q2"))
        self.run_opts(sig_masks=True, queues=("host", "q0", "q1", "q2"), n_ops=80)

    def test_host_signalled_waits(self):
        # device waits on host-signalled values at restricted stage masks
        self.run_opts(host_sig_waits=True)
        self.run_opts(host_sig_waits=True, sig_masks=True, queues=("host", "q0", "q1", "q2"), n_ops=80)

    def test_regions(self):
        # Theorem 1' gap: a barrier region must contain the whole earlier
        # access region to capture it, so the checker may be stricter here
        self.run_opts(exact=False, regions=True)

    def test_same_command(self):
        self.run_opts(cmd=True)

    def test_load_store_operations(self):
        self.run_opts(passes=True, cmd=True)

    def test_atomic_accesses(self):
        self.run_opts(atomics=True, cmd=True)

    def test_unknown_stage_and_flag(self):
        self.run_opts(none_stage=2, none_flag=True)

    def test_long(self):
        self.run_opts(n_ops=80)

    def test_framebuffer_local(self):
        # BY_REGION / VIEW_LOCAL barriers and framebuffer tokens
        self.run_opts(fb=True)
        self.run_opts(fb=True, passes=True, cmd=True, none_stage=1)
        self.run_opts(fb=True, sig_masks=True, queues=("host", "q0", "q1", "q2"), n_ops=80)

    def test_automatic_transitions(self):
        # R-LT (transitions ordered among themselves) only when one transition is an image memory barrier's
        self.run_opts(auto_lt=True)
        self.run_opts(auto_lt=True, fb=True, n_ops=80)


class Bugs(unittest.TestCase):
    """Each frame: the checker equals the reference and reports `code`."""

    def check_frame(self, ops, code, queues=None):
        fr = Frame(queues=queues or {"gfx": ops})
        spec = spec_pairs(fr, False)
        self.assertTrue(spec)
        self.assertEqual(checker_pairs(fr, False, True), spec)
        codes = {d.code for d in timing.check(fr, Report()).items}
        self.assertIn(code, codes)

    def test_A_transition_checked_against_later_access(self):
        self.check_frame([
            Barrier("img", "general", "shader_read", "barrier", F({"TOP_OF_PIPE"}), F(), F({"TOP_OF_PIPE"}), F()),
            Use("img", "read", None, "sample", FS, "SHADER_SAMPLED_READ"),
        ], "timing.barrier-scope")

    def test_B_transition_vs_unordered_cross_queue_access(self):
        self.check_frame(None, "timing.cross-queue-race", queues={
            "gfx": [Barrier("img", "general", "shader_read", "barrier", F({"ALL_COMMANDS"}), F(),
                            F({"ALL_COMMANDS"}), F({"SHADER_READ"}))],
            "comp": [Use("img", "read", None, "dispatch", "COMPUTE_SHADER", "SHADER_STORAGE_READ")]})

    def test_C_wait_chain_through_barrier_on_other_resource(self):
        self.check_frame(None, "timing.wait-stage-scope", queues={
            "gfx": [Use("a", "write", None, "draw", CA, CAW), Signal("s", "signal")],
            "comp": [Wait("s", "wait", F({FS})),
                     Barrier("b", None, None, "barrier", F({FS}), F({"SHADER_READ"}),
                             F({"TRANSFER"}), F({"TRANSFER_READ"})),
                     Use("a", "read", None, "copy", "TRANSFER", "TRANSFER_READ")]})

    def test_D_wait_with_empty_stage_mask_orders_nothing(self):
        self.check_frame(None, "timing.cross-queue-race", queues={
            "gfx": [Use("a", "write", None, "draw", CA, CAW), Signal("s", "signal")],
            "comp": [Wait("s", "wait", F()),
                     Barrier(None, None, None, "barrier", F({"ALL_COMMANDS"}), F({"MEMORY_WRITE"}),
                             F({"TRANSFER"}), F({"TRANSFER_READ"})),
                     Use("a", "read", None, "copy", "TRANSFER", "TRANSFER_READ")]})

    def test_E_waw_needs_later_write_in_dst_access_scope(self):
        self.check_frame([
            Use("a", "write", None, "draw", CA, CAW),
            Barrier("a", None, None, "barrier", F({CA}), F({CAW}), F({FS}), F({"SHADER_SAMPLED_READ"})),
            Use("a", "write", None, "store", FS, "SHADER_STORAGE_WRITE"),
        ], "timing.barrier-scope")

    def test_F_shader_read_excludes_uniform_read(self):
        self.check_frame([
            Use("a", "write", None, "copy", "TRANSFER", "TRANSFER_WRITE"),
            Barrier("a", None, None, "barrier", F({"TRANSFER"}), F({"TRANSFER_WRITE"}), F({FS}), F({"SHADER_READ"})),
            Use("a", "read", None, "ubo", FS, "UNIFORM_READ"),
        ], "timing.barrier-scope")

    def test_G_unknown_stage_is_not_in_every_scope(self):
        self.check_frame([
            Use("a", "write", None, "copy", "TRANSFER", "TRANSFER_WRITE"),
            Barrier("a", None, None, "barrier", F({"TRANSFER"}), F({"TRANSFER_WRITE"}), F({FS}), F({"SHADER_READ"})),
            Use("a", "read", None, "unknown"),
        ], "timing.barrier-scope")

    def test_H_all_commands_does_not_contain_an_earlier_transition(self):
        self.check_frame([
            Barrier("img", "general", "shader_read", "t", F({"TOP_OF_PIPE"}), F(), F({"BOTTOM_OF_PIPE"}), F()),
            Barrier(None, None, None, "full", F({"ALL_COMMANDS"}), F({"MEMORY_WRITE"}),
                    F({"ALL_COMMANDS"}), F({"MEMORY_READ"})),
            Use("img", "read", None, "sample", FS, "SHADER_SAMPLED_READ"),
        ], "timing.barrier-scope")

    def test_meet_keeps_explicit_transition_stage(self):
        # an ALL_COMMANDS struct earlier in the chain must not hide the
        # explicit LAYOUT_TRANSITION src stage of a later barrier (elide's
        # B1 ; LT ; B2 split); without it the checker was stricter than spec
        ops = [
            Use("img", "write", None, "draw", CA, CAW),
            Barrier(None, None, None, "full", F({"ALL_COMMANDS"}), F({"MEMORY_WRITE"}),
                    F({"ALL_COMMANDS"}), F({"MEMORY_READ", "MEMORY_WRITE"})),
            Barrier("img", None, None, "b1", F({CA}), F({CAW}), LT, LTW, call="c", struct=0),
            Use("img", "write", None, "lt", "LAYOUT_TRANSITION", "LAYOUT_TRANSITION_WRITE", call="c", struct=0),
            Barrier("img", None, None, "b2", LT, LTW, F({FS}), F({"SHADER_SAMPLED_READ"}), call="c", struct=0),
            Use("img", "read", None, "sample", FS, "SHADER_SAMPLED_READ"),
        ]
        fr = Frame(queues={"gfx": ops})
        self.assertEqual(checker_pairs(fr, False, True), spec_pairs(fr, False))
        self.assertEqual(timing.check(fr, Report()).items, [])


if __name__ == "__main__":
    unittest.main()
