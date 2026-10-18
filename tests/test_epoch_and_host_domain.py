"""Epoch-mode read stand-ins under framebuffer-local dependencies, the host-domain
premise (A-HOST) and the eligibility report.

The epoch checker keeps one read per key and lets a later read stand in for an
earlier one. A framebuffer-local chain from an input-attachment read
orders only accesses at that read's framebuffer position, so it does not order
what a chain from an earlier sampled read of the same image must order. The
framebuffer token is therefore part of the read key and of the supersession
test; without it, epoch mode would miss the race below (criterion E: every racing
pair has at least one event flagged).
"""
import sys
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from covenant import timing, timing_ref  # noqa: E402
from covenant.diagnostics import Report  # noqa: E402
from covenant.timing import Frame, Use  # noqa: E402
from test_framebuffer_local import CO, CW, FS, IR, bar  # noqa: E402


def checked(frame, exhaustive):
    hz = []
    timing.check(frame, Report(), hazards=hz, exhaustive=exhaustive)
    return {frozenset((a, b)) for a, b, _ in hz}


class EpochStandInsUnderLocality(unittest.TestCase):
    TOK = ("fb", "imgA", 0, 0)

    def frame(self, same_cmd):
        cmd = object() if same_cmd else None
        ops = [Use("imgA", "read", None, "sampled", FS, "SHADER_SAMPLED_READ", cmd=cmd),
               Use("imgA", "read", None, "input", FS, IR, fb=self.TOK, cmd=cmd),
               bar({FS}, {IR}, {CO}, {CW}, local=True),
               Use("imgA", "write", None, "color", CO, CW, fb=self.TOK)]
        return Frame(queues={"gfx": ops})

    def test_criterion_e(self):
        for same_cmd in (False, True):
            with self.subTest(same_cmd=same_cmd):
                fr = self.frame(same_cmd)
                spec = timing_ref.race_keys(fr)
                self.assertTrue(spec)                     # sampled read vs color write
                full = checked(fr, True)
                ep = checked(fr, False)
                self.assertEqual(full, spec)
                self.assertFalse(ep - full)
                flagged = {k for p in ep for k in p}
                self.assertFalse([p for p in spec if not p & flagged], "criterion E")


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------- premise A-HOST
from covenant.timing import Barrier, Signal, Wait  # noqa: E402
from timing_gen import random_frame  # noqa: E402

XF = "TRANSFER"


def host_frame(*between, after_signal=()):
    """A device transfer write, then `between` on the device queue, a fence signal the
    host waits on, `after_signal` on the device queue, then a host write."""
    dev = [Use("buf", "write", None, "copy", XF, "TRANSFER_WRITE")] + list(between) + \
        [Signal("fence", "submit-end")] + list(after_signal)
    host = [Wait("fence", "vkWaitForFences"), Use("buf", "write", None, "memcpy", "HOST", "HOST_WRITE")]
    return Frame(queues={"host": host, "gfx": dev})


def premise_of(frame, exhaustive=False):
    pp = []
    rep = timing.check(frame, Report(), exhaustive=exhaustive, premise=pp)
    return pp, rep


class HostDomainPremise(unittest.TestCase):
    def host_bar(self, ds=("HOST",), da=("HOST_READ",), sa=("TRANSFER_WRITE",), where="host-bar"):
        return bar({XF}, set(sa), set(ds), set(da), where=where)

    def test_fence_only_is_reported(self):
        pp, rep = premise_of(host_frame())
        self.assertEqual(pp, [("copy", "memcpy")])
        self.assertEqual(rep.codes(), ["timing.premise-host-domain"])
        self.assertTrue(all(d.severity == "warning" for d in rep.items))
        el = rep.meta["covenant.eligibility"]
        self.assertFalse(el["guarantee_eligible"])
        self.assertEqual(el["host_domain_premise_pairs"], 1)

    def test_host_domain_barrier_quiets(self):
        for da in (("HOST_READ",), ("HOST_WRITE",), ("MEMORY_READ",)):
            for ds in (("HOST",), ("ALL_COMMANDS",)):
                with self.subTest(da=da, ds=ds):
                    pp, rep = premise_of(host_frame(self.host_bar(ds, da)))
                    self.assertEqual(pp, [])
                    self.assertTrue(rep.ok, str(rep))
                    self.assertTrue(rep.meta["covenant.eligibility"]["guarantee_eligible"])

    def test_barrier_must_hold_the_write_and_precede_the_signal(self):
        # the write is not in the barrier's first access scope
        self.assertEqual(len(premise_of(host_frame(self.host_bar(sa=("SHADER_WRITE",))))[0]), 1)
        # the barrier's second scope is not the host
        self.assertEqual(len(premise_of(host_frame(self.host_bar(ds=("TRANSFER",))))[0]), 1)
        # recorded after the signal the host waited on
        self.assertEqual(len(premise_of(host_frame(after_signal=[self.host_bar()]))[0]), 1)

    def test_chain_into_the_host_barrier(self):
        # TRANSFER_WRITE made available by a first link; the HOST link only chains
        pp, _ = premise_of(host_frame(bar({XF}, {"TRANSFER_WRITE"}, {"COMPUTE_SHADER"}, set(), where="b1"),
                                      bar({"COMPUTE_SHADER"}, set(), {"HOST"}, {"HOST_READ"}, where="b2")))
        self.assertEqual(pp, [])

    def test_device_read_needs_no_domain_operation(self):
        fr = host_frame()
        fr.queues["gfx"][0] = Use("buf", "read", None, "copy", XF, "TRANSFER_READ")
        pp, rep = premise_of(fr)
        self.assertEqual(pp, [])
        self.assertTrue(rep.ok, str(rep))

    def test_epoch_flags_every_premise_pair(self):
        # analogue of criterion E: for every pair the pairwise checker accepts under
        # A-HOST only, the epoch checker reports the host write (a premise warning, or
        # a hazard that already makes the trace non-clean). So a clean epoch result has
        # no pair relying on A-HOST. Random frames with and without HOST-domain barriers
        n = 0
        for seed in range(300):
            for queues in (("host", "q0"), ("host", "q0", "q1")):
                fr = random_frame(seed, queues=queues, n_ops=40)
                for q in queues[1:]:
                    for k, op in enumerate(fr.queues[q]):
                        if isinstance(op, Barrier) and (seed + k) % 3 == 0:
                            op.dst_stages = op.dst_stages | {"HOST"}
                            op.dst_access = op.dst_access | {"HOST_READ"}
                full, _ = premise_of(fr, True)
                pp, hz = [], []
                timing.check(fr, Report(), hazards=hz, premise=pp)
                flagged = {u for _, u in pp} | {f"{q}#{s - 1}" for _, (q, s), _ in hz}
                n += bool(full)
                with self.subTest(seed=seed, queues=queues):
                    self.assertFalse([p for p in full if p[1] not in flagged])
                    self.assertFalse(set(pp) - set(full))     # the epoch checker adds none
        self.assertGreater(n, 40)


class Eligibility(unittest.TestCase):
    def test_clean_frame_is_eligible(self):
        fr = Frame(queues={"gfx": [Use("buf", "write", None, "w", XF, "TRANSFER_WRITE")]})
        el = timing.check(fr, Report()).meta["covenant.eligibility"]
        self.assertTrue(el["guarantee_eligible"], el)

    def test_deadlock_is_not_complete(self):
        fr = Frame(queues={"gfx": [Wait("never", "w"), Use("buf", "write", None, "u", XF, "TRANSFER_WRITE")]})
        el = timing.check(fr, Report()).meta["covenant.eligibility"]
        self.assertFalse(el["checking_complete"])
        self.assertFalse(el["guarantee_eligible"])

    def test_heuristic_configuration_is_not_eligible(self):
        fr = Frame(queues={"gfx": [Use("buf", "write", None, "w", XF, "TRANSFER_WRITE")]})
        with unittest.mock.patch.dict("os.environ", {"COVENANT_HEURISTIC_DISCARD": "1"}):
            el = timing.check(fr, Report()).meta["covenant.eligibility"]
        self.assertFalse(el["verified_configuration"])
        self.assertFalse(el["guarantee_eligible"])
        with unittest.mock.patch.dict("os.environ", {"COVENANT_LEGACY_SUBPASS_FLAGS": "global"}):
            el = timing.check(fr, Report()).meta["covenant.eligibility"]
        self.assertFalse(el["guarantee_eligible"])
