"""Spec-strict UNDEFINED transitions, signal and wait stage masks, and host-write capture.

1. Transitions from UNDEFINED: the default is spec-strict. A write that
   precedes a layout transition must be made available to it, also when the old
   layout is UNDEFINED (synchronization.adoc, "Image Layout Transitions":
   "applications must ensure that all memory writes have been made available
   before a layout transition is executed"). The relaxation R1 (an execution
   dependency suffices before a transition from UNDEFINED) is an opt-in
   heuristic (COVENANT_HEURISTIC_DISCARD=1), outside the soundness theorem.
2. Epoch mode keeps the earlier access when the dominating access is atomic
   (an atomic write orders nothing for later non-atomic accesses).
3. Sync2 semaphore signal stage masks limit the first scope on every hop of a
   relay (third queue, host), not only on the last one; timing_ref models them.
   A device wait on a host-signalled semaphore value keeps its stage mask.
4. vkCmdPushConstants with VK_SHADER_STAGE_ALL does not abort the check.
5. Diagnostics name the scope test that failed (execution / availability /
   visibility), not the stage mask that passed.
6. Descriptor accesses are charged at the stages whose entry point uses the
   binding (SPIR-V), not every stage of the set layout's stageFlags.
7. Host-write capture: conservative by default, so a store of the bytes
   already there and a write restored within one capture interval are
   accesses; observed-modification mode is an option; writes from another
   thread are reported (they violate the single-thread capture premise)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from covenant import modes, timing, timing_ref  # noqa: E402
from covenant.diagnostics import Report  # noqa: E402
from covenant.host_trace import load_trace  # noqa: E402
from covenant.timing import Barrier, Frame, Signal, Use, Wait  # noqa: E402

F = frozenset
MD = ROOT / "model_driver"
BUILD = MD / "build"
LDSO = "/lib64/ld-linux-x86-64.so.2"


def _bar(res, old, new, ss, sa, ds, da):
    return Barrier(res, old, new, src_stages=F(ss), src_access=F(sa), dst_stages=F(ds), dst_access=F(da))


def _frame(*queues):
    fr = Frame()
    for name, ops in queues:
        fr.q(name).extend(ops)
    return fr


def _pairs(fr, relaxed=None, exhaustive=True):
    hz = []
    r = timing.check(fr, Report(), hazards=hz, exhaustive=exhaustive, window=False, discard_relaxed=relaxed)
    return {frozenset((a, b)) for a, b, _ in hz}, r


def _ref(fr, relaxed=False):
    return timing_ref.race_keys(fr, discard_relaxed=relaxed)


def _write_color():
    return Use("img", "write", stage="COLOR_ATTACHMENT_OUTPUT", flag="COLOR_ATTACHMENT_WRITE")


# ------------------------------------------------------------------ 1. UNDEFINED
class UndefinedTransitionIsSpecStrict(unittest.TestCase):
    """A minimal write / execution-only dependency / transition-from-UNDEFINED frame and its variants, on the checker (pairwise and
    epoch) and on the reference semantics."""
    P12 = frozenset({("graphics0", 1), ("graphics0", 2)})

    def _all(self, ops, relaxed):
        fr = _frame(("graphics0", ops))
        ex, r = _pairs(fr, relaxed, True)
        ep, _ = _pairs(fr, relaxed, False)
        ref = _ref(fr, bool(relaxed))
        self.assertEqual(ex, ep, "pairwise and epoch modes disagree")
        self.assertEqual(ex, ref, "checker and reference semantics disagree")
        return ex, r

    def test_minimal_case_is_a_hazard_by_default(self):
        # write; execution-only dependency (srcAccessMask 0); transition from UNDEFINED
        ops = [_write_color(), _bar("img", None, "transfer_dst", {"COLOR_ATTACHMENT_OUTPUT"}, (),
                                    {"COPY"}, {"TRANSFER_WRITE"})]
        got, r = self._all(ops, None)
        self.assertEqual(got, {self.P12})
        (d,) = [d for d in r.items if d.code == "timing.barrier-scope"]
        self.assertIn("srcAccessMask (NONE) does not include COLOR_ATTACHMENT_WRITE", d.message)
        self.assertIn("from UNDEFINED", d.message)
        self.assertIn("made available before a layout transition", d.message)

    def test_default_is_strict_and_reported_in_meta(self):
        self.assertFalse(modes.discard_heuristic())
        self.assertTrue(modes.verified())
        r = timing.check(_frame(("graphics0", [_write_color()])), Report(), window=False)
        self.assertEqual(r.meta["covenant.modes"]["discard"], "strict")

    def test_heuristic_flag_accepts_it(self):
        ops = [_write_color(), _bar("img", None, "transfer_dst", {"COLOR_ATTACHMENT_OUTPUT"}, (),
                                    {"COPY"}, {"TRANSFER_WRITE"})]
        self.assertEqual(self._all(ops, True)[0], set())
        with mock.patch.dict(os.environ, {"COVENANT_HEURISTIC_DISCARD": "1"}):
            fr = _frame(("graphics0", ops))
            got, r = _pairs(fr, None)
            self.assertEqual(got, set())
            self.assertEqual(r.meta["covenant.modes"]["discard"], "heuristic-R1")
            self.assertFalse(modes.verified())

    def test_full_memory_dependency_is_clean(self):
        ops = [_write_color(), _bar("img", None, "transfer_dst", {"COLOR_ATTACHMENT_OUTPUT"},
                                    {"COLOR_ATTACHMENT_WRITE"}, {"COPY"}, {"TRANSFER_WRITE"})]
        for relaxed in (False, True):
            self.assertEqual(self._all(ops, relaxed)[0], set())

    def test_no_execution_dependency_is_a_hazard_in_both_modes(self):
        ops = [_write_color(), _bar("img", None, "transfer_dst", {"COPY"}, {"TRANSFER_WRITE"},
                                    {"COPY"}, {"TRANSFER_WRITE"})]
        for relaxed in (False, True):
            got, r = self._all(ops, relaxed)
            self.assertEqual(got, {self.P12})
        (d,) = r.items
        self.assertIn("first synchronization scope (srcStageMask COPY) does not include "
                      "COLOR_ATTACHMENT_OUTPUT", d.message)

    def test_defined_old_layout_needs_availability_in_both_modes(self):
        ops = [_write_color(), _bar("img", "color_attachment", "transfer_dst", {"COLOR_ATTACHMENT_OUTPUT"}, (),
                                    {"COPY"}, {"TRANSFER_WRITE"})]
        for relaxed in (False, True):
            self.assertEqual(self._all(ops, relaxed)[0], {self.P12})

    def test_later_access_is_still_checked(self):
        # after the transition a read of the image needs the transition's writes visible;
        # the heuristic drops only the write -> transition pair
        ops = [_write_color(), _bar("img", None, "transfer_dst", {"COLOR_ATTACHMENT_OUTPUT"}, (),
                                    {"COPY"}, {"TRANSFER_WRITE"}),
               Use("img", "read", stage="COPY", flag="TRANSFER_READ", layout="transfer_dst")]
        strict, _ = self._all(ops, False)
        relaxed, _ = self._all(ops, True)
        p23 = frozenset({("graphics0", 2), ("graphics0", 3)})
        self.assertIn(self.P12, strict)
        self.assertNotIn(self.P12, relaxed)
        self.assertIn(p23, strict)
        self.assertIn(p23, relaxed)

    def test_semaphore_carries_the_memory_dependency(self):
        # a semaphore signal's first access scope is every memory access: across queues the
        # write is made available by the signal, so the transition needs no source access
        fr = _frame(("graphics0", [_write_color(), Signal("s")]),
                    ("compute0", [Wait("s", stages=F({"ALL_COMMANDS"})),
                                  _bar("img", None, "general", {"TOP_OF_PIPE"}, (), {"COMPUTE_SHADER"},
                                       {"SHADER_STORAGE_WRITE"})]))
        for relaxed in (False, True):
            self.assertEqual(_pairs(fr, relaxed)[0], set())
            self.assertEqual(_ref(fr, relaxed), set())

    def test_reference_default_is_strict(self):
        fr = _frame(("graphics0", [_write_color(), _bar("img", None, "transfer_dst", {"COLOR_ATTACHMENT_OUTPUT"},
                                                       (), {"COPY"}, {"TRANSFER_WRITE"})]))
        self.assertEqual(timing_ref.race_keys(fr), {self.P12})


# ------------------------------------------------------------------ 2. atomic dominator
class AtomicDominatorKeepsEarlierAccess(unittest.TestCase):
    def test_fill_compute_atomic_fragment_atomic(self):
        A = F({("atomic", None)})
        fr = _frame(("graphics0", [
            Use("counter", "write", stage="CLEAR", flag="TRANSFER_WRITE", cmd="fill"),
            _bar(None, None, None, {"TRANSFER"}, {"TRANSFER_WRITE"}, {"COMPUTE_SHADER"},
                 {"SHADER_STORAGE_READ", "SHADER_STORAGE_WRITE"}),
            Use("counter", "write", stage="COMPUTE_SHADER", flag="SHADER_STORAGE_WRITE", cmd="dispatch", atomic=A),
            Use("counter", "write", stage="FRAGMENT_SHADER", flag="SHADER_STORAGE_WRITE", cmd="draw", atomic=A)]))
        ref = timing_ref.race_keys(fr)
        self.assertEqual(ref, {frozenset({("graphics0", 1), ("graphics0", 4)})})
        # an epoch mode that drops the earlier access would report nothing here (violating
        # criterion E): the atomic compute
        # write replaced the fill as the last write although it orders nothing
        for ex in (True, False):
            self.assertEqual(_pairs(fr, False, ex)[0], ref, f"exhaustive={ex}")

    def test_non_atomic_dominator_still_drops(self):
        # control: with a plain dominating write the fill is superseded (same pairs as the reference)
        fr = _frame(("graphics0", [
            Use("counter", "write", stage="CLEAR", flag="TRANSFER_WRITE", cmd="fill"),
            _bar(None, None, None, {"TRANSFER"}, {"TRANSFER_WRITE"}, {"COMPUTE_SHADER"},
                 {"SHADER_STORAGE_READ", "SHADER_STORAGE_WRITE"}),
            Use("counter", "write", stage="COMPUTE_SHADER", flag="SHADER_STORAGE_WRITE", cmd="dispatch"),
            Use("counter", "write", stage="FRAGMENT_SHADER", flag="SHADER_STORAGE_WRITE", cmd="draw")]))
        ref = timing_ref.race_keys(fr)
        self.assertEqual(_pairs(fr, False, True)[0], ref)
        ep = _pairs(fr, False, False)[0]
        # epoch mode reports (3,4) only: (1,4) is covered by criterion E (event 4 is flagged)
        self.assertEqual(ep, {frozenset({("graphics0", 3), ("graphics0", 4)})})
        self.assertFalse([p for p in ref if not p & {k for q in ep for k in q}])


class ShadowEntryOutlivesSignal(unittest.TestCase):
    """Epoch mode: an entry dominated by a render-pass-scope access stays checkable by
    later accesses of that scope even across a Wait/Signal (random frame seed 916 of
    test_timing_ref.test_long, on which an epoch mode without this rule violates criterion E)."""

    def test_transition_depth_write_signal_depth_read(self):
        S = object()
        fr = _frame(("q1", [
            _bar("r1", "transfer_src", "shader_read", {"TOP_OF_PIPE"}, (), {"LATE_FRAGMENT_TESTS"}, {"MEMORY_WRITE"}),
            Use("r1", "write", stage="LATE_FRAGMENT_TESTS", flag="DEPTH_STENCIL_ATTACHMENT_WRITE", scope=S),
            Signal("s6"),
            Use("r1", "read", stage="EARLY_FRAGMENT_TESTS", flag="DEPTH_STENCIL_ATTACHMENT_READ", scope=S)]))
        ref = timing_ref.race_keys(fr)
        self.assertEqual(ref, {frozenset({("q1", 1), ("q1", 4)})})
        self.assertEqual(_pairs(fr, False, True)[0], ref)
        ep = _pairs(fr, False, False)[0]
        self.assertFalse([p for p in ref if not p & {k for q in ep for k in q}], "criterion E")


# ------------------------------------------------------------------ 3. signal stage masks
class SignalStageMasksOnEveryHop(unittest.TestCase):
    def _agree(self, fr):
        ref = timing_ref.race_keys(fr)
        for ex in (True, False):
            got, r = _pairs(fr, False, ex)
            self.assertEqual(got, ref, f"exhaustive={ex}")
        return ref, r

    def test_relay_through_third_queue(self):
        fr = _frame(("compute0", [Use("buf", "write", stage="COMPUTE_SHADER", flag="SHADER_STORAGE_WRITE"),
                                  Signal("a")]),
                    ("graphics0", [Wait("a", stages=F({"VERTEX_SHADER"})),
                                   Signal("b", stages=F({"COMPUTE_SHADER"}))]),
                    ("transfer0", [Wait("b"), Use("buf", "read", stage="COPY", flag="TRANSFER_READ")]))
        ref, r = self._agree(fr)
        self.assertEqual(ref, {frozenset({("compute0", 1), ("transfer0", 2)})})
        self.assertIn("timing.signal-stage-scope", r.codes())

    def test_relay_through_third_queue_with_meeting_masks_is_clean(self):
        fr = _frame(("compute0", [Use("buf", "write", stage="COMPUTE_SHADER", flag="SHADER_STORAGE_WRITE"),
                                  Signal("a")]),
                    ("graphics0", [Wait("a", stages=F({"COMPUTE_SHADER"})),
                                   Signal("b", stages=F({"COMPUTE_SHADER"}))]),
                    ("transfer0", [Wait("b"), Use("buf", "read", stage="COPY", flag="TRANSFER_READ")]))
        self.assertEqual(self._agree(fr)[0], set())

    def test_direct_masked_signal(self):
        fr = _frame(("compute0", [Use("buf", "write", stage="FRAGMENT_SHADER", flag="SHADER_STORAGE_WRITE"),
                                  Signal("a", stages=F({"COMPUTE_SHADER"}))]),
                    ("transfer0", [Wait("a"), Use("buf", "read", stage="COPY", flag="TRANSFER_READ")]))
        self.assertEqual(self._agree(fr)[0], {frozenset({("compute0", 1), ("transfer0", 2)})})

    def test_relay_through_host(self):
        fr = _frame(("graphics0", [Use("buf", "write", stage="FRAGMENT_SHADER", flag="SHADER_STORAGE_WRITE"),
                                   Signal("t=1", stages=F({"COMPUTE_SHADER"}))]),
                    ("host", [Wait("t=1"), Signal("submit@9")]),
                    ("compute0", [Wait("submit@9"),
                                  Use("buf", "read", stage="COMPUTE_SHADER", flag="SHADER_STORAGE_READ")]))
        self.assertEqual(self._agree(fr)[0], {frozenset({("graphics0", 1), ("compute0", 2)})})

    def test_covering_masked_signal_is_clean(self):
        fr = _frame(("graphics0", [Use("buf", "write", stage="FRAGMENT_SHADER", flag="SHADER_STORAGE_WRITE"),
                                   Signal("t=1", stages=F({"FRAGMENT_SHADER"}))]),
                    ("host", [Wait("t=1"), Signal("submit@9")]),
                    ("compute0", [Wait("submit@9"),
                                  Use("buf", "read", stage="COMPUTE_SHADER", flag="SHADER_STORAGE_READ")]))
        self.assertEqual(self._agree(fr)[0], set())

    def test_empty_scope_signal_relays_nothing(self):
        # a full wait relayed into a signal whose first scope holds no stage (TOP_OF_PIPE)
        fr = _frame(("compute0", [Use("buf", "write", stage="COMPUTE_SHADER", flag="SHADER_STORAGE_WRITE"),
                                  Signal("a")]),
                    ("graphics0", [Wait("a"), Signal("b", stages=F({"TOP_OF_PIPE"}))]),
                    ("transfer0", [Wait("b"), Use("buf", "read", stage="COPY", flag="TRANSFER_READ")]))
        self.assertEqual(self._agree(fr)[0], {frozenset({("compute0", 1), ("transfer0", 2)})})


class HostSignalledWaitKeepsItsStageMask(unittest.TestCase):
    """A device wait on a value the host signalled (vkSignalSemaphore) is limited by
    its pWaitDstStageMask; it is not treated as a queue submission (which would order
    every later op)."""

    def _frame(self, stages):
        return _frame(("host", [Signal("submit@1"), Use("buf", "write", stage="HOST", flag="HOST_WRITE"),
                                Signal("t=5")]),
                      ("graphics0", [Wait("submit@1"), Wait("t=5", stages=stages),
                                     Use("buf", "read", stage="VERTEX_SHADER", flag="UNIFORM_READ")]))

    def test_restricted_stage_does_not_order_other_stages(self):
        fr = self._frame(F({"FRAGMENT_SHADER"}))
        want = {frozenset({("host", 2), ("graphics0", 3)})}
        self.assertEqual(timing_ref.race_keys(fr), want)
        for ex in (True, False):
            self.assertEqual(_pairs(fr, False, ex)[0], want, f"exhaustive={ex}")

    def test_covering_stage_and_full_wait_are_clean(self):
        for st in (F({"VERTEX_SHADER"}), None, F({"ALL_COMMANDS"})):
            fr = self._frame(st)
            self.assertEqual(timing_ref.race_keys(fr), set(), st)
            for ex in (True, False):
                self.assertEqual(_pairs(fr, False, ex)[0], set(), (st, ex))


# ------------------------------------------------------------------ 4. push constants, VK_SHADER_STAGE_ALL
class _Trace:
    def __init__(self):
        self.recs = []

    def add(self, fn, **kw):
        self.recs.append(dict(seq=len(self.recs) + 1, fn=fn, **kw))

    def write(self, path):
        Path(path).write_text("".join(json.dumps(r) + "\n" for r in self.recs))


NAMED_ALL = ["VK_SHADER_STAGE_VERTEX_BIT", "VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT",
             "VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT", "VK_SHADER_STAGE_GEOMETRY_BIT",
             "VK_SHADER_STAGE_FRAGMENT_BIT", "VK_SHADER_STAGE_COMPUTE_BIT"]
STAGE_ALL = NAMED_ALL + [0x7FFFFFFF & ~0x3F]      # the driver's encoding: names + one integer remainder


def _push_trace(path, layout_stages, call_stages):
    t = _Trace()
    t.add("covenant_trace_begin", version=1, host_write_tracking=True, page_size=4096)
    t.add("vkGetDeviceQueue", queue="queue#1", family=0, index=0)
    t.add("vkCreatePipelineLayout", layout="pl#1", set_layouts=[],
          push_constants=[{"stages": layout_stages, "offset": 0, "size": 64}])
    t.add("vkAllocateCommandBuffers", pool="cp#1", level="VK_COMMAND_BUFFER_LEVEL_PRIMARY",
          command_buffers=["command_buffer#1"])
    t.add("vkBeginCommandBuffer", cb="command_buffer#1", flags=[])
    t.add("vkCmdPushConstants", cb="command_buffer#1", layout="pl#1", stages=call_stages, offset=0, size=64,
          data="00" * 64)
    t.add("vkEndCommandBuffer", cb="command_buffer#1")
    t.add("vkQueueSubmit", queue="queue#1", submits=[{"wait": [], "command_buffers": ["command_buffer#1"],
                                                      "signal": []}], fence=None)
    t.write(path)
    return path


class PushConstantsStageAll(unittest.TestCase):
    def _codes(self, layout_stages, call_stages):
        with tempfile.TemporaryDirectory() as d:
            m = load_trace(_push_trace(os.path.join(d, "t.jsonl"), layout_stages, call_stages))
            return timing.check(m.frame, Report(), window=False)

    def test_stage_all_does_not_abort(self):
        r = self._codes(STAGE_ALL, STAGE_ALL)
        self.assertNotIn("structure.push-constant-range", r.codes())

    def test_real_mismatch_with_stage_all_is_still_reported(self):
        # VUID-vkCmdPushConstants-offset-01796: the call must name every stage of an overlapping range
        r = self._codes(STAGE_ALL, ["VK_SHADER_STAGE_VERTEX_BIT"])
        self.assertIn("structure.push-constant-range", r.codes())

    def test_uncovered_stage_is_still_reported(self):
        # VUID-vkCmdPushConstants-offset-01795: every stage of the call needs a covering range
        r = self._codes(["VK_SHADER_STAGE_VERTEX_BIT"], STAGE_ALL)
        self.assertIn("structure.push-constant-range", r.codes())


# ------------------------------------------------------------------ 5. diagnostics
class DiagnosticsNameTheFailedScopeTest(unittest.TestCase):
    def _msg(self, ops):
        _, r = _pairs(_frame(("graphics0", ops)), False)
        (d,) = r.items
        return d.message

    def test_ray_tracing_write_under_all_commands(self):
        # SW-13 (raytracingbasic): srcStageMask ALL_COMMANDS holds the write; srcAccessMask does not
        m = self._msg([Use("img", "write", stage="RAY_TRACING_SHADER", flag="SHADER_STORAGE_WRITE"),
                       _bar("img", "general", "transfer_src", {"ALL_COMMANDS"}, (), {"ALL_COMMANDS"}, ())])
        self.assertIn("the first synchronization scope ALL_COMMANDS holds RAY_TRACING_SHADER", m)
        self.assertIn("srcAccessMask (NONE) does not include SHADER_STORAGE_WRITE", m)
        self.assertNotIn("not in the first synchronization scope", m)

    def test_execution_scope(self):
        m = self._msg([Use("buf", "write", stage="COPY", flag="TRANSFER_WRITE"),
                       _bar(None, None, None, {"FRAGMENT_SHADER"}, {"SHADER_STORAGE_WRITE"}, {"COMPUTE_SHADER"},
                            {"SHADER_STORAGE_READ"}),
                       Use("buf", "read", stage="COMPUTE_SHADER", flag="SHADER_STORAGE_READ")])
        self.assertIn("COPY is not in the first synchronization scope of the barrier (srcStageMask "
                      "FRAGMENT_SHADER)", m)

    def test_visibility_scope(self):
        m = self._msg([Use("buf", "write", stage="COPY", flag="TRANSFER_WRITE"),
                       _bar(None, None, None, {"COPY"}, {"TRANSFER_WRITE"}, {"COMPUTE_SHADER"}, ()),
                       Use("buf", "read", stage="COMPUTE_SHADER", flag="SHADER_STORAGE_READ")])
        self.assertIn("made available, but not visible", m)
        self.assertIn("dstAccessMask (NONE) does not include SHADER_STORAGE_READ", m)

    def test_broken_chain(self):
        # whole-frame difftest W1-F3-0018 (b_sbuf_war src_none): the first barrier holds the write,
        # the last names the later stage in dstStageMask but chains from nothing. Before this test
        # the text read "COMPUTE_SHADER is not in ... (dstStageMask COMPUTE_SHADER of the last barrier)"
        m = self._msg([Use("buf", "write", stage="COMPUTE_SHADER", flag="SHADER_STORAGE_WRITE"),
                       _bar(None, None, None, {"COMPUTE_SHADER"}, {"SHADER_STORAGE_WRITE"}, {"VERTEX_INPUT"},
                            {"VERTEX_ATTRIBUTE_READ"}),
                       _bar(None, None, None, (), (), {"COMPUTE_SHADER"}, {"SHADER_STORAGE_WRITE"}),
                       Use("buf", "write", stage="COMPUTE_SHADER", flag="SHADER_STORAGE_WRITE")])
        self.assertIn("the dependency chain is broken there", m)
        self.assertIn("srcStageMask NONE", m)
        self.assertNotIn("is not in the second synchronization scope", m)


# ------------------------------------------------------------------ elision premise
ELIDE_TRACES = ROOT / "bench" / "elide" / "traces"


@unittest.skipUnless((ELIDE_TRACES / "particles.jsonl").exists(), "demonstration traces not present")
class ElisionComparesEventPairs(unittest.TestCase):
    """The elision premise Haz_pw(F*(tau_P)) <= Haz_pw(F*(tau)) is checked by
    event identity (stable across plans), for the strict plan and for the chain plan."""

    @classmethod
    def setUpClass(cls):
        from covenant import elide
        cls.el = elide.Elider(str(ELIDE_TRACES / "particles.jsonl"), str(ELIDE_TRACES / "particles-spv"))
        cls.el.run()

    def test_strict_and_chain_plans_verified_pairwise(self):
        self.assertEqual(self.el.pairwise["new_pairs"], 0)
        self.assertEqual(self.el.pairwise["new"], 0)
        self.assertGreater(self.el.pairwise["baseline_pairs"] + 1, 0)
        self.assertIn("pairwise, by event identity", self.el.chain_note)

    def test_identities_are_stable_across_builds(self):
        self.el.hazards({}, exhaustive=True)
        a = self.el.last_pairs
        self.el.hazards({}, exhaustive=True)
        self.assertEqual(a, self.el.last_pairs)

    def test_removing_every_barrier_adds_pairs(self):
        plan = {c.key: {e.label: "remove" for e in c.elems} for c in self.el.calls}
        pw = self.el._pairwise_verify(plan)
        self.assertGreater(pw["new_pairs"], 0)


# ------------------------------------------------------------------ 6/7. driver-based
def _tools():
    cxx = shutil.which("clang++-20") or shutil.which("g++")
    return cxx and shutil.which("glslangValidator") and shutil.which("make")


def _make(*targets):
    r = subprocess.run(["make", "-s", "-C", str(MD), *targets], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("model driver build failed:\n" + r.stdout + r.stderr)


def _run(exe, args, env):
    try:
        r = subprocess.run([str(exe)] + args, capture_output=True, text=True, env=env, timeout=60)
    except PermissionError:          # the shared project folder does not keep the exec bit
        r = subprocess.run([LDSO, str(exe)] + args, capture_output=True, text=True, env=env, timeout=60)
    if r.returncode:
        raise RuntimeError(f"{exe.name} failed ({r.returncode}):\n{r.stdout}{r.stderr}")


@unittest.skipUnless(_tools(), "needs a C++ compiler, make and glslangValidator")
class StagesFromEntryPoints(unittest.TestCase):
    """The triangle's uniform buffer is read by the vertex shader only. With the set
    layout widened to VERTEX|FRAGMENT, the read is charged at the stage whose entry
    point uses it, not at both stages (the layout charge is kept when the SPIR-V is
    unknown)."""

    @classmethod
    def setUpClass(cls):
        _make("all", "example")
        cls.tmp = Path(tempfile.mkdtemp(prefix="cov-stages-"))
        trace, spv = cls.tmp / "t.jsonl", cls.tmp / "spv"
        spv.mkdir()
        _run(BUILD / "triangle", [str(BUILD / "tri.vert.spv"), str(BUILD / "tri.frag.spv")],
             dict(os.environ, COVENANT_TRACE=str(trace), COVENANT_SPIRV_DIR=str(spv)))
        recs = [json.loads(x) for x in trace.read_text().splitlines() if x.strip()]
        for r in recs:
            if r["fn"] == "vkCreateDescriptorSetLayout":
                for b in r["bindings"]:
                    b["stages"] = ["VK_SHADER_STAGE_VERTEX_BIT", "VK_SHADER_STAGE_FRAGMENT_BIT"]
        cls.wide = cls.tmp / "wide.jsonl"
        cls.wide.write_text("".join(json.dumps(r) + "\n" for r in recs))
        cls.spv = spv

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _ubo_stages(self, spirv_dir):
        m = load_trace(str(self.wide), spirv_dir=spirv_dir)
        return {op.stage for op in m.frame.queues["graphics0"]
                if isinstance(op, Use) and str(op.res).startswith("ubo")}

    def test_entry_point_stage(self):
        self.assertEqual(self._ubo_stages(str(self.spv)), {"VERTEX_SHADER"})

    def test_layout_fallback_without_spirv(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertEqual(self._ubo_stages(empty), {"VERTEX_SHADER", "FRAGMENT_SHADER"})


def _capture_trace(path, rec23):
    """The hostcapture probe's record sequence (model_driver/examples/hostcapture.cpp):
    buffer#1 = bytes 0..255 and buffer#2 = 256..511 of one host-coherent page; a copy
    reads buffer#1; rec23 is the host_write record between the submit and the fence wait."""
    t = _Trace()
    t.add("covenant_trace_begin", version=1, host_write_tracking=True, page_size=4096)
    t.add("vkGetDeviceQueue", queue="queue#1", family=0, index=0)
    for b, u in ((1, "SRC"), (2, "SRC"), (3, "DST")):
        t.add("vkCreateBuffer", buffer=f"buffer#{b}", size=256, usage=[f"VK_BUFFER_USAGE_TRANSFER_{u}_BIT"],
              sharing="VK_SHARING_MODE_EXCLUSIVE")
    t.add("vkAllocateMemory", memory="memory#1", size=4096,
          properties=["VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT", "VK_MEMORY_PROPERTY_HOST_COHERENT_BIT"])
    t.add("vkAllocateMemory", memory="memory#2", size=4096, properties=["VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT"])
    t.add("vkBindBufferMemory", buffer="buffer#1", memory="memory#1", offset=0, size=256)
    t.add("vkBindBufferMemory", buffer="buffer#2", memory="memory#1", offset=256, size=256)
    t.add("vkBindBufferMemory", buffer="buffer#3", memory="memory#2", offset=0, size=256)
    t.add("vkMapMemory", memory="memory#1", offset=0, size="whole", tracked=True)
    t.add("host_write", memory="memory#1", offset=0, page_size=4096, offsets=[0], fault_tids=[7],
          call_tids=[7, 7], threads=1, writes=[[0, "11" * 256], [256, "22" * 256]])
    t.add("vkAllocateCommandBuffers", pool="cp#1", level="VK_COMMAND_BUFFER_LEVEL_PRIMARY",
          command_buffers=["command_buffer#1"])
    t.add("vkBeginCommandBuffer", cb="command_buffer#1", flags=[])
    t.add("vkCmdCopyBuffer", cb="command_buffer#1", src="buffer#1", dst="buffer#3",
          regions=[{"src_offset": 0, "dst_offset": 0, "size": 256}])
    t.add("vkEndCommandBuffer", cb="command_buffer#1")
    t.add("vkCreateFence", fence="fence#1", signaled=False)
    t.add("vkQueueSubmit", queue="queue#1", submits=[{"wait": [], "command_buffers": ["command_buffer#1"],
                                                      "signal": []}], fence="fence#1")
    if rec23 is not None:
        t.add("host_write", memory="memory#1", offset=0, page_size=4096, offsets=[0], **rec23)
    t.add("vkWaitForFences", fences=["fence#1"], wait_all=True, timeout="infinite")
    t.add("host_write", memory="memory#1", offset=0, page_size=4096, offsets=[0], fault_tids=[7],
          call_tids=[7, 7], threads=1, writes=[[32, "77"]])
    t.add("vkDeviceWaitIdle", device="device#1")
    t.write(path)
    return path


ONE = dict(fault_tids=[7], call_tids=[7, 7], threads=1)
CAPTURE_CASES = {           # name: (record between submit and wait, race in conservative, race in observed)
    "ok": (None, False, False),
    "changed": (dict(ONE, writes=[[16, "33"]]), True, True),
    "same": (dict(ONE, writes=[]), True, False),             # same-value store: no byte changed
    "restore": (dict(ONE, writes=[]), True, False),          # written and restored: no byte changed
    "neighbour": (dict(ONE, writes=[[272, "55"]]), True, False),   # buffer#2 only: page-granularity cost
}


def _capture_check(path, mode, spirv_dir=None):
    with mock.patch.dict(os.environ, {"COVENANT_HOST_CAPTURE": mode}):
        m = load_trace(str(path), spirv_dir=spirv_dir)
    r = timing.check(m.frame, Report(), window=False)
    races = [d for d in r.items if d.code == "timing.cross-queue-race" and "buffer#1" in d.message]
    return bool(races), r, m


class HostCaptureModes(unittest.TestCase):
    def test_default_is_conservative(self):
        self.assertEqual(modes.host_capture(), "conservative")
        with mock.patch.dict(os.environ, {"COVENANT_HOST_CAPTURE": "bogus"}):
            self.assertRaises(ValueError, modes.host_capture)

    def test_synthetic_records(self):
        with tempfile.TemporaryDirectory() as d:
            for name, (rec, cons, obs) in CAPTURE_CASES.items():
                p = _capture_trace(os.path.join(d, f"{name}.jsonl"), rec)
                for mode, want in (("conservative", cons), ("observed", obs)):
                    got, r, m = _capture_check(p, mode)
                    self.assertEqual(got, want, f"{name} / {mode}: {r}")
                    self.assertEqual(m.capture["mode"], mode)
                    self.assertEqual(m.capture["same_value_records"], 1 if name in ("same", "restore") else 0)

    def test_page_only_record_is_charged_in_both_modes(self):
        # a driver without shadow copies (no "writes"): whole pages in both modes
        with tempfile.TemporaryDirectory() as d:
            p = _capture_trace(os.path.join(d, "v1.jsonl"), dict(ONE))
            for mode in ("conservative", "observed"):
                self.assertTrue(_capture_check(p, mode)[0], mode)

    def test_write_from_another_thread_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            p = _capture_trace(os.path.join(d, "thr.jsonl"),
                               dict(fault_tids=[8], call_tids=[7, 7], threads=2, writes=[[16, "66"]]))
            got, r, m = _capture_check(p, "conservative")
            self.assertTrue(got)
            self.assertEqual(m.capture["cross_thread_records"], 1)
            self.assertTrue(any(d.code.startswith("timing.unmodelled") and "thread" in d.message
                                for d in r.items), str(r))

    def test_untracked_mapping_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(_capture_trace(os.path.join(d, "u.jsonl"), None))
            recs = [json.loads(x) for x in p.read_text().splitlines()]
            for r in recs:
                if r["fn"] == "vkMapMemory":
                    r["tracked"] = False
            p.write_text("".join(json.dumps(r) + "\n" for r in recs))
            _, r, _ = _capture_check(p, "conservative")
            self.assertTrue(any("without host-write capture" in d.message for d in r.items), str(r))


@unittest.skipUnless(_tools(), "needs a C++ compiler and make")
class HostCaptureProbe(unittest.TestCase):
    """The same cases end to end: the probe program on the model driver."""

    @classmethod
    def setUpClass(cls):
        _make("all", "hostcapture")
        cls.tmp = Path(tempfile.mkdtemp(prefix="cov-hostcapture-"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _trace(self, mode):
        p = self.tmp / f"{mode}.jsonl"
        if not p.exists():
            _run(BUILD / "hostcapture", [mode], dict(os.environ, COVENANT_TRACE=str(p)))
        return p

    def test_modes(self):
        for name, (_, cons, obs) in CAPTURE_CASES.items():
            p = self._trace(name)
            for mode, want in (("conservative", cons), ("observed", obs)):
                self.assertEqual(_capture_check(p, mode)[0], want, f"{name} / {mode}")

    def test_same_value_and_restore_leave_no_diff(self):
        for name in ("same", "restore"):
            recs = [json.loads(x) for x in self._trace(name).read_text().splitlines()]
            hw = [r for r in recs if r["fn"] == "host_write"]
            self.assertEqual(len(hw), 3)
            self.assertEqual(hw[1]["writes"], [])        # the trap saw the page written
            self.assertEqual(hw[1]["offsets"], [0])

    def test_thread(self):
        p = self._trace("thread")
        recs = [json.loads(x) for x in p.read_text().splitlines()]
        hw = [r for r in recs if r["fn"] == "host_write"]
        self.assertNotEqual(hw[1]["fault_tids"], [hw[1]["call_tids"][0]])
        got, r, m = _capture_check(p, "conservative")
        self.assertTrue(got)
        self.assertEqual(m.capture["cross_thread_records"], 1)


# minimal frames of the paper's appendix (run as standalone scripts)
REPRO = Path(os.environ.get("COVENANT_FORMAL_REPRO", Path(__file__).resolve().parent / "fixtures" / "minimal_frames"))


def _repro(name):
    out = subprocess.run([sys.executable, "-I", str(REPRO / name), str(ROOT)],
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.splitlines()


def _line(lines, prefix):
    hit = [l for l in lines if l.startswith(prefix)]
    assert len(hit) == 1, (prefix, lines)
    return hit[0][len(prefix):].strip()


@unittest.skipUnless(REPRO.is_dir(), "tests/fixtures/minimal_frames not present")
class FormalRepros(unittest.TestCase):
    """The minimal frames of the paper's appendix, run as standalone scripts
    (python3 -I <script> <covenant root>)."""

    def test_masked_signal_relay(self):
        lines = _repro("masked_signal_relay.py")
        for case in ("relay-queue", "direct", "host-relay"):
            l = _line(lines, case + " pairwise:")
            self.assertIn("timing.signal-stage-scope", l, case)
            self.assertIn("epoch: 1", l, case)
            self.assertNotIn("ref: set()", l, case)
        self.assertEqual(_line(lines, "undefined relaxed=True"), "[] set()")
        self.assertIn("timing.barrier-scope", _line(lines, "undefined relaxed=False"))

    def test_host_signal_wait(self):
        lines = _repro("host_signal_wait.py")
        for k in ("pairwise", "epoch"):
            self.assertIn("timing.wait-stage-scope", _line(lines, k), k)
        self.assertNotEqual(_line(lines, "ref"), "set()")

    def test_atomic_stand_in(self):
        lines = _repro("atomic_stand_in.py")
        for k in ("exhaustive", "epoch"):
            self.assertIn("(('graphics0', 1), ('graphics0', 4)", _line(lines, k), k)
        self.assertNotEqual(_line(lines, "ref"), "set()")

    def test_region_nonuniform(self):
        # the known region cut: pairwise over-approximates (a false alarm, allowed by
        # Theorem 1), epoch and the reference report nothing
        lines = _repro("region_cut.py")
        self.assertIn("timing.barrier-scope", _line(lines, "pairwise"))
        self.assertEqual(_line(lines, "epoch"), "[]")
        self.assertEqual(_line(lines, "ref"), "set()")


if __name__ == "__main__":
    unittest.main()
