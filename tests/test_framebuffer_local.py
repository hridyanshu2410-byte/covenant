"""Framebuffer-local dependencies and the ordering of layout transitions.

1. Framebuffer-local dependencies. A dependency whose dependencyFlags hold
VK_DEPENDENCY_BY_REGION_BIT or VK_DEPENDENCY_VIEW_LOCAL_BIT orders a pair of
operations that are both in framebuffer-space stages only at the same
framebuffer position (synchronization.adoc, "Framebuffer Region
Dependencies"). The model orders such a pair only when both accesses carry the
same framebuffer token (Use.fb: attachment accesses through the same image /
view base); never a storage, sampled or uniform access. Pairs with a
non-framebuffer-space stage on either side are ordinary dependencies. The
model driver records the dependencyFlags of pipeline barriers and of subpass
dependencies, and the translation reads them.

2. Rule R-LT (layout transitions on one queue are ordered among themselves).
Two layout transitions on one queue are exempt only when at least one of them
is performed by an image memory barrier; two automatic render-pass transitions
are checked like any other pair.

Every frame is also checked against the reference semantics
(covenant/timing_ref.py): the pairwise checker reports exactly its races.
"""
import hashlib
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from covenant import host_trace, timing, timing_ref  # noqa: E402
from covenant.diagnostics import Report  # noqa: E402
from covenant.host_trace import load_trace  # noqa: E402
from covenant.timing import Barrier, Frame, Signal, Use, Wait  # noqa: E402
from test_timing_fixes import RP_ATT, Trace  # noqa: E402

F = frozenset
FS, CO, VS, XF = "FRAGMENT_SHADER", "COLOR_ATTACHMENT_OUTPUT", "VERTEX_SHADER", "TRANSFER"
SW, SR = "SHADER_STORAGE_WRITE", "SHADER_STORAGE_READ"
CW, IR = "COLOR_ATTACHMENT_WRITE", "INPUT_ATTACHMENT_READ"
LITMUS = ROOT / "tests" / "litmus"
MD = ROOT / "model_driver"
LDSO = "/lib64/ld-linux-x86-64.so.2"


def pairs(frame):
    hz = []
    timing.check(frame, Report(), hazards=hz, exhaustive=True)
    return {frozenset((a, b)) for a, b, _ in hz}


def spec(frame):
    return timing_ref.race_keys(frame)


def bar(src, sa, dst, da, res=None, old=None, new=None, local=False, fb=None, auto=False, where="bar"):
    return Barrier(res, old, new, where, F(src), F(sa), F(dst), F(da), fb_local=local, fb=fb, auto=auto)


class FramebufferLocalFrames(unittest.TestCase):
    """Hand-built frames: the semantics of Barrier.fb_local and Use.fb."""

    def check(self, ops, racing, queues=None):
        fr = Frame(queues=queues or {"gfx": ops})
        self.assertEqual(pairs(fr), spec(fr))       # checker == reference
        self.assertEqual(bool(spec(fr)), racing)
        return fr

    def test_storage_buffer_by_region_races(self):
        # subpass 0 fragment writes storageBuffer[0]; BY_REGION
        # FRAGMENT/SHADER_WRITE -> FRAGMENT/SHADER_READ; subpass 1 fragment reads it
        ops = [Use("buf", "write", None, "w", FS, SW, region=((0, 4),)),
               bar({FS}, {"SHADER_WRITE"}, {FS}, {"SHADER_READ"}, local=True),
               Use("buf", "read", None, "r", FS, SR, region=((0, 4),))]
        fr = self.check(ops, True)
        msgs = [d.message for d in timing.check(fr, Report()).items]
        self.assertTrue(any("framebuffer-local" in m for m in msgs), msgs)
        ops[1].fb_local = False
        self.check(ops, False)                       # the same without BY_REGION: ordered

    def test_same_position_attachment_pair_is_ordered(self):
        tok = ("fb", "imgA", 0, 0)
        ops = [Use("imgA", "write", None, "w", CO, CW, fb=tok),
               bar({CO}, {CW}, {FS}, {IR}, local=True),
               Use("imgA", "read", None, "r", FS, IR, fb=tok)]
        self.check(ops, False)
        ops[2].fb = ("fb", "imgA", 0, 1)             # another framebuffer mapping
        self.check(ops, True)
        ops[2].fb = None                             # e.g. a sampled read of the same image
        self.check(ops, True)

    def test_non_framebuffer_stage_pairs_are_ordinary(self):
        # "A dependency with non-framebuffer-space pipeline stages is neither
        # framebuffer-global nor framebuffer-local": BY_REGION does not restrict it
        self.check([Use("buf", "write", None, "w", VS, SW),
                    bar({VS}, {"SHADER_WRITE"}, {FS}, {"SHADER_READ"}, local=True),
                    Use("buf", "read", None, "r", FS, SR)], False)
        self.check([Use("buf", "write", None, "w", FS, SW),
                    bar({FS}, {"SHADER_WRITE"}, {VS}, {"SHADER_READ"}, local=True),
                    Use("buf", "read", None, "r", VS, SR)], False)
        self.check([Use("buf", "write", None, "w", FS, SW),
                    bar({FS}, {"SHADER_WRITE"}, {XF}, {"TRANSFER_READ"}, local=True),
                    Use("buf", "read", None, "r", XF, "TRANSFER_READ")], False)
        # an expanded meta stage is split per pair: FRAGMENT -> FRAGMENT stays local
        self.check([Use("buf", "write", None, "w", FS, SW),
                    bar({"ALL_COMMANDS"}, {"MEMORY_WRITE"}, {"ALL_COMMANDS"}, {"MEMORY_READ"}, local=True),
                    Use("buf", "read", None, "r", FS, SR)], True)

    def test_local_then_global_link_is_global(self):
        ops = [Use("buf", "write", None, "w", FS, SW),
               bar({FS}, {"SHADER_WRITE"}, {FS}, set(), local=True, where="b1"),
               bar({FS}, set(), {FS}, {"SHADER_READ"}, where="b2"),
               Use("buf", "read", None, "r", FS, SR)]
        self.check(ops, False)
        ops[2].fb_local = True                       # local -> local: still only local
        self.check(ops, True)
        ops[1].fb_local = False                      # global -> local: global
        self.check(ops, False)

    def test_view_offset_orders_no_position(self):
        tok = ("fb", "imgA", 0, 0)
        ops = [Use("imgA", "write", None, "w", CO, CW, fb=tok),
               bar({CO}, {CW}, {FS}, {IR}, local="none"),
               Use("imgA", "read", None, "r", FS, IR, fb=tok)]
        self.check(ops, True)

    def test_transition_of_a_local_dependency(self):
        tok = ("fb", "imgA", 0, 0)
        ops = [Use("imgA", "write", "color_attachment", "w", CO, CW, fb=tok),
               bar({CO}, {CW}, {FS}, {IR}, res="imgA", old="color_attachment", new="shader_read",
                   local=True, fb=tok, auto=True),
               Use("imgA", "read", "shader_read", "r", FS, IR, fb=tok)]
        self.check(ops, False)
        ops[2].fb = None
        self.check(ops, True)
        ops[2].fb, ops[0].fb = tok, None
        self.check(ops, True)

    def test_cross_queue_waits_are_global(self):
        q0 = [Use("buf", "write", None, "w", FS, SW),
              bar({FS}, {"SHADER_WRITE"}, {FS}, set(), local=True),
              Signal("s", "sig", F({FS}))]
        q1 = [Wait("s", "wait", F({FS})), Use("buf", "read", None, "r", FS, SR)]
        # the write reaches the signal's first scope through a local link, then the
        # semaphore (an ordinary dependency): global
        self.check(None, False, queues={"q0": q0, "q1": q1})


class TransitionPairs(unittest.TestCase):
    """The transition-pair matrix (same queue, no dependency chain
    between the two transitions)."""

    def frame(self, auto1, auto2, queues=False):
        t1 = bar({"TOP_OF_PIPE"}, set(), {"BOTTOM_OF_PIPE"}, set(), res="img", old="color_attachment",
                 new="shader_read", auto=auto1, where="t1")
        t2 = bar({"TOP_OF_PIPE"}, set(), {"BOTTOM_OF_PIPE"}, set(), res="img", old="shader_read",
                 new="color_attachment", auto=auto2, where="t2")
        if queues:
            return Frame(queues={"q0": [t1], "q1": [t2]}, initial_layouts={"img": "color_attachment"})
        return Frame(queues={"q": [t1, t2]}, initial_layouts={"img": "color_attachment"})

    def test_barrier_barrier_exempt(self):
        fr = self.frame(False, False)
        self.assertEqual(pairs(fr), set())
        self.assertEqual(spec(fr), set())

    def test_render_pass_then_barrier_exempt(self):
        for a1, a2 in ((True, False), (False, True)):
            fr = self.frame(a1, a2)
            self.assertEqual(pairs(fr), set())
            self.assertEqual(spec(fr), set())

    def test_render_pass_render_pass_reported(self):
        fr = self.frame(True, True)
        self.assertEqual(pairs(fr), {frozenset({("q", 1), ("q", 2)})})
        self.assertEqual(spec(fr), pairs(fr))
        msgs = [d.message for d in timing.check(fr, Report()).items]
        self.assertTrue(any("automatic layout transition" in m for m in msgs), msgs)

    def test_other_queue_never_exempt(self):
        fr = self.frame(False, False, queues=True)
        self.assertEqual(spec(fr), pairs(fr))
        self.assertTrue(pairs(fr))

    def test_not_transitive(self):
        # R-LT orders only the two transitions: a write before the first one is
        # still checked against the second
        fr = self.frame(False, False)
        fr.queues["q"].insert(0, Use("img", "write", "color_attachment", "w", CO, CW))
        self.assertIn(frozenset({("q", 1), ("q", 3)}), pairs(fr))
        self.assertEqual(spec(fr), pairs(fr))


class Translation(unittest.TestCase):
    """Synthetic model-driver records -> Barrier.fb_local / Use.fb / Barrier.auto."""

    def rp_model(self, flags, view_offset=0, samples=1):
        t = Trace()
        t.add("vkCreateImage", image="image#1", format="VK_FORMAT_R8G8B8A8_UNORM", usage=[], extent=[4, 4, 1],
              mip_levels=1, array_layers=1, samples=samples, sharing="VK_SHARING_MODE_CONCURRENT",
              initial_layout="VK_IMAGE_LAYOUT_UNDEFINED")
        t.add("vkCreateImageView", view="view#1", image="image#1", format="VK_FORMAT_R8G8B8A8_UNORM",
              view_type="VK_IMAGE_VIEW_TYPE_2D", aspect=["VK_IMAGE_ASPECT_COLOR_BIT"], base_mip=0,
              level_count=1, base_layer=0, layer_count=1)
        dep = {"src_subpass": 0, "dst_subpass": 1,
               "src_stage": ["VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT"],
               "dst_stage": ["VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT"],
               "src_access": ["VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT"],
               "dst_access": ["VK_ACCESS_INPUT_ATTACHMENT_READ_BIT"]}
        if flags is not None:
            dep["dependency_flags"] = flags
            dep["view_offset"] = view_offset
        t.add("vkCreateRenderPass", render_pass="rp#1", attachments=[dict(RP_ATT, samples=samples)],
              dependencies=[dep],
              subpasses=[{"color": [{"attachment": 0, "layout": "VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL"}]},
                         {"color": [], "input": [{"attachment": 0,
                                                  "layout": "VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL"}]}])
        t.add("vkCreateFramebuffer", framebuffer="fb#1", attachments=["view#1"])
        t.submit([("vkCmdBeginRenderPass", dict(render_pass="rp#1", framebuffer="fb#1", clear_values=[])),
                  ("vkCmdNextSubpass", {}), ("vkCmdEndRenderPass", {})])
        return t.model()

    def ops(self, m):
        return m.frame.queues["graphics0"]

    def test_subpass_dependency_flags(self):
        for flags, want in (([], False), (["VK_DEPENDENCY_BY_REGION_BIT"], True),
                            (["VK_DEPENDENCY_VIEW_LOCAL_BIT"], True)):
            m = self.rp_model(flags)
            deps = [op for op in self.ops(m) if isinstance(op, Barrier) and "subpass dependency into" in op.where]
            self.assertEqual([op.fb_local for op in deps], [want])
            lt = [op for op in self.ops(m) if isinstance(op, Barrier) and op.new and "vkCmdNextSubpass" in op.where]
            self.assertTrue(lt and all(op.auto and op.fb_local == want and op.fb for op in lt))
            self.assertNotIn("subpass dependency flags not recorded", m.unsupported)
        m = self.rp_model(["VK_DEPENDENCY_VIEW_LOCAL_BIT"], view_offset=1)
        self.assertIn("none", [op.fb_local for op in self.ops(m) if isinstance(op, Barrier)])

    def test_attachment_tokens(self):
        m = self.rp_model(["VK_DEPENDENCY_BY_REGION_BIT"])
        uses = [op for op in self.ops(m) if isinstance(op, Use) and op.res.startswith("image#1")]
        self.assertTrue(uses)
        self.assertEqual({op.fb for op in uses}, {("fb", "image#1", 0, 0)})
        # a multisampled input attachment may be read at any sample: no token
        m = self.rp_model(["VK_DEPENDENCY_BY_REGION_BIT"], samples=4)
        inp = [op for op in self.ops(m) if isinstance(op, Use) and op.flag == IR]
        self.assertTrue(inp and all(op.fb is None for op in inp))

    def test_legacy_trace_without_flags(self):
        # a trace from a model driver that did not record subpass dependency flags:
        # unknown flags are taken as framebuffer-local
        m = self.rp_model(None)
        deps = [op for op in self.ops(m) if isinstance(op, Barrier) and "subpass dependency into" in op.where]
        self.assertEqual([op.fb_local for op in deps], [True])
        self.assertIn("subpass dependency flags not recorded", m.unsupported)
        with mock.patch.object(host_trace, "_LEGACY_GLOBAL", True):
            m = self.rp_model(None)
        deps = [op for op in self.ops(m) if isinstance(op, Barrier) and "subpass dependency into" in op.where]
        self.assertEqual([op.fb_local for op in deps], [False])

    def test_pipeline_barrier_flags(self):
        t = Trace()
        t.buffer("buffer#1")
        mb = {"src_stage": ["VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT"], "dst_stage": ["VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT"],
              "src_access": ["VK_ACCESS_SHADER_WRITE_BIT"], "dst_access": ["VK_ACCESS_SHADER_READ_BIT"]}
        t.submit([("vkCmdPipelineBarrier", dict(dependency_flags=["VK_DEPENDENCY_BY_REGION_BIT"],
                                                src_stage=mb["src_stage"], dst_stage=mb["dst_stage"],
                                                memory_barriers=[mb], buffer_barriers=[], image_barriers=[])),
                  ("vkCmdPipelineBarrier2", dict(dependency_flags=[], memory_barriers=[mb], buffer_barriers=[],
                                                 image_barriers=[]))])
        bars = [op for op in t.model().frame.queues["graphics0"] if isinstance(op, Barrier)]
        self.assertEqual([op.fb_local for op in bars], [True, False])


def _tools():
    return shutil.which("glslangValidator") and (shutil.which("clang++-20") or shutil.which("g++")) \
        and shutil.which("make")


def _run(cmd, env=None):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    except PermissionError:
        return subprocess.run([LDSO] + cmd, capture_output=True, text=True, env=env, timeout=120)


@unittest.skipUnless(_tools(), "needs a C++ compiler, make and glslangValidator")
class Litmus(unittest.TestCase):
    """tests/litmus/sync_litmus.cpp against the model driver: the recorded flags,
    the translation and the checker end to end."""

    @classmethod
    def setUpClass(cls):
        # the model driver needs Vulkan-Headers 1.4.365 (README): COVENANT_VKINC=<headers>/include
        vkinc = os.environ.get("COVENANT_VKINC")
        mk = ["make", "-s", "-C", str(MD), "all"]
        if vkinc:
            mk += [f"VKINC={vkinc}", f"VKH={vkinc}/vulkan/vulkan_core.h"]
        r = subprocess.run(mk, capture_output=True, text=True)
        if r.returncode:
            raise unittest.SkipTest("model driver build failed (set COVENANT_VKINC to a Vulkan-Headers "
                                    "1.4.365 include dir):\n" + (r.stdout + r.stderr)[-600:])
        cls.tmp = Path(tempfile.mkdtemp(prefix="covenant-litmus-"))
        for src in sorted(LITMUS.glob("*.vert")) + sorted(LITMUS.glob("*.frag")):
            subprocess.run(["glslangValidator", "-V", "-o", str(cls.tmp / (src.stem + ".spv")), str(src)],
                           check=True, capture_output=True)
        cxx = shutil.which("clang++-20") or shutil.which("g++")
        build = MD / "build"
        cls.exe = cls.tmp / "sync_litmus"
        subprocess.run([cxx, "-std=c++17", "-O0"] + ([f"-I{vkinc}"] if vkinc else []) +
                       ["-o", str(cls.exe), str(LITMUS / "sync_litmus.cpp"), f"-L{build}", "-lcovvk", f"-Wl,-rpath,{build}"], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_case(self, case):
        trace, spv = self.tmp / f"{case}.jsonl", self.tmp / f"spv-{case}"
        spv.mkdir(exist_ok=True)
        env = dict(os.environ, COVENANT_TRACE=str(trace), COVENANT_SPIRV_DIR=str(spv))
        r = _run([str(self.exe), str(self.tmp), case], env)
        self.assertEqual(r.returncode, 0, r.stderr)
        m = load_trace(str(trace), spirv_dir=str(spv))
        fr = m.frame
        self.assertEqual(pairs(fr), spec(fr))       # checker == reference on the real translation
        items = timing.check(fr, Report()).items
        recs = [json.loads(x) for x in trace.read_text().splitlines() if x.strip()]
        return m, items, recs

    def local(self, items):
        return [d for d in items if "framebuffer-local" in d.message]

    def test_storage_by_region(self):
        for case in ("storage_byregion", "storage_byregion_rp2"):
            m, items, recs = self.run_case(case)
            rp = [r for r in recs if r["fn"] == "vkCreateRenderPass"][0]
            self.assertIn("VK_DEPENDENCY_BY_REGION_BIT", rp["dependencies"][0]["dependency_flags"])
            self.assertEqual(len(items), 1, [d.message for d in items])
            self.assertEqual(len(self.local(items)), 1)
            self.assertIn("buffer#1", items[0].message)

    def test_controls_are_clean(self):
        for case in ("storage_global", "input_byregion", "vs_write_byregion", "vs_read_byregion",
                     "mv_input_viewlocal", "barrier_global", "rp_rp_ordered", "mb2_global"):
            with self.subTest(case=case):
                m, items, recs = self.run_case(case)
                self.assertEqual([d.message for d in items], [])

    def test_sync2_memory_barrier_on_subpass_dependency(self):
        m, items, recs = self.run_case("mb2_global")
        dep = [r for r in recs if r["fn"] == "vkCreateRenderPass"][0]["dependencies"][0]
        self.assertEqual(dep["src_stage"], ["VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT"])

    def test_multiview(self):
        m, items, recs = self.run_case("mv_storage_viewlocal")
        rp = [r for r in recs if r["fn"] == "vkCreateRenderPass"][0]
        self.assertEqual(rp["view_masks"], [3, 3])
        self.assertIn("VK_DEPENDENCY_VIEW_LOCAL_BIT", rp["dependencies"][0]["dependency_flags"])
        self.assertEqual(len(self.local(items)), 1)
        m, items, recs = self.run_case("mv_input_viewoffset")
        rp = [r for r in recs if r["fn"] == "vkCreateRenderPass"][0]
        self.assertEqual(rp["dependencies"][0]["view_offset"], 1)
        self.assertTrue(any("input attachment" in d.where for d in self.local(items)))

    def test_pipeline_barrier_between_render_passes(self):
        m, items, recs = self.run_case("barrier_byregion")
        self.assertEqual(len(items), 1)
        self.assertEqual(len(self.local(items)), 1)

    def test_render_pass_transition_pairs(self):
        # the shape seen in GTK: two passes, no dependencies; the pair of automatic
        # transitions is reported (it is not exempt), next to the access diagnostics
        m, items, recs = self.run_case("rp_rp_transitions")
        auto = [d for d in items if "automatic layout transition" in d.message]
        self.assertEqual(len(auto), 2)
        hz = []
        timing.check(m.frame, Report(), hazards=hz, exhaustive=True)
        q = m.frame.queues["graphics0"]
        lts = [i + 1 for i, op in enumerate(q) if isinstance(op, Barrier) and op.new]
        # the image barrier's transition (first) is never paired with a render-pass one
        self.assertFalse([p for p in hz if lts[0] in (p[0][1], p[1][1]) and
                          isinstance(q[p[0][1] - 1], Barrier) and isinstance(q[p[1][1] - 1], Barrier)])
        m, items, recs = self.run_case("rp_barrier_transitions")
        q = m.frame.queues["graphics0"]
        hz = []
        timing.check(m.frame, Report(), hazards=hz, exhaustive=True)
        lt_pairs = {frozenset((a[1], b[1])) for a, b, _ in hz
                    if q[a[1] - 1].__class__ is Barrier and q[b[1] - 1].__class__ is Barrier}
        kinds = {i + 1: op.auto for i, op in enumerate(q) if isinstance(op, Barrier) and op.new}
        for p in lt_pairs:
            self.assertTrue(all(kinds[x] for x in p), "a pair with a barrier transition is exempt")
        self.assertTrue(lt_pairs)      # first pass final vs second pass final: both automatic


if __name__ == "__main__":
    unittest.main()
