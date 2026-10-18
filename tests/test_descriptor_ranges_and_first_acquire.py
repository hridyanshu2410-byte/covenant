"""Descriptor byte ranges derived from push constants and dispatch sizes,
truncated host-write records, depth accesses with the depth test off,
report-group merging by command buffer, and the first-acquire rule (a
swapchain image's first use before its first acquire).

Traces are written as JSON lines in the model driver's format; shaders are
compiled with glslangValidator (skipped without it)."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from covenant import ranges, spirv, timing  # noqa: E402
from covenant.diagnostics import Report  # noqa: E402
from covenant.host_trace import load_trace  # noqa: E402
from covenant import report_groups  # noqa: E402

COMP = """#version 450
layout(local_size_x = 64) in;
layout(push_constant) uniform PC { uint base; uint n; } pc;
layout(std430, set = 0, binding = 0) readonly buffer B { uint data[]; };
layout(std430, set = 0, binding = 1) buffer O { uint outv[]; };
layout(std430, set = 0, binding = 2) buffer L { uint loopv[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    outv[pc.base + i] = data[pc.base + i];
    for (uint k = 0; k < pc.n; k++) loopv[k] = i;
}
"""


def _compile(src, stage, out_dir):
    p = Path(out_dir) / f"s.{stage}"
    p.write_text(src)
    o = Path(out_dir) / f"s_{stage}.spv"
    subprocess.run(["glslangValidator", "-V", "-o", str(o), str(p)], check=True, capture_output=True)
    return o


class _Trace:
    def __init__(self):
        self.recs = []

    def add(self, fn, **kw):
        self.recs.append(dict(seq=len(self.recs) + 1, fn=fn, **kw))
        return self.recs[-1]["seq"]

    def write(self, path):
        with open(path, "w") as f:
            for r in self.recs:
                f.write(json.dumps(r) + "\n")


@unittest.skipUnless(shutil.which("glslangValidator"), "needs glslangValidator")
class Ranges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="covv2-"))
        cls.spv = _compile(COMP, "comp", cls.tmp)
        cls.mod = spirv.parse(cls.spv)
        cls.sha = hashlib.sha1(cls.spv.read_bytes()).hexdigest()
        (cls.tmp / "spv").mkdir()
        shutil.copy(cls.spv, cls.tmp / "spv" / f"{cls.sha}.spv")

    def ctx(self, base, groups):
        pc = bytearray(256)
        mask = bytearray(256)
        pc[0:4] = base.to_bytes(4, "little")
        mask[0:8] = b"\x01" * 8
        return ranges.Ctx(pc, mask, {("GlobalInvocationId", 0): (0, groups * 64 - 1)})

    def test_push_constant_and_invocation_range(self):
        ex = ranges.binding_exprs(self.mod, "main")
        self.assertEqual(ranges.local_size(self.mod, "main"), (64, 1, 1))
        for b in ((0, 0), (0, 1)):
            r = ranges.ranges(ex[b], self.ctx(1024, 4))
            self.assertEqual(r, [(4096, 4096 + 256 * 4)])

    def test_loop_index_is_unbounded(self):
        ex = ranges.binding_exprs(self.mod, "main")
        r = ranges.ranges(ex[(0, 2)], self.ctx(0, 1))
        self.assertEqual(r[0][0], 0)
        self.assertGreaterEqual(r[-1][1], 1 << 32)      # only the index type bounds it

    def test_unset_push_constant_is_unknown(self):
        ex = ranges.binding_exprs(self.mod, "main")
        r = ranges.ranges(ex[(0, 0)], ranges.Ctx(bytearray(256), bytearray(256), {}))
        self.assertGreaterEqual(r[-1][1], 1 << 32)

    # -- end to end: a host write to the other half of a ring buffer --------------
    def _ring_trace(self, host_off, base, truncated=False):
        t = _Trace()
        t.add("covenant_trace_begin", version=1, host_write_tracking=True, page_size=4096)
        t.add("vkGetDeviceQueue", queue="queue#1", family=0, index=0)
        t.add("vkCreateBuffer", buffer="buffer#1", size=8192, usage=["VK_BUFFER_USAGE_STORAGE_BUFFER_BIT"])
        t.add("vkAllocateMemory", memory="memory#1", size=8192,
              properties=["VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT", "VK_MEMORY_PROPERTY_HOST_COHERENT_BIT"])
        t.add("vkBindBufferMemory", buffer="buffer#1", memory="memory#1", offset=0, size=8192)
        t.add("vkMapMemory", memory="memory#1", offset=0, size=8192, tracked=True)
        t.add("vkCreateDescriptorSetLayout", layout="dsl#1", create_flags=0, bindings=[
            {"binding": b, "type": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER", "count": 1,
             "stages": ["VK_SHADER_STAGE_COMPUTE_BIT"], "immutable_samplers": False} for b in (0, 1, 2)])
        t.add("vkAllocateDescriptorSets", pool="pool#1", sets=["ds#1"], set_layouts=["dsl#1"])
        t.add("vkUpdateDescriptorSets", copies=[], writes=[
            {"set": "ds#1", "binding": b, "array_element": 0, "type": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
             "elements": [{"buffer": "buffer#1" if b == 0 else "buffer#2", "offset": 0, "range": "whole"}]}
            for b in (0, 1, 2)])
        t.add("vkCreateBuffer", buffer="buffer#2", size=8192, usage=["VK_BUFFER_USAGE_STORAGE_BUFFER_BIT"])
        t.add("vkCreateShaderModule", module="sm#1", sha1=self.sha, code_size=1)
        t.add("vkCreatePipelineLayout", layout="pl#1", set_layouts=["dsl#1"],
              push_constants=[{"stages": ["VK_SHADER_STAGE_COMPUTE_BIT"], "offset": 0, "size": 8}])
        t.add("vkCreateComputePipelines", pipelines=[{"pipeline": "pipeline#1", "layout": "pl#1", "stages": [
            {"stage": ["VK_SHADER_STAGE_COMPUTE_BIT"], "module": "sm#1", "sha1": self.sha, "entry": "main"}]}])
        t.add("vkAllocateCommandBuffers", pool="cp#1", level="VK_COMMAND_BUFFER_LEVEL_PRIMARY",
              command_buffers=["command_buffer#1"])
        t.add("vkBeginCommandBuffer", cb="command_buffer#1", flags=[])
        t.add("vkCmdBindPipeline", cb="command_buffer#1", bind_point="VK_PIPELINE_BIND_POINT_COMPUTE",
              pipeline="pipeline#1")
        t.add("vkCmdBindDescriptorSets", cb="command_buffer#1", bind_point="VK_PIPELINE_BIND_POINT_COMPUTE",
              layout="pl#1", first_set=0, sets=["ds#1"], dynamic_offsets=[])
        t.add("vkCmdPushConstants", cb="command_buffer#1", layout="pl#1", stages=["VK_SHADER_STAGE_COMPUTE_BIT"],
              offset=0, size=8, data=(base.to_bytes(4, "little") + (0).to_bytes(4, "little")).hex())
        t.add("vkCmdDispatch", cb="command_buffer#1", groups=[16, 1, 1])
        t.add("vkEndCommandBuffer", cb="command_buffer#1")
        t.add("vkCreateFence", fence="fence#1", signaled=False)
        t.add("vkQueueSubmit", queue="queue#1", submits=[{"wait": [], "command_buffers": ["command_buffer#1"],
                                                          "signal": []}], fence="fence#1")
        rec = dict(memory="memory#1", offset=host_off & ~4095, page_size=4096, offsets=[host_off & ~4095],
                   writes=[[host_off, "00" * 64]])
        if truncated:
            rec.update(writes=[[host_off, "00" * 64]], writes_truncated=True, truncated_at=host_off + 64)
        t.add("host_write", **rec)
        t.add("vkWaitForFences", fences=["fence#1"], wait_all=True, timeout="infinite")
        p = self.tmp / f"ring_{host_off}_{base}_{truncated}.jsonl"
        t.write(p)
        return p

    def _races(self, path):
        m = load_trace(path, spirv_dir=self.tmp / "spv")
        r = timing.check(m.frame, Report(), window=False)
        return [d for d in r.items if d.code == "timing.cross-queue-race" and "buffer#1" in d.message]

    def test_host_write_to_other_half_is_not_a_race(self):
        # the dispatch reads data[1024 .. 2047] (bytes 4096..8191); the host writes bytes 0..63
        self.assertEqual(self._races(self._ring_trace(0, 1024)), [])

    def test_host_write_to_same_half_is_a_race(self):
        self.assertTrue(self._races(self._ring_trace(4096, 1024)))

    def test_ranges_can_be_disabled(self):
        os.environ["COVENANT_NO_RANGES"] = "1"
        try:
            self.assertTrue(self._races(self._ring_trace(0, 1024)))
        finally:
            del os.environ["COVENANT_NO_RANGES"]

    def test_truncated_record_keeps_exact_spans(self):
        # a truncated record keeps its exact spans; it is not widened to the whole page
        # 0x1000 (bytes 4096..8191)
        self.assertEqual(self._races(self._ring_trace(64, 1024, truncated=True)), [])


class ReportGroupMerge(unittest.TestCase):
    def test_unrepaired_groups_differing_by_command_buffer_merge(self):
        fr = timing.Frame()
        for i, (cb, stg) in enumerate((("command_buffer#3", "VERTEX_SHADER"), ("command_buffer#5", "FRAGMENT_SHADER"))):
            fr.q("graphics0").append(timing.Use("buf", "read", where=f"vkCmdDraw @seq {10 + i} ({cb}, on graphics0)",
                                                stage=stg))
        fr.q("host").append(timing.Use("buf", "write", where="host write to buf, before call seq 99"))
        res = report_groups.group(fr, repair=False)
        groups = [g for g in res["other_groups"] if g["code"] == "timing.cross-queue-race"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["occurrences"], 2)


def _gfx_trace(path, depth_test, depth_write, dyn_test=None, first_use_src="VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT",
               present_first=False):
    """Acquire a swapchain image, transition it from UNDEFINED (src stage first_use_src),
    draw into it with a depth attachment, present."""
    t = _Trace()
    t.add("covenant_trace_begin", version=1, host_write_tracking=True, page_size=4096)
    t.add("vkGetDeviceQueue", queue="queue#1", family=0, index=0)
    t.add("vkCreateSwapchainKHR", swapchain="swapchain#1", surface="surface#1", format="VK_FORMAT_B8G8R8A8_UNORM",
          color_space="VK_COLOR_SPACE_SRGB_NONLINEAR_KHR", extent=[64, 64], min_image_count=2, image_count=2,
          surface_min_image_count=2, usage=["VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT"], images=["image#1", "image#2"])
    for i in (1, 2):
        t.add("vkCreateImageView", view=f"image_view#{i}", image=f"image#{i}", format="VK_FORMAT_B8G8R8A8_UNORM",
              view_type="VK_IMAGE_VIEW_TYPE_2D", aspect=["VK_IMAGE_ASPECT_COLOR_BIT"], base_mip=0, level_count=1,
              base_layer=0, layer_count=1)
    t.add("vkCreateImage", image="image#3", type="VK_IMAGE_TYPE_2D", format="VK_FORMAT_D32_SFLOAT", extent=[64, 64, 1],
          mip_levels=1, array_layers=1, samples=1, usage=["VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT"])
    t.add("vkCreateImageView", view="image_view#3", image="image#3", format="VK_FORMAT_D32_SFLOAT",
          view_type="VK_IMAGE_VIEW_TYPE_2D", aspect=["VK_IMAGE_ASPECT_DEPTH_BIT"], base_mip=0, level_count=1,
          base_layer=0, layer_count=1)
    t.add("vkCreatePipelineLayout", layout="pl#1", set_layouts=[], push_constants=[])
    pipe = {"pipeline": "pipeline#1", "layout": "pl#1", "render_pass": None, "subpass": 0, "stages": [],
            "color_formats": ["VK_FORMAT_B8G8R8A8_UNORM"], "depth_format": "VK_FORMAT_D32_SFLOAT",
            "stencil_format": "VK_FORMAT_UNDEFINED", "depth_test": depth_test, "depth_write": depth_write,
            "stencil_test": False, "dynamic_states": ["VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE"] if dyn_test is not None else []}
    t.add("vkCreateGraphicsPipelines", pipelines=[pipe])
    t.add("vkCreateSemaphore", semaphore="semaphore#1")
    t.add("vkCreateSemaphore", semaphore="semaphore#2")
    t.add("vkAllocateCommandBuffers", pool="cp#1", level="VK_COMMAND_BUFFER_LEVEL_PRIMARY",
          command_buffers=["command_buffer#1"])
    for frame in range(2 if present_first else 1):
        img = 1
        t.add("vkAcquireNextImageKHR", swapchain="swapchain#1", semaphore="semaphore#1", fence=None,
              timeout="infinite", image_index=0)
        t.add("vkBeginCommandBuffer", cb="command_buffer#1", flags=[])
        old = "VK_IMAGE_LAYOUT_UNDEFINED"
        t.add("vkCmdPipelineBarrier2", cb="command_buffer#1", dependency_flags=[], memory_barriers=[], buffer_barriers=[],
              image_barriers=[{"image": f"image#{img}", "old_layout": old,
                               "new_layout": "VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL", "src_stage": [first_use_src],
                               "src_access": [], "dst_stage": ["VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT"],
                               "dst_access": ["VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT"], "src_family": None,
                               "dst_family": None, "aspect": ["VK_IMAGE_ASPECT_COLOR_BIT"], "base_mip": 0,
                               "level_count": 1, "base_layer": 0, "layer_count": 1}])
        att = {"layout": "VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL", "load": "VK_ATTACHMENT_LOAD_OP_CLEAR",
               "store": "VK_ATTACHMENT_STORE_OP_STORE", "resolve_view": None}
        t.add("vkCmdBeginRendering", cb="command_buffer#1", area=[0, 0, 64, 64], layer_count=1, view_mask=0, flags=0,
              color=[dict(att, view=f"image_view#{img}")],
              depth=dict(att, view="image_view#3", layout="VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL",
                         load="VK_ATTACHMENT_LOAD_OP_LOAD"), stencil=None)
        t.add("vkCmdBindPipeline", cb="command_buffer#1", bind_point="VK_PIPELINE_BIND_POINT_GRAPHICS",
              pipeline="pipeline#1")
        if dyn_test is not None:
            t.add("vkCmdSetDepthTestEnable", cb="command_buffer#1", enable=dyn_test)
        t.add("vkCmdDraw", cb="command_buffer#1", vertex_count=3, instance_count=1, first_vertex=0, first_instance=0)
        t.add("vkCmdEndRendering", cb="command_buffer#1")
        t.add("vkCmdPipelineBarrier2", cb="command_buffer#1", dependency_flags=[], memory_barriers=[], buffer_barriers=[],
              image_barriers=[{"image": f"image#{img}", "old_layout": "VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL",
                               "new_layout": "VK_IMAGE_LAYOUT_PRESENT_SRC_KHR",
                               "src_stage": ["VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT"],
                               "src_access": ["VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT"],
                               "dst_stage": ["VK_PIPELINE_STAGE_2_NONE"], "dst_access": [], "src_family": None,
                               "dst_family": None, "aspect": ["VK_IMAGE_ASPECT_COLOR_BIT"], "base_mip": 0,
                               "level_count": 1, "base_layer": 0, "layer_count": 1}])
        t.add("vkEndCommandBuffer", cb="command_buffer#1")
        t.add("vkCreateFence", fence=f"fence#{frame + 1}", signaled=False)
        t.add("vkQueueSubmit", queue="queue#1", fence=f"fence#{frame + 1}", submits=[{
            "wait": [{"semaphore": "semaphore#1", "value": None,
                      "stage": ["VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT"]}],
            "command_buffers": ["command_buffer#1"],
            "signal": [{"semaphore": "semaphore#2", "value": None, "stage": ["VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT"]}]}])
        t.add("vkQueuePresentKHR", queue="queue#1", wait=[{"semaphore": "semaphore#2", "value": None}],
              swapchains=[{"swapchain": "swapchain#1", "image_index": 0}])
        t.add("vkWaitForFences", fences=[f"fence#{frame + 1}"], wait_all=True, timeout="infinite")
    t.write(path)
    return path


class FixedFunctionAndAcquire(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="covv2-gfx-"))

    def _depth_uses(self, **kw):
        m = load_trace(_gfx_trace(self.tmp / f"g{len(kw)}{sorted(kw.items())}.jsonl".replace(" ", ""), **kw))
        return [o for o in m.frame.queues["graphics0"] if isinstance(o, timing.Use) and "image#3" in str(o.res)
                and "vkCmdDraw" in o.where]

    def test_depth_test_off_means_no_depth_access(self):
        self.assertEqual(self._depth_uses(depth_test=False, depth_write=True), [])

    def test_depth_test_on_writes(self):
        u = self._depth_uses(depth_test=True, depth_write=True)
        self.assertTrue(u and all(x.access == "write" for x in u))

    def test_dynamic_depth_test_overrides_only_when_declared_dynamic(self):
        self.assertEqual(self._depth_uses(depth_test=True, depth_write=True, dyn_test=False), [])

    def _codes(self, **kw):
        m = load_trace(_gfx_trace(self.tmp / f"a{sorted(kw.items())}.jsonl".replace(" ", ""),
                                  depth_test=True, depth_write=True, **kw))
        return [d.code for d in timing.check(m.frame, Report(), window=False).items]

    def test_first_acquire_transition_from_top_of_pipe(self):
        self.assertIn("timing.acquire-first-use", self._codes())

    def test_first_acquire_ordered_after_wait_is_clean(self):
        self.assertNotIn("timing.acquire-first-use",
                         self._codes(first_use_src="VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT"))

    def test_presented_image_reported_as_before(self):
        # second frame: the image was presented, so the old (error) rule applies
        codes = self._codes(present_first=True)
        self.assertIn("timing.acquire-first-use", codes)
        self.assertIn("timing.wait-stage-scope", codes)


class HostIntervalCap(unittest.TestCase):
    """Host-write regions are capped (without the cap, a 10-frame trace with 400k byte runs
    runs out of memory)."""

    def test_region_coarsen_is_a_superset_with_at_most_k_intervals(self):
        from covenant.host_trace import _region_coarsen
        reg = tuple((i * 10, i * 10 + 5) for i in range(100)) + ((5000, 5001),)
        out = _region_coarsen(reg, 4)
        self.assertLessEqual(len(out), 4)
        for lo, hi in reg:
            self.assertTrue(any(a <= lo and hi <= b for a, b in out))
        self.assertEqual(out[-1], (5000, 5001))       # the far run keeps its own interval
        self.assertEqual(_region_coarsen(reg[:3], 4), reg[:3])

    def test_coarsen_spans_never_closes_a_gap_across_a_binding_boundary(self):
        from covenant.host_trace import _coarsen_spans
        spans = [(i * 10, i * 10 + 5) for i in range(200)]          # [0, 1995)
        out = _coarsen_spans(spans, [0, 997, 997, 4000], k=4)    # 997 lies in the gap (995, 1000)
        self.assertFalse(any(a < 997 < b for a, b in out))
        self.assertIn(995, [b for a, b in out])
        self.assertIn(1000, [a for a, b in out])
        self.assertLessEqual(len(out), 4)
        for lo, hi in spans:
            self.assertTrue(any(a <= lo and hi <= b for a, b in out))


class PreAcquireUse(unittest.TestCase):
    def test_transition_before_the_acquire_names_the_use_rule(self):
        with tempfile.TemporaryDirectory() as d:
            p = _gfx_trace(os.path.join(d, "t.jsonl"), True, True)
            recs = [json.loads(x) for x in open(p)]
            acq = next(r for r in recs if r["fn"] == "vkAcquireNextImageKHR")
            sub = next(r for r in recs if r["fn"] == "vkQueueSubmit")
            sub["submits"][0]["wait"] = []
            recs.remove(acq)
            recs.insert(recs.index(sub) + 1, acq)        # submit (with the transition), then acquire
            for n, r in enumerate(recs):
                r["seq"] = n
            with open(p, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in recs)
            m = load_trace(p)
            ds = [x for x in timing.check(m.frame, Report(), window=False).items
                  if x.code == "timing.acquire-first-use"]
        self.assertTrue(ds)
        self.assertIn("before vkAcquireNextImageKHR returned it", ds[0].message)
        self.assertIn("vkCmdPipelineBarrier2", ds[0].where)
        self.assertTrue(any("Use of a presentable image must occur only after" in n for n in ds[0].notes))

if __name__ == "__main__":
    unittest.main()
