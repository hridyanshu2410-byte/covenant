"""Descriptor-array reads charged only to the elements a draw can dynamically use,
decoded from host-written instance data, and only while that decoding is
schedule-independent (timing.IndexGuard); otherwise every written element is charged
(also with COVENANT_NO_DESC_REFINE=1).

A GTK-like trace: one descriptor set with a COMBINED_IMAGE_SAMPLER array, element 0 a
sampled image in SHADER_READ_ONLY, element 1 an image never transitioned (UNDEFINED).
The draw samples ``tex[id >> 1]`` with ``id`` a flat per-instance attribute that the host
wrote as 0, so it never touches element 1."""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from covenant import desc_index, spirv, timing, timing_ref  # noqa: E402
from covenant.diagnostics import Report  # noqa: E402
from covenant.host_trace import load_trace  # noqa: E402

VERT = """#version 450
layout(location = 0) in uint in_id;
layout(location = 3) flat out uint v_id;
void pass_id(uint x) { v_id = x; }
void main() { pass_id(in_id); gl_Position = vec4(0.0); }
"""
FRAG = """#version 450
#extension GL_EXT_nonuniform_qualifier : require
layout(set = 0, binding = 0) uniform sampler2D tex[4];
layout(location = 3) flat in uint v_id;
layout(location = 0) out vec4 color;
vec4 sample_it(uint id) {
    if ((id & 1u) != 0u) return vec4(1.0);
    id = id >> 1;
    return texture(tex[nonuniformEXT(id)], vec2(0.5));
}
void main() { color = sample_it(v_id); }
"""
# the index also depends on a push constant: not a function of the attribute alone
FRAG_PC = """#version 450
#extension GL_EXT_nonuniform_qualifier : require
layout(set = 0, binding = 0) uniform sampler2D tex[4];
layout(push_constant) uniform PC { uint k; } pc;
layout(location = 3) flat in uint v_id;
layout(location = 0) out vec4 color;
void main() { color = texture(tex[nonuniformEXT((v_id >> 1) + pc.k)], vec2(0.5)); }
"""
# the index comes from a storage buffer (uber-shader style)
FRAG_SSBO = """#version 450
#extension GL_EXT_nonuniform_qualifier : require
layout(set = 0, binding = 0) uniform sampler2D tex[4];
layout(std430, set = 0, binding = 1) readonly buffer P { uint ids[]; };
layout(location = 3) flat in uint v_id;
layout(location = 0) out vec4 color;
void main() { color = texture(tex[nonuniformEXT(ids[v_id])], vec2(0.5)); }
"""

HV = ["VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT", "VK_MEMORY_PROPERTY_HOST_COHERENT_BIT"]
CB = "command_buffer#1"
GFX = "VK_PIPELINE_BIND_POINT_GRAPHICS"


def _spv(src, stage, d, name="x"):
    p = Path(d) / f"{name}.{stage}"
    p.write_text(src)
    o = Path(d) / f"{name}_{stage}.spv"
    subprocess.run(["glslangValidator", "-V", "-o", str(o), str(p)], check=True, capture_output=True)
    sha = hashlib.sha1(o.read_bytes()).hexdigest()
    (Path(d) / "spv").mkdir(exist_ok=True)
    shutil.copy(o, Path(d) / "spv" / f"{sha}.spv")
    return sha


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
class DescRefine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="cov-descidx-"))
        cls.vs = _spv(VERT, "vert", cls.tmp)
        cls.fs = _spv(FRAG, "frag", cls.tmp)
        cls.fs_pc = _spv(FRAG_PC, "frag", cls.tmp, "pc")
        cls.fs_ssbo = _spv(FRAG_SSBO, "frag", cls.tmp, "ssbo")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def trace(self, fs=None, rewrite=None, ids=(0,), indexed=False, device_write=False):
        """rewrite: None, "racing" (host rewrites the instance data after the submit with no
        fence wait) or "after-fence" (the same rewrite after waiting for the fence)."""
        fs = fs or self.fs
        t = _Trace()
        t.add("covenant_trace_begin", version=1, host_write_tracking=True, page_size=4096)
        t.add("vkGetDeviceQueue", queue="queue#1", family=0, index=0)
        for i in (1, 2):
            t.add("vkCreateImage", image=f"image#{i}", format="VK_FORMAT_R8G8B8A8_UNORM",
                  usage=["VK_IMAGE_USAGE_SAMPLED_BIT", "VK_IMAGE_USAGE_TRANSFER_DST_BIT"], extent=[4, 4, 1],
                  mip_levels=1, array_layers=1, initial_layout="VK_IMAGE_LAYOUT_UNDEFINED")
            t.add("vkCreateImageView", view=f"image_view#{i}", image=f"image#{i}",
                  format="VK_FORMAT_R8G8B8A8_UNORM", view_type="VK_IMAGE_VIEW_TYPE_2D")
        t.add("vkCreateBuffer", buffer="buffer#1", size=4096, usage=["VK_BUFFER_USAGE_VERTEX_BUFFER_BIT"])
        t.add("vkAllocateMemory", memory="memory#1", size=4096, properties=HV)
        t.add("vkBindBufferMemory", buffer="buffer#1", memory="memory#1", offset=0, size=4096)
        t.add("vkMapMemory", memory="memory#1", offset=0, size=4096, tracked=True)
        bindings = [{"binding": 0, "type": "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER", "count": 4,
                     "stages": ["VK_SHADER_STAGE_FRAGMENT_BIT"], "immutable_samplers": False}]
        writes = [{"set": "ds#1", "binding": 0, "array_element": 0,
                   "type": "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER",
                   "elements": [{"view": f"image_view#{i}", "sampler": "sampler#1",
                                 "layout": "VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL"} for i in (1, 2)]}]
        if fs == self.fs_ssbo:
            t.add("vkCreateBuffer", buffer="buffer#2", size=256, usage=["VK_BUFFER_USAGE_STORAGE_BUFFER_BIT"])
            bindings.append({"binding": 1, "type": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER", "count": 1,
                             "stages": ["VK_SHADER_STAGE_FRAGMENT_BIT"], "immutable_samplers": False})
            writes.append({"set": "ds#1", "binding": 1, "array_element": 0,
                           "type": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
                           "elements": [{"buffer": "buffer#2", "offset": 0, "range": "whole"}]})
        t.add("vkCreateDescriptorSetLayout", layout="dsl#1", create_flags=0, bindings=bindings)
        t.add("vkAllocateDescriptorSets", pool="pool#1", sets=["ds#1"], set_layouts=["dsl#1"])
        t.add("vkUpdateDescriptorSets", copies=[], writes=writes)
        t.add("vkCreateShaderModule", module="sm#1", sha1=self.vs, code_size=1)
        t.add("vkCreateShaderModule", module="sm#2", sha1=fs, code_size=1)
        t.add("vkCreatePipelineLayout", layout="pl#1", set_layouts=["dsl#1"],
              push_constants=[{"stages": ["VK_SHADER_STAGE_FRAGMENT_BIT"], "offset": 0, "size": 4}])
        t.add("vkCreateGraphicsPipelines", pipelines=[{
            "pipeline": "pipeline#1", "layout": "pl#1", "render_pass": None, "subpass": 0,
            "stages": [{"stage": ["VK_SHADER_STAGE_VERTEX_BIT"], "module": "sm#1", "sha1": self.vs, "entry": "main"},
                       {"stage": ["VK_SHADER_STAGE_FRAGMENT_BIT"], "module": "sm#2", "sha1": fs, "entry": "main"}],
            "color_formats": [], "depth_format": None, "stencil_format": None,
            "vertex_bindings": [{"binding": 0, "stride": 8, "rate": "VK_VERTEX_INPUT_RATE_INSTANCE"}],
            "vertex_attributes": [{"location": 0, "binding": 0, "format": "VK_FORMAT_R32_UINT", "offset": 4}],
            "dynamic_states": []}])
        # the instance records: id at byte 4 of each 8-byte record
        data = b"".join((0x3F800000).to_bytes(4, "little") + i.to_bytes(4, "little") for i in ids)
        t.add("host_write", memory="memory#1", offset=0, page_size=4096, offsets=[0], writes=[[0, data.hex()]])
        t.add("vkAllocateCommandBuffers", pool="cp#1", level="VK_COMMAND_BUFFER_LEVEL_PRIMARY", command_buffers=[CB])
        t.add("vkBeginCommandBuffer", cb=CB, flags=[])
        t.add("vkCmdPipelineBarrier", cb=CB, src_stage=["VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT"],
              dst_stage=["VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT"], dependency_flags=0, memory_barriers=[],
              buffer_barriers=[], image_barriers=[{
                  "image": "image#1", "src_access": [], "dst_access": ["VK_ACCESS_SHADER_READ_BIT"],
                  "old_layout": "VK_IMAGE_LAYOUT_UNDEFINED", "new_layout": "VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL",
                  "src_family": -1, "dst_family": -1, "aspect": ["VK_IMAGE_ASPECT_COLOR_BIT"],
                  "base_mip": 0, "level_count": 1, "base_layer": 0, "layer_count": 1}])
        if device_write:
            t.add("vkCmdFillBuffer", cb=CB, dst="buffer#1", offset=0, size=4096, data=0)
            t.add("vkCmdPipelineBarrier", cb=CB, src_stage=["VK_PIPELINE_STAGE_TRANSFER_BIT"],
                  dst_stage=["VK_PIPELINE_STAGE_VERTEX_INPUT_BIT"], dependency_flags=0, memory_barriers=[{
                      "src_access": ["VK_ACCESS_TRANSFER_WRITE_BIT"],
                      "dst_access": ["VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT"]}],
                  buffer_barriers=[], image_barriers=[])
        t.add("vkCmdBindPipeline", cb=CB, bind_point=GFX, pipeline="pipeline#1")
        t.add("vkCmdBindDescriptorSets", cb=CB, bind_point=GFX, layout="pl#1", first_set=0, sets=["ds#1"],
              dynamic_offsets=[])
        t.add("vkCmdPushConstants", cb=CB, layout="pl#1", stages=["VK_SHADER_STAGE_FRAGMENT_BIT"],
              offset=0, size=4, data="00000000")
        t.add("vkCmdBindVertexBuffers", cb=CB, first_binding=0, buffers=["buffer#1"], offsets=[0])
        if indexed:
            t.add("vkCmdBindIndexBuffer", cb=CB, buffer="buffer#1", offset=0, index_type="VK_INDEX_TYPE_UINT16")
            t.add("vkCmdDrawIndexed", cb=CB, index_count=3, instance_count=len(ids), first_index=0,
                  vertex_offset=0, first_instance=0)
        else:
            t.add("vkCmdDraw", cb=CB, vertex_count=6, instance_count=len(ids), first_vertex=0, first_instance=0)
        t.add("vkEndCommandBuffer", cb=CB)
        t.add("vkCreateFence", fence="fence#1", signaled=False)
        t.add("vkQueueSubmit", queue="queue#1", submits=[{"wait": [], "command_buffers": [CB], "signal": []}],
              fence="fence#1")
        new = (0x3F800000).to_bytes(4, "little") + (2).to_bytes(4, "little")   # id 2: element 1
        if rewrite == "after-fence":
            t.add("vkWaitForFences", fences=["fence#1"], wait_all=True, timeout="infinite")
        if rewrite:
            t.add("host_write", memory="memory#1", offset=0, page_size=4096, offsets=[0], writes=[[0, new.hex()]])
        if rewrite != "after-fence":
            t.add("vkWaitForFences", fences=["fence#1"], wait_all=True, timeout="infinite")
        name = f"t{len(list(self.tmp.glob('*.jsonl')))}.jsonl"
        t.write(self.tmp / name)
        return load_trace(str(self.tmp / name), spirv_dir=str(self.tmp / "spv"))

    @staticmethod
    def codes(r, res):
        return sorted(d.code for d in r.items if f"'{res}'" in d.message)

    def check(self, m):
        r = timing.check(m.frame)
        return r, r.meta.get("timing.desc_refine", {})

    # -- the SPIR-V pattern -----------------------------------------------------------
    def test_index_terms_from_spirv(self):
        load = lambda sha: spirv.parse(self.tmp / "spv" / f"{sha}.spv")
        st = [("VK_SHADER_STAGE_VERTEX_BIT", self.vs, "main"), ("VK_SHADER_STAGE_FRAGMENT_BIT", self.fs, "main")]
        self.assertEqual(desc_index.pipeline_index_terms(st, load),
                         {(0, 0): frozenset({("attr", 0, (("shr", 1),))})})
        for fs in (self.fs_pc, self.fs_ssbo):
            st[1] = ("VK_SHADER_STAGE_FRAGMENT_BIT", fs, "main")
            self.assertIsNone(desc_index.pipeline_index_terms(st, load)[(0, 0)])

    # -- GTK-like: the false alarm goes away ---------------------------------------------
    def test_gtk_like_false_alarm_removed(self):
        m = self.trace()
        r, meta = self.check(m)
        self.assertEqual(self.codes(r, "image#2"), [], str(r))
        self.assertEqual(r.items, [], str(r))
        self.assertEqual(meta, {"draw_guards": 1, "widened_static": 0, "widened_dynamic": 0})

    def test_switch_off_charges_every_element(self):
        import os
        os.environ["COVENANT_NO_DESC_REFINE"] = "1"
        try:
            m = self.trace()
        finally:
            del os.environ["COVENANT_NO_DESC_REFINE"]
        r, _ = self.check(m)
        self.assertIn("timing.wrong-layout", self.codes(r, "image#2"))

    def test_sampled_element_still_checked(self):
        # the draw samples element 1 (id 2): the UNDEFINED image is reported
        m = self.trace(ids=(0, 2))
        r, _ = self.check(m)
        self.assertIn("timing.wrong-layout", self.codes(r, "image#2"))

    def test_ordered_rewrite_keeps_refinement(self):
        m = self.trace(rewrite="after-fence")
        r, meta = self.check(m)
        self.assertEqual(r.items, [], str(r))
        self.assertEqual(meta["widened_dynamic"], 0)

    # -- the index data races with the draw: widen and report -----------------------------
    def test_racing_rewrite_widens(self):
        m = self.trace(rewrite="racing")
        r, meta = self.check(m)
        self.assertIn("timing.cross-queue-race", self.codes(r, "buffer#1"))
        self.assertIn("timing.wrong-layout", self.codes(r, "image#2"))
        self.assertEqual(meta["widened_dynamic"], 1)

    # -- the index is not a function of the attribute alone: widen ----------------------------
    def test_push_constant_index_widens(self):
        r, meta = self.check(self.trace(fs=self.fs_pc))
        self.assertIn("timing.wrong-layout", self.codes(r, "image#2"))
        self.assertEqual(meta, {})

    def test_storage_buffer_index_widens(self):
        r, _ = self.check(self.trace(fs=self.fs_ssbo))
        self.assertIn("timing.wrong-layout", self.codes(r, "image#2"))

    def test_indexed_draw_widens(self):
        r, _ = self.check(self.trace(indexed=True))
        self.assertIn("timing.wrong-layout", self.codes(r, "image#2"))

    def test_device_written_vertex_memory_widens(self):
        r, meta = self.check(self.trace(device_write=True))
        self.assertIn("timing.wrong-layout", self.codes(r, "image#2"))
        self.assertEqual(meta["widened_static"], 1)

    # -- grouping drops the excluded reads of guards that hold, and only those ----------------
    def test_grouping_drops_held_exclusions(self):
        from covenant import report_groups
        for kw, dropped in (({}, True), ({"rewrite": "racing"}, False)):
            m = self.trace(**kw)
            ref = timing.check(m.frame, Report(), window=False)
            res = report_groups.group(m.frame)
            self.assertEqual(res["diagnostics"], len([d for d in ref.items]), kw)
            self.assertEqual(res["excluded_reads_dropped"] > 0, dropped, kw)
            self.assertEqual(res["status"], "complete")

    # -- Theorem 1 with guards: every reference race is reported -----------------------------
    def test_reference_semantics_agree(self):
        for kw in ({}, {"rewrite": "racing"}, {"rewrite": "after-fence"}, {"ids": (0, 2)},
                   {"device_write": True}):
            m = self.trace(**kw)
            hz = []
            timing.check(m.frame, Report(), hazards=hz, exhaustive=True, window=False)
            got = {frozenset((e, u)) for e, u, _ in hz}
            ref = timing_ref.race_keys(m.frame)
            self.assertLessEqual(ref, got, kw)


class GuardSetTest(unittest.TestCase):
    """The union-find behind epoch guard inheritance: a read that supersedes others stands for
    their guards (a hazard on it fails all of them), merged in near-linear time."""

    def test_merge_and_fail(self):
        gs = [timing.IndexGuard(snap) for snap in (30, 10, 20, 40)]
        sets = [timing._GSet(g) for g in gs]
        u = sets[0]
        for s in sets[1:3]:
            u = u.merge(s)
        self.assertEqual(u.root().snap, 10)
        failed = set()
        sets[2].fail(failed)
        self.assertEqual(failed, set(gs[:3]))
        failed.clear()
        sets[1].fail(failed)                      # already failed: nothing new
        self.assertEqual(failed, set())
        u = sets[3].merge(sets[0])                # a later member joins a failed set
        u.fail(failed)
        self.assertEqual(failed, {gs[3]})
        self.assertIs(sets[3].root(), sets[0].root())

    def test_long_chain_is_fast(self):
        u = timing._GSet(timing.IndexGuard(0))
        for i in range(1, 20000):
            u = timing._GSet(timing.IndexGuard(i)).merge(u)
        failed = set()
        u.fail(failed)
        self.assertEqual(len(failed), 20000)


if __name__ == "__main__":
    unittest.main()
