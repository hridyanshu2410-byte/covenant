"""The shader-to-event boundary and the reported configuration.

Small SPIR-V modules are assembled here from the opcode grammar shipped with
covenant (no shader compiler needed) and put into model-driver traces, which are
loaded with covenant/host_trace.py and checked in both modes and by the reference:

* an index taken from an atomic does not make two stores disjoint: atomicAdd(c, 0)
  hands every invocation of both dispatches index 0, so the two stores race;
* an atomic is exempt only at a scope that holds both commands (Device, ...);
* a missing or unreadable shader module never turns a storage write into a read:
  the access is taken as a write and the gap makes the check not guarantee-eligible;
* the configuration in report.meta is the one the check ran with (explicit
  arguments included), and a relaxed check is never called verified.
"""
import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from covenant import spirv, timing, timing_ref  # noqa: E402
from covenant.diagnostics import Report  # noqa: E402
from covenant.host_trace import load_trace  # noqa: E402

_GRAMMAR = os.path.join(os.path.dirname(spirv.__file__), "data", "spirv.core.grammar.json")
with open(_GRAMMAR) as _f:
    OPS = {x["opname"]: x["opcode"] for x in json.load(_f)["instructions"]}

DEVICE, WORKGROUP = 1, 2


def _string(s):
    b = s.encode() + b"\0"
    b += b"\0" * (-len(b) % 4)
    return list(struct.unpack("<%dI" % (len(b) // 4), b))


def storage_module(stage="compute", index="atomic", scope=DEVICE, value=123):
    """One storage buffer {uint counter; uint data[];} at set 0 binding 0.
    index "atomic": data[atomicAdd(counter, 0)] = value; "zero": data[0] = value;
    "atomic-only": atomicAdd(counter, 0) and nothing else."""
    w = [0x07230203, 0x00010300, 0, 40, 0]

    def e(op, *a):
        w.extend([(len(a) + 1) << 16 | OPS[op], *a])
    e("OpCapability", 1)
    e("OpMemoryModel", 0, 1)
    e("OpEntryPoint", 5 if stage == "compute" else 4, 20, *_string("main"))
    e("OpExecutionMode", 20, 17, 1, 1, 1) if stage == "compute" else e("OpExecutionMode", 20, 7)
    e("OpDecorate", 4, 2)                                   # Block
    e("OpMemberDecorate", 4, 0, 35, 0)
    e("OpMemberDecorate", 4, 1, 35, 4)
    e("OpDecorate", 11, 6, 4)                               # ArrayStride 4
    e("OpDecorate", 7, 34, 0)                               # DescriptorSet 0
    e("OpDecorate", 7, 33, 0)                               # Binding 0
    e("OpTypeVoid", 1)
    e("OpTypeFunction", 2, 1)
    e("OpTypeInt", 3, 32, 0)
    e("OpTypeRuntimeArray", 11, 3)
    e("OpTypeStruct", 4, 3, 11)
    e("OpTypePointer", 5, 12, 4)                            # StorageBuffer
    e("OpTypePointer", 6, 12, 3)
    e("OpVariable", 5, 7, 12)
    e("OpConstant", 3, 8, 0)
    e("OpConstant", 3, 9, 1)
    e("OpConstant", 3, 10, value)
    e("OpConstant", 3, 12, scope)
    e("OpFunction", 1, 20, 0, 2)
    e("OpLabel", 21)
    e("OpAccessChain", 6, 22, 7, 8)
    e("OpAtomicIAdd", 3, 23, 22, 12, 8, 8)                  # atomicAdd(counter, 0)
    if index != "atomic-only":
        e("OpAccessChain", 6, 24, 7, 9, 23 if index == "atomic" else 8)
        e("OpStore", 24, 10)
    e("OpReturn")
    e("OpFunctionEnd")
    return struct.pack("<%dI" % len(w), *w)


def vertex_module():
    """A vertex shader with no descriptor accesses."""
    w = [0x07230203, 0x00010300, 0, 40, 0]

    def e(op, *a):
        w.extend([(len(a) + 1) << 16 | OPS[op], *a])
    e("OpCapability", 1)
    e("OpMemoryModel", 0, 1)
    e("OpEntryPoint", 0, 20, *_string("main"))
    e("OpTypeVoid", 1)
    e("OpTypeFunction", 2, 1)
    e("OpFunction", 1, 20, 0, 2)
    e("OpLabel", 21)
    e("OpReturn")
    e("OpFunctionEnd")
    return struct.pack("<%dI" % len(w), *w)


def _sha(b):
    return hashlib.sha1(b).hexdigest()


def trace(modules, graphics=False):
    """Records: fill buffer#1, barrier transfer -> compute, then two dispatches
    (modules[0], modules[1]) with nothing between them; graphics: one draw whose
    fragment shader is modules[0], then an unsynchronized copy out of buffer#1."""
    recs = []

    def a(fn, **kw):
        recs.append(dict(seq=len(recs) + 1, fn=fn, **kw))
    usage = ["VK_BUFFER_USAGE_STORAGE_BUFFER_BIT", "VK_BUFFER_USAGE_TRANSFER_SRC_BIT",
             "VK_BUFFER_USAGE_TRANSFER_DST_BIT"]
    a("covenant_trace_begin", version=1, host_write_tracking=True, page_size=4096)
    a("vkGetDeviceQueue", queue="queue#1", family=0, index=0)
    for i in (1, 2):
        a("vkCreateBuffer", buffer=f"buffer#{i}", size=256, usage=usage, sharing="VK_SHARING_MODE_EXCLUSIVE")
        a("vkAllocateMemory", memory=f"memory#{i}", size=256, properties=["VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT"])
        a("vkBindBufferMemory", buffer=f"buffer#{i}", memory=f"memory#{i}", offset=0, size=256)
    stage = "VK_SHADER_STAGE_FRAGMENT_BIT" if graphics else "VK_SHADER_STAGE_COMPUTE_BIT"
    a("vkCreateDescriptorSetLayout", layout="dsl#1", create_flags=0,
      bindings=[dict(binding=0, type="VK_DESCRIPTOR_TYPE_STORAGE_BUFFER", count=1, stages=[stage],
                     immutable_samplers=False)])
    a("vkAllocateDescriptorSets", pool="pool#1", sets=["ds#1"], set_layouts=["dsl#1"])
    a("vkUpdateDescriptorSets", copies=[], writes=[dict(set="ds#1", binding=0, array_element=0,
      type="VK_DESCRIPTOR_TYPE_STORAGE_BUFFER", elements=[dict(buffer="buffer#1", offset=0, range=256)])])
    a("vkCreatePipelineLayout", layout="pl#1", set_layouts=["dsl#1"], push_constants=[])
    for k, sha in enumerate(modules if not graphics else modules[:1], 1):
        a("vkCreateShaderModule", module=f"sm#{k}", sha1=sha, code_size=4)
        stages = [dict(stage=[stage], module=f"sm#{k}", sha1=sha, entry="main")]
        if graphics:
            vsha = _sha(vertex_module())
            a("vkCreateShaderModule", module="sm#v", sha1=vsha, code_size=4)
            stages.insert(0, dict(stage=["VK_SHADER_STAGE_VERTEX_BIT"], module="sm#v", sha1=vsha, entry="main"))
        a("vkCreateGraphicsPipelines" if graphics else "vkCreateComputePipelines",
          pipelines=[dict(pipeline=f"pipeline#{k}", layout="pl#1", render_pass=None, subpass=0, stages=stages,
                          color_formats=[], depth_format=None, stencil_format=None, vertex_bindings=[],
                          vertex_attributes=[], dynamic_states=[])])
    a("vkAllocateCommandBuffers", pool="cp#1", level="VK_COMMAND_BUFFER_LEVEL_PRIMARY", command_buffers=["cb#1"])
    a("vkBeginCommandBuffer", cb="cb#1", flags=[])
    bp = "VK_PIPELINE_BIND_POINT_GRAPHICS" if graphics else "VK_PIPELINE_BIND_POINT_COMPUTE"
    if graphics:
        a("vkCmdBindPipeline", cb="cb#1", bind_point=bp, pipeline="pipeline#1")
        a("vkCmdBindDescriptorSets", cb="cb#1", bind_point=bp, layout="pl#1", first_set=0, sets=["ds#1"],
          dynamic_offsets=[])
        a("vkCmdBeginRendering", cb="cb#1", color=[], depth=None, stencil=None, render_area=[0, 0, 1, 1],
          layer_count=1, view_mask=0)
        a("vkCmdDraw", cb="cb#1", vertex_count=3, instance_count=1, first_vertex=0, first_instance=0)
        a("vkCmdEndRendering", cb="cb#1")
        a("vkCmdCopyBuffer", cb="cb#1", src="buffer#1", dst="buffer#2",
          regions=[dict(src_offset=0, dst_offset=0, size=4)])
    else:
        a("vkCmdFillBuffer", cb="cb#1", dst="buffer#1", offset=0, size=256, data=0)
        a("vkCmdPipelineBarrier", cb="cb#1", src_stage=["VK_PIPELINE_STAGE_TRANSFER_BIT"],
          dst_stage=["VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT"], dependency_flags=0,
          memory_barriers=[dict(src_access=["VK_ACCESS_TRANSFER_WRITE_BIT"],
                                dst_access=["VK_ACCESS_SHADER_READ_BIT", "VK_ACCESS_SHADER_WRITE_BIT"])],
          buffer_barriers=[], image_barriers=[])
        for k in range(1, len(modules) + 1):
            a("vkCmdBindPipeline", cb="cb#1", bind_point=bp, pipeline=f"pipeline#{k}")
            a("vkCmdBindDescriptorSets", cb="cb#1", bind_point=bp, layout="pl#1", first_set=0, sets=["ds#1"],
              dynamic_offsets=[])
            a("vkCmdDispatch", cb="cb#1", x=1, y=1, z=1)
    a("vkEndCommandBuffer", cb="cb#1")
    a("vkQueueSubmit", queue="queue#1", fence=None, submits=[dict(wait=[], command_buffers=["cb#1"], signal=[])])
    return recs


class BoundaryTest(unittest.TestCase):
    def load(self, recs, files):
        """files: {sha or name: bytes} written to the SPIR-V directory."""
        with tempfile.TemporaryDirectory() as d:
            for name, b in files.items():
                with open(os.path.join(d, f"{name}.spv"), "wb") as f:
                    f.write(b)
            with open(os.path.join(d, "t.jsonl"), "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in recs)
            return load_trace(os.path.join(d, "t.jsonl"), spirv_dir=d)

    def verdicts(self, m, **kw):
        out = []
        for exhaustive in (False, True):
            r = timing.check(m.frame, Report(), exhaustive=exhaustive, window=False, **kw)
            out.append((sorted(set(r.codes())), r.meta["covenant.eligibility"]))
        return out

    def compute(self, *mods):
        files = {_sha(b): b for b in mods}
        return self.load(trace([_sha(b) for b in mods]), files)

    def test_atomic_derived_index_is_not_a_fresh_slot(self):
        for second in (storage_module(), storage_module(value=456)):
            m = self.compute(storage_module(), second)
            self.assertEqual(len(timing_ref.races(m.frame)), 1)
            for codes, elig in self.verdicts(m):
                self.assertEqual(codes, ["timing.missing-barrier"])
                self.assertTrue(elig["guarantee_eligible"])
            # the opt-in heuristic accepts it, and says the result is outside the theorem
            for codes, elig in self.verdicts(m, slots=True):
                self.assertEqual(codes, [])
                self.assertFalse(elig["verified_configuration"])
                self.assertFalse(elig["guarantee_eligible"])

    def test_literal_index_control(self):
        m = self.compute(storage_module(index="zero"), storage_module(index="zero"))
        for codes, _ in self.verdicts(m):
            self.assertEqual(codes, ["timing.missing-barrier"])

    def test_atomic_exemption_needs_device_scope(self):
        m = self.compute(storage_module(index="atomic-only"), storage_module(index="atomic-only", value=7))
        self.assertEqual(timing_ref.races(m.frame), [])
        for codes, _ in self.verdicts(m):
            self.assertEqual(codes, [])
        mw = storage_module(index="atomic-only", scope=WORKGROUP)
        m = self.compute(mw, storage_module(index="atomic-only", scope=WORKGROUP, value=7))
        self.assertEqual(len(timing_ref.races(m.frame)), 1)
        for codes, _ in self.verdicts(m):
            self.assertEqual(codes, ["timing.missing-barrier"])

    def test_missing_or_unreadable_fragment_shader_fails_closed(self):
        frag = storage_module(stage="fragment", index="zero")
        recs = trace([_sha(frag)], graphics=True)
        vert = vertex_module()
        present = self.load(recs, {_sha(frag): frag, _sha(vert): vert})
        for codes, elig in self.verdicts(present):
            self.assertEqual(codes, ["timing.missing-barrier"])
            self.assertTrue(elig["guarantee_eligible"])
        for files in ({_sha(vert): vert}, {_sha(vert): vert, _sha(frag): b"not a module"}):
            m = self.load(recs, files)
            self.assertEqual(len(timing_ref.races(m.frame)), 1)
            for codes, elig in self.verdicts(m):
                self.assertEqual(codes, ["timing.missing-barrier"])
                self.assertIn("shader code not available", elig["model_gaps"])
                self.assertFalse(elig["guarantee_eligible"])

    def test_reported_configuration_is_the_one_checked(self):
        fr = timing.Frame(queues={"q": [
            timing.Use("img", "write", "general", "write", "TRANSFER", "TRANSFER_WRITE"),
            timing.Barrier("img", None, "general", "discard", frozenset({"TRANSFER"}), frozenset(),
                           frozenset({"TRANSFER"}), frozenset())]}, initial_layouts={"img": "general"})
        strict = timing.check(fr, Report(), window=False)
        self.assertEqual(strict.codes(), ["timing.barrier-scope"])
        self.assertTrue(strict.meta["covenant.eligibility"]["verified_configuration"])
        relaxed = timing.check(fr, Report(), discard_relaxed=True, window=False)
        self.assertEqual(relaxed.codes(), [])
        self.assertEqual(relaxed.meta["covenant.modes"]["discard"], "heuristic-R1")
        self.assertFalse(relaxed.meta["covenant.modes"]["verified_configuration"])
        self.assertFalse(relaxed.meta["covenant.eligibility"]["guarantee_eligible"])
        # a trace loaded with observed capture is not verified, whatever the environment says now
        fr.host_capture = "observed"
        r = timing.check(fr, Report(), window=False)
        self.assertEqual(r.meta["covenant.modes"]["host_capture"], "observed")
        self.assertFalse(r.meta["covenant.eligibility"]["guarantee_eligible"])


if __name__ == "__main__":
    unittest.main()
