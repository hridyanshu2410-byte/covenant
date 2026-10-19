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
  arguments included), and a relaxed check is never called verified;
* a helper function called with two different resources writes both: descriptor
  provenance keeps every root a parameter can have (call order, nesting, returned
  and selected pointers), and an access it cannot attribute fails closed;
* atomic exemptions compare buffer bytes, not member numbers: one shader run
  through two descriptor offsets of one buffer, whose ordinary read of one run
  overlaps the atomic of the other, races;
* a layout transition orders nothing between two barriers: the transitions of two
  image barriers do not chain them, in the checked frame or in the optimizer's
  frame where each transition is split into two halves around a pseudo-stage.
"""
import copy
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


def two_buffer_module(form):
    """Storage buffers A (binding 0) and B (binding 1), each {uint x;}; writes 123
    to A.x and/or B.x: "inline", "helper" (store(A.x); store(B.x)), "reversed"
    (store(B.x); store(A.x)), "nested" (outer(p) calls store(p)), "return"
    (store through pick(A.x), a helper returning its parameter), "select"
    (OpSelect between A.x and B.x), "undef" (a store through an undefined
    storage-buffer pointer)."""
    w = [0x07230203, 0x00010300, 0, 60, 0]

    def e(op, *a):
        w.extend([(len(a) + 1) << 16 | OPS[op], *a])
    e("OpCapability", 1)
    e("OpMemoryModel", 0, 1)
    e("OpEntryPoint", 5, 20, *_string("main"))
    e("OpExecutionMode", 20, 17, 1, 1, 1)
    e("OpDecorate", 4, 2)
    e("OpMemberDecorate", 4, 0, 35, 0)
    for var, binding in ((7, 0), (8, 1)):
        e("OpDecorate", var, 34, 0)
        e("OpDecorate", var, 33, binding)
    e("OpTypeVoid", 1)
    e("OpTypeFunction", 2, 1)
    e("OpTypeInt", 3, 32, 0)
    e("OpTypeStruct", 4, 3)
    e("OpTypePointer", 5, 12, 4)
    e("OpTypePointer", 6, 12, 3)
    e("OpTypeBool", 15)
    e("OpTypeFunction", 13, 1, 6)
    e("OpTypeFunction", 14, 6, 6)
    e("OpVariable", 5, 7, 12)
    e("OpVariable", 5, 8, 12)
    e("OpConstant", 3, 9, 0)
    e("OpConstant", 3, 10, 123)
    e("OpConstantTrue", 15, 16)
    e("OpFunction", 1, 20, 0, 2)
    e("OpLabel", 21)
    e("OpAccessChain", 6, 22, 7, 9)
    e("OpAccessChain", 6, 23, 8, 9)
    if form == "inline":
        e("OpStore", 22, 10)
        e("OpStore", 23, 10)
    elif form in ("helper", "reversed", "nested"):
        f = 40 if form == "nested" else 30
        first, second = (23, 22) if form == "reversed" else (22, 23)
        e("OpFunctionCall", 1, 24, f, first)
        e("OpFunctionCall", 1, 25, f, second)
    elif form == "return":
        e("OpFunctionCall", 6, 26, 50, 22)
        e("OpStore", 26, 10)
    elif form == "select":
        e("OpSelect", 6, 26, 16, 22, 23)
        e("OpStore", 26, 10)
    elif form == "undef":
        e("OpUndef", 6, 26)
        e("OpStore", 26, 10)
    e("OpReturn")
    e("OpFunctionEnd")
    e("OpFunction", 1, 30, 0, 13)                           # store(p): *p = 123
    e("OpFunctionParameter", 6, 31)
    e("OpLabel", 32)
    e("OpStore", 31, 10)
    e("OpReturn")
    e("OpFunctionEnd")
    e("OpFunction", 1, 40, 0, 13)                           # outer(p): store(p)
    e("OpFunctionParameter", 6, 41)
    e("OpLabel", 42)
    e("OpFunctionCall", 1, 43, 30, 41)
    e("OpReturn")
    e("OpFunctionEnd")
    e("OpFunction", 6, 50, 0, 14)                           # pick(p): return p
    e("OpFunctionParameter", 6, 51)
    e("OpLabel", 52)
    e("OpReturnValue", 51)
    e("OpFunctionEnd")
    return struct.pack("<%dI" % len(w), *w)


def two_buffer_trace(sha):
    """buffer#1 (binding 0) and buffer#2 (binding 1), filled and made visible to
    the compute stage; one dispatch of the module; then a copy out of buffer#1
    with no dependency on the dispatch."""
    recs = trace([sha])
    out = []
    for r in recs:
        r = copy.deepcopy(r)
        if r["fn"] == "vkCreateDescriptorSetLayout":
            r["bindings"].append(dict(r["bindings"][0], binding=1))
        if r["fn"] == "vkUpdateDescriptorSets":
            r["writes"].append(copy.deepcopy(r["writes"][0]))
            r["writes"][1]["binding"] = 1
            r["writes"][1]["elements"][0]["buffer"] = "buffer#2"
        if r["fn"] == "vkCmdPipelineBarrier":     # the fills are also visible to the copy
            r["dst_stage"] = r["dst_stage"] + ["VK_PIPELINE_STAGE_TRANSFER_BIT"]
            r["memory_barriers"][0]["dst_access"] = r["memory_barriers"][0]["dst_access"] + \
                ["VK_ACCESS_TRANSFER_READ_BIT"]
            out.append(dict(seq=0, fn="vkCmdFillBuffer", cb="cb#1", dst="buffer#2", offset=0, size=256, data=0))
        out.append(r)
        if r.get("buffer") == "buffer#2":       # buffer#3: the copy's destination
            out.append({k: (v.replace("#2", "#3") if isinstance(v, str) else v) for k, v in r.items()})
        if r["fn"] == "vkCmdDispatch":
            out.append(dict(seq=0, fn="vkCmdCopyBuffer", cb="cb#1", src="buffer#1", dst="buffer#3",
                            regions=[dict(src_offset=0, dst_offset=0, size=4)]))
    for k, r in enumerate(out, 1):
        r["seq"] = k
    return out


def counter_module():
    """{uint counter (offset 0); uint source (offset 256);}: atomicAdd(counter, source)."""
    w = [0x07230203, 0x00010300, 0, 40, 0]

    def e(op, *a):
        w.extend([(len(a) + 1) << 16 | OPS[op], *a])
    e("OpCapability", 1)
    e("OpMemoryModel", 0, 1)
    e("OpEntryPoint", 5, 20, *_string("main"))
    e("OpExecutionMode", 20, 17, 1, 1, 1)
    e("OpDecorate", 4, 2)
    e("OpMemberDecorate", 4, 0, 35, 0)
    e("OpMemberDecorate", 4, 1, 35, 256)
    e("OpDecorate", 7, 34, 0)
    e("OpDecorate", 7, 33, 0)
    e("OpTypeVoid", 1)
    e("OpTypeFunction", 2, 1)
    e("OpTypeInt", 3, 32, 0)
    e("OpTypeStruct", 4, 3, 3)
    e("OpTypePointer", 5, 12, 4)
    e("OpTypePointer", 6, 12, 3)
    e("OpVariable", 5, 7, 12)
    e("OpConstant", 3, 8, 0)
    e("OpConstant", 3, 9, 1)
    e("OpFunction", 1, 20, 0, 2)
    e("OpLabel", 21)
    e("OpAccessChain", 6, 22, 7, 9)
    e("OpLoad", 3, 23, 22)
    e("OpAccessChain", 6, 24, 7, 8)
    e("OpAtomicIAdd", 3, 25, 24, 9, 8, 23)                  # Device scope
    e("OpReturn")
    e("OpFunctionEnd")
    return struct.pack("<%dI" % len(w), *w)


def shifted_trace(sha, second_offset):
    """One 1,024-byte buffer seen through two descriptor sets (offsets 0 and
    second_offset, range 512); the module dispatched once with each, nothing between."""
    out = []
    for r in trace([sha, sha]):
        r = copy.deepcopy(r)
        if r["fn"] in ("vkCreateBuffer", "vkAllocateMemory", "vkBindBufferMemory", "vkCmdFillBuffer"):
            r["size"] = 1024
        if r["fn"] == "vkAllocateDescriptorSets":
            r["sets"], r["set_layouts"] = ["ds#1", "ds#2"], ["dsl#1", "dsl#1"]
        if r["fn"] == "vkUpdateDescriptorSets":
            r["writes"][0]["elements"][0]["range"] = 512
            r["writes"].append(copy.deepcopy(r["writes"][0]))
            r["writes"][1]["set"] = "ds#2"
            r["writes"][1]["elements"][0]["offset"] = second_offset
        out.append(r)
    binds = [r for r in out if r["fn"] == "vkCmdBindDescriptorSets"]
    binds[1]["sets"] = ["ds#2"]
    return out


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


    def test_helper_called_with_two_resources_writes_both(self):
        both = {(0, 0): True, (0, 1): True}
        for form in ("inline", "helper", "reversed", "nested", "select"):
            with tempfile.NamedTemporaryFile(suffix=".spv") as f:
                f.write(two_buffer_module(form))
                f.flush()
                mod = spirv.parse(f.name)
            self.assertEqual(spirv.entry_bindings(mod, "main"), both, form)
        with tempfile.NamedTemporaryFile(suffix=".spv") as f:
            f.write(two_buffer_module("return"))
            f.flush()
            self.assertEqual(spirv.entry_bindings(spirv.parse(f.name), "main"), {(0, 0): True, (0, 1): False})
        # end to end: the copy out of A races with the dispatch whatever the call order
        for form in ("inline", "helper", "reversed", "nested"):
            b = two_buffer_module(form)
            m = self.load(two_buffer_trace(_sha(b)), {_sha(b): b})
            self.assertEqual(len(timing_ref.races(m.frame)), 1, form)
            for codes, elig in self.verdicts(m):
                self.assertEqual(codes, ["timing.missing-barrier"], form)
                self.assertTrue(elig["guarantee_eligible"], form)

    def test_unattributed_access_fails_closed(self):
        b = two_buffer_module("undef")
        with tempfile.NamedTemporaryFile(suffix=".spv") as f:
            f.write(b)
            f.flush()
            with self.assertRaises(spirv.AnalysisIncomplete):
                spirv.entry_bindings(spirv.parse(f.name), "main")
        m = self.load(two_buffer_trace(_sha(b)), {_sha(b): b})
        for codes, elig in self.verdicts(m):
            self.assertEqual(codes, ["timing.missing-barrier"])
            self.assertIn("shader accesses not attributed", elig["model_gaps"])
            self.assertFalse(elig["guarantee_eligible"])

    def test_atomic_exemption_compares_bytes_not_members(self):
        b = counter_module()
        # same offset: atomics meet atomics, reads meet reads, so no race
        m = self.load(shifted_trace(_sha(b), 0), {_sha(b): b})
        self.assertEqual(timing_ref.races(m.frame), [])
        for codes, elig in self.verdicts(m):
            self.assertEqual(codes, [])
            self.assertTrue(elig["guarantee_eligible"])
        # offset 256: the first run's read of bytes [256, 260) meets the second run's atomic
        m = self.load(shifted_trace(_sha(b), 256), {_sha(b): b})
        self.assertEqual(len(timing_ref.races(m.frame)), 1)
        for codes, elig in self.verdicts(m):
            self.assertEqual(codes, ["timing.missing-barrier"])
            self.assertTrue(elig["guarantee_eligible"])
        # offset 512: the two runs touch disjoint bytes
        m = self.load(shifted_trace(_sha(b), 512), {_sha(b): b})
        for codes, _ in self.verdicts(m):
            self.assertEqual(codes, [])

    def test_transitions_do_not_chain_two_barriers(self):
        F, B, U = frozenset, timing.Barrier, timing.Use
        lt, ltf = F({timing.LT_STAGE}), F({timing.LT_FLAG})
        w = U("img", "write", "general", "W", "TRANSFER", "TRANSFER_WRITE")
        r = U("img", "read", "shader_read", "R", "FRAGMENT_SHADER", "SHADER_READ")
        x = ("img", "general", "transfer_src", "X", F({"TRANSFER"}), F({"TRANSFER_WRITE"}))
        y = ("img", "transfer_src", "shader_read", "Y", F(), F())
        plain = [w, B(*x, F(), F(), call="c1", struct=0), B(*y, F({"FRAGMENT_SHADER"}), F({"SHADER_READ"}),
                                                              call="c2", struct=0), r]
        # the optimizer's frame: B1 (to LT) ; the transition ; B2 (from LT), one call and struct each
        split = [w]
        for (res, old, new, where, ss, sa), ds, da, call in ((x, F(), F(), "c1"),
                                                             (y, F({"FRAGMENT_SHADER"}), F({"SHADER_READ"}), "c2")):
            split += [B(res, old, new, where, ss, sa, lt, ltf, call=call, struct=0),
                      U(res, "write", None, where + " (layout transition)", timing.LT_STAGE, timing.LT_FLAG,
                        call=call, struct=0),
                      B(res, None, None, where + " (after)", lt, ltf, ds, da, call=call, struct=0)]
        split.append(r)
        for ops in (plain, split):
            fr = timing.Frame(queues={"q": ops}, initial_layouts={"img": "general"})
            races = timing_ref.races(fr)
            self.assertTrue(any(a.op is w and b.op is r for a, b in races))
            # the optimizer checks its frames pairwise; epoch mode may report a stand-in instead
            for exhaustive in ((False, True) if ops is plain else (True,)):
                rep = timing.check(fr, Report(), exhaustive=exhaustive, window=False)
                self.assertTrue(any(d.where == "R" for d in rep.items))
        # one split transition still links its own halves: W before X's transition and R
        fr = timing.Frame(queues={"q": [w] + split[1:4] + [r]}, initial_layouts={"img": "general"})
        self.assertFalse(any(a.op is w for a, b in timing_ref.races(fr) if b.op.where.endswith("transition)")))

if __name__ == "__main__":
    unittest.main()
