"""Ray tracing, mesh shading and compute in the timing model.

Each test builds a small model-driver trace (the records libcovvk writes),
loads it with covenant/host_trace.py and checks it with covenant/timing.py.
Every frame is also decided by the reference semantics (covenant/timing_ref.py):
the pairwise checker must report exactly the reference's races.

Mutants: each correct program has a twin with one barrier removed or narrowed
(BLAS->TLAS, build->trace, compute write->mesh indirect read, ...); the
correct one must be clean and the mutant must be reported.

The random tests at the end rerun the Theorem 1 property tests of
test_timing_ref.py with the new stages (mesh, ray tracing, acceleration
structure build/copy) and access flags in the generator.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from covenant import timing, timing_ref  # noqa: E402
from covenant.diagnostics import Report  # noqa: E402
from covenant.host_trace import load_trace  # noqa: E402
import timing_gen  # noqa: E402

S2 = "VK_PIPELINE_STAGE_2_"
A2 = "VK_ACCESS_2_"
BUILD, COPY = S2 + "ACCELERATION_STRUCTURE_BUILD_BIT_KHR", S2 + "ACCELERATION_STRUCTURE_COPY_BIT_KHR"
RTS, FS_, CS = S2 + "RAY_TRACING_SHADER_BIT_KHR", S2 + "FRAGMENT_SHADER_BIT", S2 + "COMPUTE_SHADER_BIT"
DI, TASK, MESH = S2 + "DRAW_INDIRECT_BIT", S2 + "TASK_SHADER_BIT_EXT", S2 + "MESH_SHADER_BIT_EXT"
XFER, HOST = S2 + "TRANSFER_BIT", S2 + "HOST_BIT"
ASR, ASW = A2 + "ACCELERATION_STRUCTURE_READ_BIT_KHR", A2 + "ACCELERATION_STRUCTURE_WRITE_BIT_KHR"
SBT, SR = A2 + "SHADER_BINDING_TABLE_READ_BIT_KHR", A2 + "SHADER_READ_BIT"
SW, SSW = A2 + "SHADER_WRITE_BIT", A2 + "SHADER_STORAGE_WRITE_BIT"
ICR, TW = A2 + "INDIRECT_COMMAND_READ_BIT", A2 + "TRANSFER_WRITE_BIT"
DT = "VK_DESCRIPTOR_TYPE_"


class Trace:
    """Builds model-driver records. Buffers: buffer#N; one queue."""

    def __init__(self):
        self.recs, self.cb = [], "command_buffer#1"
        self.r("vkGetDeviceQueue", queue="queue#1", family=0, index=0)
        self.r("vkAllocateCommandBuffers", pool="command_pool#1", command_buffers=[self.cb])
        self.r("vkBeginCommandBuffer", cb=self.cb, flags=[])

    def r(self, fn, **kw):
        self.recs.append(dict(seq=len(self.recs) + 1, fn=fn, **kw))
        return self

    def c(self, fn, **kw):
        return self.r(fn, cb=self.cb, **kw)

    def buffer(self, b, size=65536):
        return self.r("vkCreateBuffer", buffer=b, size=size, usage=[], sharing="VK_SHARING_MODE_CONCURRENT")

    def accel(self, a, buf, typ, offset=0, size=4352):
        return self.r("vkCreateAccelerationStructureKHR", acceleration_structure=a, buffer=buf,
                      offset=offset, size=size, type=typ, address=0)

    def build(self, *builds, indirect=None):
        kw = {"indirect": indirect} if indirect else {}
        fn = "vkCmdBuildAccelerationStructuresIndirectKHR" if indirect else "vkCmdBuildAccelerationStructuresKHR"
        return self.c(fn, builds=list(builds), **kw)

    @staticmethod
    def blas(dst, scratch="buffer#90", vb="buffer#80", mode="build", src=None, scratch_off=0):
        return {"type": "bottom", "mode": mode, "flags": 0, "src": src, "dst": dst,
                "scratch": {"buffer": scratch, "offset": scratch_off, "size": 1280},
                "geometries": [{"primitive_count": 1, "known_count": True, "type": "triangles",
                                "vertex": {"buffer": vb, "offset": 0, "size": 36}}]}

    @staticmethod
    def tlas(dst, scratch="buffer#91", inst="buffer#81"):
        return {"type": "top", "mode": "build", "flags": 0, "src": None, "dst": dst,
                "scratch": {"buffer": scratch, "offset": 0, "size": 1280},
                "geometries": [{"primitive_count": 1, "known_count": True, "type": "instances",
                                "array_of_pointers": False, "data": {"buffer": inst, "offset": 0, "size": 64}}]}

    def barrier(self, src, sacc, dst, dacc, buffer=None):
        b = {"src_stage": src, "src_access": sacc, "dst_stage": dst, "dst_access": dacc}
        if buffer:
            b.update(buffer=buffer, offset=0, size="whole")
            return self.c("vkCmdPipelineBarrier2", buffer_barriers=[b], image_barriers=[], memory_barriers=[])
        return self.c("vkCmdPipelineBarrier2", memory_barriers=[b], buffer_barriers=[], image_barriers=[])

    def layout(self, bindings):
        """One set layout + pipeline layout + allocated set with contents."""
        self.r("vkCreateDescriptorSetLayout", layout="dsl#1",
               bindings=[{"binding": i, "type": t, "count": 1, "stages": st} for i, (t, st, _) in enumerate(bindings)])
        self.r("vkCreatePipelineLayout", layout="pl#1", set_layouts=["dsl#1"], push_constants=[])
        self.r("vkAllocateDescriptorSets", sets=["set#1"], set_layouts=["dsl#1"])
        self.r("vkUpdateDescriptorSets", copies=[], writes=[
            {"set": "set#1", "binding": i, "array_element": 0, "type": t, "elements": [el]}
            for i, (t, _, el) in enumerate(bindings)])
        return self

    def pipeline(self, fn, pid, stages, bp, bind_set=True):
        st = [{"stage": [s], "sha1": f"sha-{pid}-{i}", "entry": "main"} for i, s in enumerate(stages)]
        self.r(fn, pipelines=[{"pipeline": pid, "layout": "pl#1", "stages": st, "groups": [], "libraries": []}])
        self.c("vkCmdBindPipeline", bind_point=bp, pipeline=pid)
        if not bind_set:
            return self
        return self.c("vkCmdBindDescriptorSets", bind_point=bp, first_set=0, sets=["set#1"], layout="pl#1",
                      dynamic_offsets=[])

    def trace_rays(self, sbt="buffer#70", **kw):
        reg = lambda o: {"buffer": sbt, "offset": o, "size": 32}  # noqa: E731
        return self.c("vkCmdTraceRaysKHR", raygen=reg(0), miss=reg(64), hit=reg(128), callable=None,
                      size=[8, 8, 1], **kw)

    def submit(self, tlas_refs=None):
        self.r("vkEndCommandBuffer", cb=self.cb)
        for cmd_seq, build, refs in tlas_refs or ():
            self.r("covenant_tlas_refs", cb=self.cb, cmd_seq=cmd_seq, build=build, resolved=True, refs=refs)
        return self.r("vkQueueSubmit", queue="queue#1", fence=None,
                      submits=[{"wait": [], "command_buffers": [self.cb], "signal": []}])

    def last_seq(self):
        return self.recs[-1]["seq"]

    def load(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for x in self.recs:
                f.write(json.dumps(x) + "\n")
        try:
            return load_trace(f.name, spirv_dir="")
        finally:
            os.unlink(f.name)


def spec_pairs(fr):
    return {frozenset((a.key, b.key)) for a, b in timing_ref.Reference(fr, False).races()}


def checker_pairs(fr):
    hz = []
    timing.check(fr, Report(), hazards=hz, exhaustive=True, window=False)
    return {frozenset((a, b)) for a, b, _ in hz}


def with_buffers(t, *names):
    for b in names:
        t.buffer(b)
    return t


class Base(unittest.TestCase):
    def check(self, t, clean, res=None):
        """Load, check, compare with the reference; returns the report."""
        m = t.load()
        fr = m.frame
        rep = timing.check(fr, Report())
        spec, chk = spec_pairs(fr), checker_pairs(fr)
        self.assertEqual(chk, spec, "checker and reference disagree")
        errs = [d for d in rep.items if d.severity == "error"]
        if clean:
            self.assertEqual(errs, [], "\n".join(map(str, errs)))
            self.assertFalse(spec)
        else:
            self.assertTrue(errs, "mutant not reported")
            self.assertTrue(spec, "reference finds no race in the mutant")
            if res:
                self.assertTrue(any(f"'{res}'" in d.message for d in errs), "\n".join(map(str, errs)))
        return m, rep


# ---------------------------------------------------------------- acceleration structures
class AccelerationStructures(Base):
    def blas_tlas(self, barrier):
        t = with_buffers(Trace(), "buffer#1", "buffer#2", "buffer#80", "buffer#81", "buffer#90", "buffer#91")
        t.accel("as#1", "buffer#1", "bottom").accel("as#2", "buffer#2", "top")
        t.build(Trace.blas("as#1"))
        if barrier:
            t.barrier(*barrier)
        t.build(Trace.tlas("as#2"))
        return t.submit([(t.last_seq(), 0, ["as#1"])])

    def test_blas_then_tlas_with_barrier_is_clean(self):
        m, _ = self.check(self.blas_tlas(([BUILD], [ASW], [BUILD], [ASR])), clean=True)
        self.assertEqual(m.accel["as#2"].refs, frozenset({"as#1"}))

    def test_mutant_blas_tlas_missing_barrier(self):
        self.check(self.blas_tlas(None), clean=False, res="buffer#1")

    def test_mutant_blas_tlas_barrier_without_as_read(self):
        self.check(self.blas_tlas(([BUILD], [ASW], [BUILD], [SR])), clean=False, res="buffer#1")

    def test_tlas_in_same_build_call_as_its_blas(self):
        # builds of one vkCmdBuildAccelerationStructuresKHR are unordered (VUID-03403 family)
        t = with_buffers(Trace(), "buffer#1", "buffer#2", "buffer#80", "buffer#81", "buffer#90", "buffer#91")
        t.accel("as#1", "buffer#1", "bottom").accel("as#2", "buffer#2", "top")
        t.build(Trace.blas("as#1"), Trace.tlas("as#2"))
        self.check(t.submit([(t.last_seq(), 1, ["as#1"])]), clean=False, res="buffer#1")

    def test_scratch_reuse_needs_barrier(self):
        def prog(barrier):
            t = with_buffers(Trace(), "buffer#1", "buffer#3", "buffer#80", "buffer#90")
            t.accel("as#1", "buffer#1", "bottom").accel("as#3", "buffer#3", "bottom")
            t.build(Trace.blas("as#1"))
            if barrier:
                t.barrier([BUILD], [ASW], [BUILD], [ASR, ASW])
            t.build(Trace.blas("as#3"))
            return t.submit()
        self.check(prog(True), clean=True)
        self.check(prog(False), clean=False, res="buffer#90")

    def test_scratch_shared_in_one_call(self):
        t = with_buffers(Trace(), "buffer#1", "buffer#3", "buffer#80", "buffer#90")
        t.accel("as#1", "buffer#1", "bottom").accel("as#3", "buffer#3", "bottom")
        t.build(Trace.blas("as#1"), Trace.blas("as#3"))
        self.check(t.submit(), clean=False, res="buffer#90")
        # disjoint scratch ranges in one call: fine
        t = with_buffers(Trace(), "buffer#1", "buffer#3", "buffer#80", "buffer#90")
        t.accel("as#1", "buffer#1", "bottom").accel("as#3", "buffer#3", "bottom")
        t.build(Trace.blas("as#1"), Trace.blas("as#3", scratch_off=4096))
        self.check(t.submit(), clean=True)

    def test_build_input_written_by_transfer(self):
        def prog(dacc):
            t = with_buffers(Trace(), "buffer#1", "buffer#80", "buffer#90", "buffer#60")
            t.accel("as#1", "buffer#1", "bottom")
            t.c("vkCmdCopyBuffer", src="buffer#60", dst="buffer#80",
                regions=[{"src_offset": 0, "dst_offset": 0, "size": 36}])
            t.barrier([XFER], [TW], [BUILD], dacc)
            t.build(Trace.blas("as#1"))
            return t.submit()
        self.check(prog([SR]), clean=True)                    # spec: SHADER_READ at BUILD
        self.check(prog([ASR]), clean=False, res="buffer#80")  # inputs are not AS reads

    def test_update_reads_source(self):
        t = with_buffers(Trace(), "buffer#1", "buffer#80", "buffer#90")
        t.accel("as#1", "buffer#1", "bottom")
        t.build(Trace.blas("as#1"))
        t.barrier([BUILD], [ASW], [BUILD], [ASW])            # scratch WAW ordered, AS read not
        t.build(Trace.blas("as#1", mode="update", src="as#1"))
        self.check(t.submit(), clean=False, res="buffer#1")

    def test_indirect_build_range(self):
        def prog(barrier):
            t = with_buffers(Trace(), "buffer#1", "buffer#80", "buffer#90", "buffer#50")
            t.accel("as#1", "buffer#1", "bottom")
            t.c("vkCmdFillBuffer", dst="buffer#50", offset=0, size=16, data=0)
            t.barrier([XFER], [TW], [BUILD] if barrier else [DI], [ICR])
            t.build(Trace.blas("as#1"), indirect=[{"range": {"buffer": "buffer#50", "offset": 0, "size": 16}}])
            return t.submit()
        self.check(prog(True), clean=True)
        self.check(prog(False), clean=False, res="buffer#50")

    def test_copy_stage_held_by_build_mask(self):
        def prog(src_stage):
            t = with_buffers(Trace(), "buffer#1", "buffer#4", "buffer#80", "buffer#90")
            t.accel("as#1", "buffer#1", "bottom").accel("as#4", "buffer#4", "bottom")
            t.build(Trace.blas("as#1"))
            t.barrier([BUILD], [ASW], [COPY], [ASR])
            t.c("vkCmdCopyAccelerationStructureKHR", src="as#1", dst="as#4", mode="compact")
            t.barrier(src_stage, [ASW], [RTS], [ASR])
            t.c("vkCmdCopyAccelerationStructureToMemoryKHR", src="as#4", mode="serialize",
                dst={"buffer": "buffer#90", "offset": 0, "size": 64})
            return t.submit()
        # serialization reads as#4 at the COPY stage: ordered by BUILD or COPY masks...
        self.check(prog([BUILD]), clean=False, res="buffer#4")    # ...but the dst stage is RTS
        t = with_buffers(Trace(), "buffer#1", "buffer#4", "buffer#80", "buffer#90")
        t.accel("as#1", "buffer#1", "bottom").accel("as#4", "buffer#4", "bottom")
        t.build(Trace.blas("as#1"))
        t.barrier([BUILD], [ASW], [BUILD], [ASR])             # BUILD holds COPY (M2)
        t.c("vkCmdCopyAccelerationStructureKHR", src="as#1", dst="as#4", mode="clone")
        t.barrier([COPY], [ASW], [BUILD], [ASR])
        t.c("vkCmdWriteAccelerationStructuresPropertiesKHR", acceleration_structures=["as#4"],
            query_type=0, query_pool="qp#1", first_query=0)
        self.check(t.submit(), clean=True)


# ---------------------------------------------------------------- ray tracing
class RayTracing(Base):
    def program(self, barrier, refs=True, stage=RTS, extra=None):
        t = with_buffers(Trace(), "buffer#1", "buffer#2", "buffer#70", "buffer#80", "buffer#81", "buffer#90",
                         "buffer#91")
        t.accel("as#1", "buffer#1", "bottom").accel("as#2", "buffer#2", "top")
        t.layout([(DT + "ACCELERATION_STRUCTURE_KHR", ["VK_SHADER_STAGE_RAYGEN_BIT_KHR"],
                   {"acceleration_structure": "as#2"})])
        t.build(Trace.blas("as#1"))
        t.barrier([BUILD], [ASW], [BUILD], [ASR])
        t.build(Trace.tlas("as#2"))
        tl = t.last_seq()
        if barrier:
            t.barrier(*barrier)
        if extra:
            extra(t)
        t.pipeline("vkCreateRayTracingPipelinesKHR", "pipeline#1",
                   ["VK_SHADER_STAGE_RAYGEN_BIT_KHR", "VK_SHADER_STAGE_MISS_BIT_KHR"],
                   "VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR")
        t.trace_rays()
        return t.submit([(tl, 0, ["as#1"])] if refs else None)

    def test_build_then_trace_with_barrier_is_clean(self):
        m, _ = self.check(self.program(([BUILD], [ASW], [RTS], [ASR])), clean=True)
        self.assertEqual([d.kind for d in m.draws], ["trace"])

    def test_mutant_build_trace_missing_barrier(self):
        self.check(self.program(None), clean=False, res="buffer#2")

    def test_mutant_build_trace_wrong_dst_stage(self):
        self.check(self.program(([BUILD], [ASW], [FS_], [ASR])), clean=False, res="buffer#2")

    def test_trace_reads_blas_through_tlas(self):
        # the TLAS is ordered by a buffer barrier; the BLAS it references is not
        def extra(t):
            t.barrier([BUILD], [ASW], [RTS], [ASR], buffer="buffer#2")
        self.check(self.program(None, extra=extra), clean=False, res="buffer#1")

    def test_unresolved_tlas_refs_fall_back_to_live_blases(self):
        m, rep = self.check(self.program(([BUILD], [ASW], [RTS], [ASR])), clean=True)
        m, rep = self.check(self.program(([BUILD], [ASW], [RTS], [ASR]), refs=False), clean=True)
        self.assertEqual(m.accel["as#2"].refs, frozenset({"as#1"}))
        self.assertIn("unresolved TLAS instance reference", rep.meta["model.unsupported"])

    def test_sbt_written_by_transfer(self):
        def extra(dacc):
            def f(t):
                t.c("vkCmdUpdateBuffer", dst="buffer#70", offset=0, size=192)
                t.barrier([XFER], [TW], [RTS], dacc)
            return f
        self.check(self.program(([BUILD], [ASW], [RTS], [ASR]), extra=extra([SBT])), clean=True)
        self.check(self.program(([BUILD], [ASW], [RTS], [ASR]), extra=extra([SR])), clean=True)  # SHADER_READ holds SBT
        self.check(self.program(([BUILD], [ASW], [RTS], [ASR]), extra=extra([A2 + "UNIFORM_READ_BIT"])),
                   clean=False, res="buffer#70")

    def test_trace_indirect(self):
        def prog(stage):
            t = with_buffers(Trace(), "buffer#70", "buffer#50")
            t.layout([])
            t.c("vkCmdFillBuffer", dst="buffer#50", offset=0, size=12, data=1)
            t.barrier([XFER], [TW], stage, [ICR])
            t.pipeline("vkCreateRayTracingPipelinesKHR", "pipeline#1", ["VK_SHADER_STAGE_RAYGEN_BIT_KHR"],
                       "VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR")
            reg = {"buffer": "buffer#70", "offset": 0, "size": 32}
            t.c("vkCmdTraceRaysIndirectKHR", raygen=reg, miss=reg, hit=reg, callable=None,
                indirect={"buffer": "buffer#50", "offset": 0, "size": 12})
            return t.submit()
        self.check(prog([DI]), clean=True)
        self.check(prog([RTS]), clean=False, res="buffer#50")

    def test_storage_image_write_then_transition_without_src_access(self):
        # the raytracing* samples' pattern (setImageLayout GENERAL -> TRANSFER_SRC, srcAccess 0)
        def prog(sacc):
            t = Trace()
            t.buffer("buffer#70")
            t.r("vkCreateImage", image="image#1", format="VK_FORMAT_R8G8B8A8_UNORM", usage=[], extent=[8, 8, 1],
                mip_levels=1, array_layers=1, initial_layout="VK_IMAGE_LAYOUT_GENERAL")
            t.r("vkCreateImageView", view="view#1", image="image#1", format="VK_FORMAT_R8G8B8A8_UNORM")
            t.layout([(DT + "STORAGE_IMAGE", ["VK_SHADER_STAGE_RAYGEN_BIT_KHR"],
                       {"view": "view#1", "layout": "VK_IMAGE_LAYOUT_GENERAL"})])
            t.pipeline("vkCreateRayTracingPipelinesKHR", "pipeline#1", ["VK_SHADER_STAGE_RAYGEN_BIT_KHR"],
                       "VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR")
            t.trace_rays()
            t.c("vkCmdPipelineBarrier2", memory_barriers=[], buffer_barriers=[], image_barriers=[{
                "image": "image#1", "old_layout": "VK_IMAGE_LAYOUT_GENERAL",
                "new_layout": "VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL",
                "src_stage": [S2 + "ALL_COMMANDS_BIT"], "src_access": sacc,
                "dst_stage": [S2 + "ALL_COMMANDS_BIT"], "dst_access": []}])
            return t.submit()
        self.check(prog([SW]), clean=True)
        self.check(prog([]), clean=False, res="image#1")

    def test_indirect2_sbt_reported_unsupported(self):
        t = with_buffers(Trace(), "buffer#50")
        t.layout([])
        t.pipeline("vkCreateRayTracingPipelinesKHR", "pipeline#1", ["VK_SHADER_STAGE_RAYGEN_BIT_KHR"],
                   "VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR")
        t.c("vkCmdTraceRaysIndirect2KHR", indirect={"buffer": "buffer#50", "offset": 0, "size": 104},
            sbt_unknown=True)
        _, rep = self.check(t.submit(), clean=True)
        self.assertIn("timing.unmodelled", rep.codes())
        self.assertIn("shader binding table of vkCmdTraceRaysIndirect2KHR", rep.meta["model.unsupported"])


# ---------------------------------------------------------------- ray query in other stages
class RayQuery(Base):
    def program(self, dst_stage, stages):
        t = with_buffers(Trace(), "buffer#1", "buffer#2", "buffer#80", "buffer#81", "buffer#90", "buffer#91")
        t.accel("as#1", "buffer#1", "bottom").accel("as#2", "buffer#2", "top")
        t.layout([(DT + "ACCELERATION_STRUCTURE_KHR", stages, {"acceleration_structure": "as#2"})])
        t.build(Trace.blas("as#1"))
        t.barrier([BUILD], [ASW], [BUILD], [ASR])
        t.build(Trace.tlas("as#2"))
        tl = t.last_seq()
        t.barrier([BUILD], [ASW], dst_stage, [ASR])
        return t, tl

    def test_fragment_ray_query(self):
        for dst, clean in (([FS_], True), ([RTS], False), ([S2 + "ALL_GRAPHICS_BIT"], True)):
            t, tl = self.program(dst, ["VK_SHADER_STAGE_FRAGMENT_BIT"])
            t.pipeline("vkCreateGraphicsPipelines", "pipeline#1",
                       ["VK_SHADER_STAGE_VERTEX_BIT", "VK_SHADER_STAGE_FRAGMENT_BIT"],
                       "VK_PIPELINE_BIND_POINT_GRAPHICS")
            t.c("vkCmdBeginRendering", color=[], depth=None, stencil=None)
            t.c("vkCmdDraw", vertex_count=3, instance_count=1)
            t.c("vkCmdEndRendering")
            with self.subTest(dst=dst):
                self.check(t.submit([(tl, 0, ["as#1"])]), clean=clean, res="buffer#2")

    def test_compute_ray_query(self):
        for dst, clean in (([CS], True), ([FS_], False)):
            t, tl = self.program(dst, ["VK_SHADER_STAGE_COMPUTE_BIT"])
            t.pipeline("vkCreateComputePipelines", "pipeline#1", ["VK_SHADER_STAGE_COMPUTE_BIT"],
                       "VK_PIPELINE_BIND_POINT_COMPUTE")
            t.c("vkCmdDispatch", groups=[1, 1, 1])
            with self.subTest(dst=dst):
                self.check(t.submit([(tl, 0, ["as#1"])]), clean=clean, res="buffer#2")


# ---------------------------------------------------------------- mesh shading and compute
class MeshAndCompute(Base):
    def program(self, barrier, draw="vkCmdDrawMeshTasksIndirectEXT", **kw):
        t = with_buffers(Trace(), "buffer#40", "buffer#41")
        t.layout([(DT + "STORAGE_BUFFER", ["VK_SHADER_STAGE_COMPUTE_BIT"],
                   {"buffer": "buffer#40", "offset": 0, "range": "whole"})])
        t.pipeline("vkCreateComputePipelines", "pipeline#1", ["VK_SHADER_STAGE_COMPUTE_BIT"],
                   "VK_PIPELINE_BIND_POINT_COMPUTE")
        t.c("vkCmdDispatch", groups=[1, 1, 1])                # writes buffer#40 (draw arguments)
        if barrier:
            t.barrier(*barrier)
        t.pipeline("vkCreateGraphicsPipelines", "pipeline#2",
                   ["VK_SHADER_STAGE_TASK_BIT_EXT", "VK_SHADER_STAGE_MESH_BIT_EXT", "VK_SHADER_STAGE_FRAGMENT_BIT"],
                   "VK_PIPELINE_BIND_POINT_GRAPHICS", bind_set=False)
        t.c("vkCmdBeginRendering", color=[], depth=None, stencil=None)
        t.c(draw, buffer="buffer#40", offset=0, draw_count=1, stride=12, **kw)
        t.c("vkCmdEndRendering")
        return t.submit()

    def test_compute_writes_mesh_indirect(self):
        m, _ = self.check(self.program(([CS], [SW], [DI], [ICR])), clean=True)
        self.assertEqual([d.fn for d in m.draws], ["vkCmdDispatch", "vkCmdDrawMeshTasksIndirectEXT"])

    def test_mutant_compute_mesh_indirect_missing_barrier(self):
        self.check(self.program(None), clean=False, res="buffer#40")

    def test_mutant_compute_mesh_indirect_task_stage(self):
        # TASK_SHADER is logically after DRAW_INDIRECT: it does not hold the indirect read
        self.check(self.program(([CS], [SW], [TASK], [ICR])), clean=False, res="buffer#40")

    def test_indirect_count(self):
        def prog(barrier):
            t = with_buffers(Trace(), "buffer#40", "buffer#41")
            t.layout([(DT + "STORAGE_BUFFER", ["VK_SHADER_STAGE_COMPUTE_BIT"],
                       {"buffer": "buffer#41", "offset": 0, "range": "whole"})])
            t.pipeline("vkCreateComputePipelines", "pipeline#1", ["VK_SHADER_STAGE_COMPUTE_BIT"],
                       "VK_PIPELINE_BIND_POINT_COMPUTE")
            t.c("vkCmdDispatch", groups=[1, 1, 1])            # writes the count buffer
            if barrier:
                t.barrier([CS], [SW], [DI], [ICR])
            t.pipeline("vkCreateGraphicsPipelines", "pipeline#2", ["VK_SHADER_STAGE_MESH_BIT_EXT"],
                       "VK_PIPELINE_BIND_POINT_GRAPHICS", bind_set=False)
            t.c("vkCmdBeginRendering", color=[], depth=None, stencil=None)
            t.c("vkCmdDrawMeshTasksIndirectCountEXT", buffer="buffer#40", offset=0, count_buffer="buffer#41",
                count_offset=0, max_draw_count=4, stride=12)
            t.c("vkCmdEndRendering")
            return t.submit()
        self.check(prog(True), clean=True)
        self.check(prog(False), clean=False, res="buffer#41")

    def test_mesh_shader_storage_read(self):
        def prog(dst):
            t = with_buffers(Trace(), "buffer#40")
            t.layout([(DT + "STORAGE_BUFFER", ["VK_SHADER_STAGE_COMPUTE_BIT", "VK_SHADER_STAGE_MESH_BIT_EXT"],
                       {"buffer": "buffer#40", "offset": 0, "range": "whole"})])
            t.pipeline("vkCreateComputePipelines", "pipeline#1", ["VK_SHADER_STAGE_COMPUTE_BIT"],
                       "VK_PIPELINE_BIND_POINT_COMPUTE")
            t.c("vkCmdDispatch", groups=[1, 1, 1])
            # no SPIR-V here: the mesh draw's storage access is taken as a write (fail closed)
            t.barrier([CS], [SW], dst, [SR, SW])
            t.pipeline("vkCreateGraphicsPipelines", "pipeline#2",
                       ["VK_SHADER_STAGE_MESH_BIT_EXT", "VK_SHADER_STAGE_FRAGMENT_BIT"],
                       "VK_PIPELINE_BIND_POINT_GRAPHICS")
            t.c("vkCmdBeginRendering", color=[], depth=None, stencil=None)
            t.c("vkCmdDrawMeshTasksEXT", groups=[1, 1, 1])
            t.c("vkCmdEndRendering")
            return t.submit()
        self.check(prog([MESH]), clean=True)
        self.check(prog([S2 + "PRE_RASTERIZATION_SHADERS_BIT"]), clean=True)
        # TASK's execution scope holds the later MESH stage, its access scope does not
        self.check(prog([TASK]), clean=False, res="buffer#40")
        self.check(prog([TASK, MESH]), clean=True)
        self.check(prog([S2 + "VERTEX_SHADER_BIT"]), clean=False, res="buffer#40")

    def test_mesh_draw_reads_no_vertex_buffers(self):
        t = with_buffers(Trace(), "buffer#40")
        t.layout([])
        t.c("vkCmdBindVertexBuffers", first_binding=0, buffers=["buffer#40"], offsets=[0])
        t.c("vkCmdCopyBuffer", src="buffer#40", dst="buffer#40", regions=[])
        t.pipeline("vkCreateGraphicsPipelines", "pipeline#2", ["VK_SHADER_STAGE_MESH_BIT_EXT"],
                   "VK_PIPELINE_BIND_POINT_GRAPHICS")
        t.c("vkCmdBeginRendering", color=[], depth=None, stencil=None)
        t.c("vkCmdDrawMeshTasksEXT", groups=[1, 1, 1])
        t.c("vkCmdEndRendering")
        self.check(t.submit(), clean=True)

    def test_dispatch_indirect_and_base(self):
        def prog(barrier):
            t = with_buffers(Trace(), "buffer#40")
            t.layout([(DT + "STORAGE_BUFFER", ["VK_SHADER_STAGE_COMPUTE_BIT"],
                       {"buffer": "buffer#40", "offset": 0, "range": "whole"})])
            t.pipeline("vkCreateComputePipelines", "pipeline#1", ["VK_SHADER_STAGE_COMPUTE_BIT"],
                       "VK_PIPELINE_BIND_POINT_COMPUTE")
            t.c("vkCmdDispatchBase", base=[0, 0, 0], groups=[1, 1, 1])
            if barrier:
                t.barrier([CS], [SW], [DI, CS], [ICR, SR, SW])
            t.c("vkCmdDispatchIndirect", buffer="buffer#40", offset=0)
            return t.submit()
        self.check(prog(True), clean=True)
        self.check(prog(False), clean=False, res="buffer#40")


class Unsupported(unittest.TestCase):
    def test_stubbed_command_is_reported(self):
        t = Trace()
        t.c("vkCmdSetDescriptorBufferOffsetsEXT", warning="unimplemented")
        t.c("vkCmdBeginQuery", warning="unimplemented")       # no memory effect: not a warning
        m = t.submit().load()
        rep = timing.check(m.frame, Report())
        self.assertEqual([d.code for d in rep.items], ["timing.unmodelled-command"])
        self.assertEqual(rep.items[0].severity, "warning")
        self.assertIn("vkCmdSetDescriptorBufferOffsetsEXT", rep.items[0].message)
        self.assertIn("@seq 4", rep.items[0].where)                 # its position in the trace
        self.assertIn("command_buffer#1", rep.items[0].where)
        self.assertEqual(set(rep.meta["model.unsupported"]), {"vkCmdSetDescriptorBufferOffsetsEXT"})

    def test_unsubmitted_stub_is_not_reported(self):
        t = Trace()
        t.c("vkCmdClearAttachments", warning="unimplemented")
        rep = timing.check(t.load().frame, Report())               # never submitted
        self.assertEqual(rep.items, [])

    def test_stub_without_command_buffer_is_reported(self):
        # traces from a driver that recorded stubs without their command buffer
        t = Trace()
        t.r("vkCmdClearAttachments", warning="unimplemented")
        t.r("vkDebugMarkerSetObjectNameEXT", warning="unimplemented")
        t.r("vkQueueBindSparse", warning="unimplemented")
        t.r("vkGetQueryPoolResults", warning="unimplemented")
        rep = timing.check(t.submit().load().frame, Report())
        self.assertEqual(sorted(d.message.split("'")[1] for d in rep.items),
                         ["vkCmdClearAttachments", "vkQueueBindSparse"])
        self.assertEqual({d.code for d in rep.items}, {"timing.unmodelled-command"})

    def test_old_trace_with_stubbed_ray_tracing_is_reported(self):
        # traces recorded before the driver modelled ray tracing: create calls and
        # commands are stubs with no fields
        t = Trace()
        t.r("vkCreateAccelerationStructureKHR", warning="unimplemented")
        t.r("vkCreateRayTracingPipelinesKHR", warning="unimplemented")
        t.r("vkCmdBuildAccelerationStructuresKHR", warning="unimplemented")
        t.r("vkCmdTraceRaysKHR", warning="unimplemented")
        rep = timing.check(t.submit().load().frame, Report())
        self.assertEqual(sorted(d.message.split("'")[1] for d in rep.items),
                         ["vkCmdBuildAccelerationStructuresKHR", "vkCmdTraceRaysKHR"])
        self.assertEqual({d.code for d in rep.items}, {"timing.unmodelled-command"})

    def test_unresolved_device_address(self):
        t = with_buffers(Trace(), "buffer#1", "buffer#80")
        t.accel("as#1", "buffer#1", "bottom")
        b = Trace.blas("as#1")
        b["scratch"] = {"address": 123, "unresolved": True, "size": 1280}
        t.build(b)
        rep = timing.check(t.submit().load().frame, Report())
        self.assertIn("unresolved device address (scratch)", rep.meta["model.unsupported"])
        self.assertIn("timing.unmodelled", rep.codes())

    def test_host_build_is_a_host_write(self):
        t = with_buffers(Trace(), "buffer#1", "buffer#80", "buffer#90")
        t.accel("as#1", "buffer#1", "bottom")
        t.build(Trace.blas("as#1"))
        t.submit()
        t.r("vkBuildAccelerationStructuresKHR", deferred=False, builds=[Trace.blas("as#1")])
        rep = timing.check(t.load().frame, Report())
        self.assertIn("timing.cross-queue-race", rep.codes())   # no fence wait before the host rebuild


# ---------------------------------------------------------------- Theorem 1 with the new stages
NEW_STAGES = ["TASK_SHADER", "MESH_SHADER", "RAY_TRACING_SHADER", "ACCELERATION_STRUCTURE_BUILD",
              "ACCELERATION_STRUCTURE_COPY"]
NEW_FLAGS = {
    "TASK_SHADER": (["UNIFORM_READ", "SHADER_STORAGE_READ", "ACCELERATION_STRUCTURE_READ"], ["SHADER_STORAGE_WRITE"]),
    "MESH_SHADER": (["UNIFORM_READ", "SHADER_STORAGE_READ"], ["SHADER_STORAGE_WRITE"]),
    "RAY_TRACING_SHADER": (["SHADER_BINDING_TABLE_READ", "ACCELERATION_STRUCTURE_READ", "SHADER_STORAGE_READ"],
                           ["SHADER_STORAGE_WRITE"]),
    "ACCELERATION_STRUCTURE_BUILD": (["ACCELERATION_STRUCTURE_READ", "SHADER_READ", "INDIRECT_COMMAND_READ"],
                                     ["ACCELERATION_STRUCTURE_WRITE"]),
    "ACCELERATION_STRUCTURE_COPY": (["ACCELERATION_STRUCTURE_READ", "TRANSFER_READ"],
                                    ["ACCELERATION_STRUCTURE_WRITE", "TRANSFER_WRITE"]),
}
NEW_BITS = ["ACCELERATION_STRUCTURE_READ", "ACCELERATION_STRUCTURE_WRITE", "SHADER_BINDING_TABLE_READ"]


class RandomFramesRtMesh(unittest.TestCase):
    SEEDS = int(os.environ.get("COVENANT_REF_SEEDS", "150"))

    def run_opts(self, exact=True, **opts):
        stages = timing_gen.STAGES + NEW_STAGES
        acc = timing_gen.ACCESS_STAGES + NEW_STAGES
        flags = dict(timing_gen.FLAGS, **NEW_FLAGS)
        bits = timing_gen.ACCESS_MASK_BITS + NEW_BITS
        with mock.patch.object(timing_gen, "STAGES", stages), mock.patch.object(timing_gen, "ACCESS_STAGES", acc), \
                mock.patch.object(timing_gen, "FLAGS", flags), mock.patch.object(timing_gen, "ACCESS_MASK_BITS", bits):
            for seed in range(self.SEEDS):
                fr = timing_gen.random_frame(10_000 + seed, **opts)
                spec = spec_pairs(fr)
                hz = []
                timing.check(fr, Report(), hazards=hz, exhaustive=True, window=False)
                full = {frozenset((a, b)) for a, b, _ in hz}
                hz = []
                timing.check(fr, Report(), hazards=hz, window=False)
                ep = {frozenset((a, b)) for a, b, _ in hz}
                with self.subTest(seed=seed, opts=opts):
                    self.assertFalse(spec - full, "Theorem 1: race the checker misses")
                    if exact:
                        self.assertFalse(full - spec, "checker stricter than the reference")
                    self.assertFalse(ep - full, "epoch pair not reported pairwise")
                    flagged = {k for p in ep for k in p}
                    self.assertFalse([p for p in spec if not p & flagged], "criterion E")

    def test_default(self):
        self.run_opts()

    def test_same_command(self):
        self.run_opts(cmd=True)

    def test_regions(self):
        self.run_opts(exact=False, regions=True)


# ---------------------------------------------------------------- events (split barriers)
class Events(Base):
    """vkCmdSetEvent*/vkCmdWaitEvents*: the wait's first synchronization scope is
    what precedes the latest set of the event (synchronization.adoc, Events)."""

    def program(self, order, sync2=False, wait_masks=(CS, SW, CS, (SR, SW))):
        t = with_buffers(Trace(), "buffer#40")
        t.layout([(DT + "STORAGE_BUFFER", ["VK_SHADER_STAGE_COMPUTE_BIT"],
                   {"buffer": "buffer#40", "offset": 0, "range": "whole"})])
        t.pipeline("vkCreateComputePipelines", "pipeline#1", ["VK_SHADER_STAGE_COMPUTE_BIT"],
                   "VK_PIPELINE_BIND_POINT_COMPUTE")
        ss, sa, ds, da = wait_masks
        mb = {"src_stage": [ss], "src_access": [sa], "dst_stage": [ds], "dst_access": list(da)}
        dep = {"memory_barriers": [mb], "buffer_barriers": [], "image_barriers": []}
        for step in order:
            if step == "write":
                t.c("vkCmdDispatch", groups=[1, 1, 1])        # writes buffer#40 (no SPIR-V: compute writes)
            elif step == "set":
                if sync2:
                    t.c("vkCmdSetEvent2", event="event#1", dependency=dep)
                else:
                    t.c("vkCmdSetEvent", event="event#1", stage=[ss])
            elif step == "reset":
                t.c("vkCmdResetEvent", event="event#1", stage=[ss])
            elif step == "wait":
                if sync2:
                    t.c("vkCmdWaitEvents2", events=["event#1"], dependencies=[dep])
                else:
                    t.c("vkCmdWaitEvents", events=["event#1"], memory_barriers=[mb], buffer_barriers=[],
                        image_barriers=[], src_stage=[ss], dst_stage=[ds])
        return t.submit()

    def test_set_then_wait_orders(self):
        for sync2 in (False, True):
            with self.subTest(sync2=sync2):
                self.check(self.program(["write", "set", "wait", "write"], sync2), clean=True)

    def test_access_between_set_and_wait_is_not_ordered(self):
        for sync2 in (False, True):
            with self.subTest(sync2=sync2):
                self.check(self.program(["set", "write", "wait", "write"], sync2), clean=False, res="buffer#40")

    def test_wait_without_set_orders_nothing(self):
        self.check(self.program(["write", "wait", "write"]), clean=False, res="buffer#40")

    def test_reset_cancels_set(self):
        self.check(self.program(["write", "set", "reset", "wait", "write"]), clean=False, res="buffer#40")
        self.check(self.program(["write", "set", "reset", "set", "wait", "write"]), clean=True)

    def test_wait_masks_still_apply(self):
        self.check(self.program(["write", "set", "wait", "write"], wait_masks=(CS, SR, CS, (SR, SW))),
                   clean=False, res="buffer#40")              # srcAccess lacks the write

    def test_event_wait_layout_transition(self):
        def prog(order):
            t = Trace()
            t.r("vkCreateImage", image="image#1", format="VK_FORMAT_R8G8B8A8_UNORM", usage=[], extent=[8, 8, 1],
                mip_levels=1, array_layers=1, initial_layout="VK_IMAGE_LAYOUT_GENERAL")
            t.r("vkCreateImageView", view="view#1", image="image#1", format="VK_FORMAT_R8G8B8A8_UNORM")
            t.layout([(DT + "STORAGE_IMAGE", ["VK_SHADER_STAGE_COMPUTE_BIT"],
                       {"view": "view#1", "layout": "VK_IMAGE_LAYOUT_GENERAL"})])
            t.pipeline("vkCreateComputePipelines", "pipeline#1", ["VK_SHADER_STAGE_COMPUTE_BIT"],
                       "VK_PIPELINE_BIND_POINT_COMPUTE")
            ib = {"image": "image#1", "old_layout": "VK_IMAGE_LAYOUT_GENERAL",
                  "new_layout": "VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL",
                  "src_stage": [CS], "src_access": [SW], "dst_stage": [XFER], "dst_access": [A2 + "TRANSFER_READ_BIT"]}
            for step in order:
                if step == "write":
                    t.c("vkCmdDispatch", groups=[1, 1, 1])
                elif step == "set":
                    t.c("vkCmdSetEvent", event="event#1", stage=[CS])
                elif step == "wait":
                    t.c("vkCmdWaitEvents", events=["event#1"], memory_barriers=[], buffer_barriers=[],
                        image_barriers=[ib], src_stage=[CS], dst_stage=[XFER])
            return t.submit()
        self.check(prog(["write", "set", "wait"]), clean=True)
        self.check(prog(["set", "write", "wait"]), clean=False, res="image#1")

    def test_host_set_event_is_reported(self):
        t = self.program(["write", "wait", "write"])
        t.recs.insert(1, dict(seq=0, fn="vkSetEvent", event="event#1"))
        rep = timing.check(t.load().frame, Report())
        self.assertIn("event signalled by the host (vkSetEvent)", rep.meta["model.unsupported"])


class RandomEvents(unittest.TestCase):
    """Theorem 1 with events: random single-queue frames of accesses, barriers,
    event sets/resets and event waits; the pairwise checker equals the
    reference and the epoch checker meets criterion E."""
    SEEDS = int(os.environ.get("COVENANT_REF_SEEDS", "150"))

    @staticmethod
    def frame(seed):
        import random
        from covenant.timing import Barrier, Frame, Signal, Use, Wait
        rng = random.Random(seed)
        st = ["COMPUTE_SHADER", "TRANSFER", "FRAGMENT_SHADER", "RAY_TRACING_SHADER"]
        fl = {"COMPUTE_SHADER": ("SHADER_STORAGE_READ", "SHADER_STORAGE_WRITE"),
              "FRAGMENT_SHADER": ("SHADER_STORAGE_READ", "SHADER_STORAGE_WRITE"),
              "RAY_TRACING_SHADER": ("SHADER_STORAGE_READ", "SHADER_STORAGE_WRITE"),
              "TRANSFER": ("TRANSFER_READ", "TRANSFER_WRITE")}
        acc = ["SHADER_READ", "SHADER_WRITE", "TRANSFER_READ", "TRANSFER_WRITE", "MEMORY_READ", "MEMORY_WRITE"]
        F = frozenset
        q, host = [], [Signal("sub", "host#0")]
        q.append(Wait("sub", "q#0"))
        for i in range(rng.randrange(6, 30)):
            k = rng.random()
            w = f"q#{len(q)}"
            if k < 0.4:
                s = rng.choice(st)
                kind = rng.choice(("read", "write"))
                q.append(Use(rng.choice(("r0", "r1")), kind, None, w, s, fl[s][kind == "write"]))
            elif k < 0.55:
                q.append(Barrier(None, None, None, w, F(), F(), F(), F(),
                                 event_mark=(rng.choice(("set", "set", "reset")), rng.choice(("e0", "e1")))))
            else:
                b = Barrier(rng.choice((None, "r0", "r1")), None, None, w,
                            F(rng.sample(st + ["ALL_COMMANDS"], rng.randrange(1, 3))),
                            F(rng.sample(acc, rng.randrange(0, 3))),
                            F(rng.sample(st + ["ALL_COMMANDS"], rng.randrange(1, 3))),
                            F(rng.sample(acc, rng.randrange(0, 3))))
                if rng.random() < 0.6:
                    b.event = rng.choice(("e0", "e1"))
                if b.res is not None and rng.random() < 0.2:
                    b.old, b.new = "general", "general2"
                q.append(b)
        return Frame(queues={"host": host, "q0": q})

    def test_events(self):
        n_ev = 0
        for seed in range(self.SEEDS):
            fr = self.frame(seed)
            n_ev += sum(1 for o in fr.queues["q0"] if getattr(o, "event", None))
            spec = spec_pairs(fr)
            full = checker_pairs(fr)
            hz = []
            timing.check(fr, Report(), hazards=hz, window=False)
            ep = {frozenset((a, b)) for a, b, _ in hz}
            with self.subTest(seed=seed):
                self.assertEqual(full, spec)
                self.assertFalse(ep - full, "epoch pair not reported pairwise")
                flagged = {k for p in ep for k in p}
                self.assertFalse([p for p in spec if not p & flagged], "criterion E")
        self.assertGreater(n_ev, self.SEEDS)


class DriverStubs(unittest.TestCase):
    """model_driver/gen.py: every vkCmd* stub records its command buffer (so its
    position in the submitted work is known), and every one that is not
    harmless dynamic state carries a warning the loader reports."""

    def test_every_command_stub_keeps_its_command_buffer(self):
        import re
        import subprocess
        md = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_driver")
        vkinc = os.environ.get("COVENANT_VKINC")       # Vulkan-Headers include dir (README)
        cands = ([os.path.join(vkinc, "vulkan", "vulkan_core.h")] if vkinc else []) + ["/usr/include/vulkan/vulkan_core.h"]
        hdr = next((h for h in cands if os.path.exists(h)), None)
        if hdr is None or not os.path.exists(os.path.join(md, "gen.py")):
            self.skipTest("needs Vulkan headers and model_driver/gen.py")
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["python3", os.path.join(md, "gen.py"), hdr, os.path.join(md, "covvk.cpp"),
                            os.path.join(d, "e.inc"), os.path.join(d, "s.inc")], check=True, capture_output=True)
            src = open(os.path.join(d, "s.inc")).read()
        stubs = re.findall(r"VKAPI_CALL (vkCmd\w+)\((VkCommandBuffer \w+)[^)]*\) \{\n(.*?)\n\}", src, re.S)
        self.assertGreater(len(stubs), 50)
        for name, _, body in stubs:
            if "covvk_" not in body:
                continue                # an alias forwarding to a hand-written entry point
            self.assertRegex(body, r'covvk_(unimplemented_cmd|record_cmd)\("%s", \(uint64_t\)\w+\)' % name, name)
            if "covvk_record_cmd" in body:
                self.assertTrue(name.startswith("vkCmdSet") and "Event" not in name, name)
        for ev in ("vkCmdSetEvent", "vkCmdWaitEvents", "vkCmdResetEvent", "vkCmdSetEvent2", "vkCmdWaitEvents2"):
            self.assertNotIn(f'covvk_unimplemented_cmd("{ev}"', src)   # hand-written: recorded with arguments
            self.assertNotIn(f'covvk_record_cmd("{ev}"', src)


class DriverEvents(unittest.TestCase):
    """End to end: the model driver records vkCmdSetEvent2/vkCmdWaitEvents2 and
    the checker orders (or does not order) through them."""

    def run_triangle(self, *flags):
        import shutil
        import subprocess
        build = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_driver", "build")
        exe = os.path.join(build, "triangle")
        if not os.path.exists(exe):
            self.skipTest("model_driver/build/triangle not built (make example)")
        d = tempfile.mkdtemp(prefix="covvk-ev-")
        try:
            env = dict(os.environ, COVENANT_TRACE=os.path.join(d, "t.jsonl"), COVENANT_SPIRV_DIR=d,
                       LD_LIBRARY_PATH=build)
            args = [os.path.join(build, "tri.vert.spv"), os.path.join(build, "tri.frag.spv"), *flags]
            try:
                subprocess.run([exe] + args, env=env, check=True, capture_output=True, timeout=60)
            except PermissionError:
                subprocess.run(["/lib64/ld-linux-x86-64.so.2", exe] + args, env=env, check=True,
                               capture_output=True, timeout=60)
            return load_trace(os.path.join(d, "t.jsonl"), spirv_dir=d)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_events_order_the_present_transition(self):
        m = self.run_triangle("--events")
        self.assertTrue(any(r["fn"] == "vkCmdWaitEvents2" for r in m.records))
        self.assertEqual(timing.check(m.frame).items, [])

    def test_event_set_too_early(self):
        rep = timing.check(self.run_triangle("--bug-event-early").frame)
        self.assertEqual({d.code for d in rep.items}, {"timing.barrier-scope"})
        self.assertTrue(all("vkCmdWaitEvents2" in d.where for d in rep.items))


# ---------------------------------------------------------------- queue family ownership per range
class OwnershipPerRange(unittest.TestCase):
    """host_trace.Ownership (interval maps) equals timing_ref.ownership_ref
    (element by element) on random operation sequences, and the loader reports
    ownership per byte range (vuk sub-allocates one VkBuffer)."""

    def test_random_against_reference(self):
        import random
        from covenant.host_trace import Ownership
        U = 24
        for seed in range(int(os.environ.get("COVENANT_REF_SEEDS", "150")) * 4):
            rng = random.Random(seed)
            ops = []
            for _ in range(rng.randrange(1, 25)):
                k = rng.choice(("write", "read", "read", "release", "acquire"))
                if rng.random() < 0.2:
                    reg = None
                else:
                    lo = rng.randrange(0, U - 1)
                    reg = ((lo, rng.randrange(lo + 1, U + 1)),)
                    if rng.random() < 0.3:
                        lo2 = rng.randrange(reg[0][1], U + 1)
                        if lo2 < U:
                            reg = reg + ((lo2, rng.randrange(lo2 + 1, U + 1)),)
                res = rng.choice(("a", "b"))
                if k in ("write", "read"):
                    ops.append((k, res, reg, rng.randrange(2)))
                else:
                    sf = rng.randrange(2)
                    ops.append((k, res, reg, sf, 1 - sf))
            ref = timing_ref.ownership_ref(ops, U)
            own = Ownership()
            for op, want in zip(ops, ref):
                reg = op[2] if op[2] is not None else ((0, U),)  # the universe: whole resource
                if op[0] in ("write", "read"):
                    bad = own.use(op[1], reg, op[3], op[0])
                    got = {x for lo, hi, _ in bad for x in range(lo, hi)} if op[0] == "read" else None
                elif op[0] == "release":
                    own.release(op[1], reg, op[3], op[4], "")
                    got = None
                else:
                    got = own.acquire(op[1], reg, op[3], op[4])
                with self.subTest(seed=seed, op=op):
                    self.assertEqual(got, want)

    def program(self, read_off, write_off=0, release=False):
        """Family 1 fills [write_off, +16) of one exclusive buffer; family 0 then
        copies [read_off, +16) of it elsewhere."""
        t = Trace()
        t.r("vkGetDeviceQueue", queue="queue#2", family=1, index=0)
        t.r("vkCreateBuffer", buffer="buffer#1", size=1 << 20, usage=[], sharing="VK_SHARING_MODE_EXCLUSIVE")
        t.buffer("buffer#2")
        t.r("vkAllocateCommandBuffers", pool="command_pool#2", command_buffers=["command_buffer#2"])
        t.r("vkBeginCommandBuffer", cb="command_buffer#2", flags=[])
        t.r("vkCmdFillBuffer", cb="command_buffer#2", dst="buffer#1", offset=write_off, size=16, data=0)
        bb = {"buffer": "buffer#1", "offset": write_off, "size": 16, "src_family": 1, "dst_family": 0,
              "src_stage": [XFER], "src_access": [TW], "dst_stage": [], "dst_access": []}
        if release:
            t.r("vkCmdPipelineBarrier2", cb="command_buffer#2", buffer_barriers=[bb], image_barriers=[],
                memory_barriers=[])
        t.r("vkEndCommandBuffer", cb="command_buffer#2")
        t.r("vkCreateSemaphore", semaphore="semaphore#1")
        t.r("vkQueueSubmit", queue="queue#2", fence=None, submits=[
            {"wait": [], "command_buffers": ["command_buffer#2"], "signal": [{"semaphore": "semaphore#1"}]}])
        if release:
            t.c("vkCmdPipelineBarrier2", buffer_barriers=[dict(bb, src_stage=[], src_access=[], dst_stage=[XFER],
                                                               dst_access=[A2 + "TRANSFER_READ_BIT"])],
                image_barriers=[], memory_barriers=[])
        t.c("vkCmdCopyBuffer", src="buffer#1", dst="buffer#2",
            regions=[{"src_offset": read_off, "dst_offset": 0, "size": 16}])
        t.r("vkEndCommandBuffer", cb=t.cb)
        t.r("vkQueueSubmit", queue="queue#1", fence=None, submits=[
            {"wait": [{"semaphore": "semaphore#1", "stage": [S2 + "ALL_COMMANDS_BIT"]}],
             "command_buffers": [t.cb], "signal": []}])
        return [d for d in timing.check(t.load().frame, Report()).items if d.code.startswith("timing.ownership")]

    def test_disjoint_suballocations_are_not_reported(self):
        self.assertEqual(self.program(read_off=128), [])          # the vuk false alarm

    def test_overlapping_read_is_reported(self):
        got = self.program(read_off=8)
        self.assertEqual([d.code for d in got], ["timing.ownership-missing"])
        self.assertIn("[8, 16)", got[0].message)

    def test_release_acquire_of_the_range(self):
        self.assertEqual(self.program(read_off=0, release=True), [])

    def test_write_takes_ownership(self):
        from covenant.host_trace import Ownership
        o = Ownership()
        o.use("b", ((0, 16),), 1, "write")
        o.use("b", ((0, 16),), 0, "write")           # family 0 overwrites: implicit acquire
        self.assertEqual(o.use("b", ((0, 16),), 0, "read"), [])
        self.assertEqual(o.use("b", ((0, 32),), 1, "read"), [(0, 16, 0)])


# ---------------------------------------------------------------- R-LT: transitions ordered among themselves
class LayoutTransitionOrder(unittest.TestCase):
    """synchronization.adoc:261-269: image layout transitions on one queue execute in
    submission order relative to each other (render passes included), but nothing
    else may rely on that order. The UI-overlay pattern of the Khronos Vulkan samples:
    copy -> barrier to PRESENT with dst BOTTOM_OF_PIPE -> render pass whose
    EXTERNAL dependency has src BOTTOM_OF_PIPE / access 0, initial layout PRESENT."""
    F = frozenset
    BOP, CA = F({"BOTTOM_OF_PIPE"}), F({"COLOR_ATTACHMENT_OUTPUT"})

    def ops(self, load=True, queue_b="gfx"):
        from covenant.timing import Barrier, Use
        F = frozenset
        a = [Use("sc", "write", None, "copy", "TRANSFER", "TRANSFER_WRITE"),
             Barrier("sc", "transfer_dst", "present", "to present", F({"TRANSFER"}), F({"TRANSFER_WRITE"}),
                     self.BOP, F())]
        b = [Barrier("sc", "present", "color", "render pass initial transition", self.BOP, F(),
                     self.CA, F({"COLOR_ATTACHMENT_READ", "COLOR_ATTACHMENT_WRITE"}))]
        if load:
            b.append(Use("sc", "read", None, "load op LOAD", "COLOR_ATTACHMENT_OUTPUT", "COLOR_ATTACHMENT_READ"))
        return a, b

    def frame(self, a, b, two_queues=False):
        from covenant.timing import Frame
        q = {"gfx": a + b} if not two_queues else {"gfx": a, "gfx2": b}
        return Frame(queues=q, initial_layouts={"sc": "transfer_dst"})

    def items(self, fr):
        return timing.check(fr, Report(), window=False).items

    def test_transition_pair_on_one_queue_is_not_a_race(self):
        a, b = self.ops(load=False)
        fr = self.frame(a[1:], b)                       # no copy: only the two transitions
        self.assertEqual(self.items(fr), [])
        self.assertEqual(spec_pairs(fr), set())
        self.assertEqual(checker_pairs(fr), set())

    def test_write_before_the_first_transition_still_races_the_second(self):
        # the copy is made available by the first barrier, but the render pass's
        # transition is ordered after it only through the transition order, which
        # orders the transitions alone: the copy -> second transition WAW remains
        fr = self.frame(*self.ops(load=False))
        its = self.items(fr)
        self.assertEqual(len(its), 1, its)
        self.assertEqual(its[0].where, "render pass initial transition")
        self.assertEqual(its[0].notes[0], "earlier access: copy")
        self.assertEqual(checker_pairs(fr), spec_pairs(fr))

    def test_not_transitive_the_load_still_races(self):
        # the load is ordered after the render pass's transition, but not after the
        # copy or the first transition: it must still be reported
        fr = self.frame(*self.ops(load=True))
        its = self.items(fr)
        self.assertTrue(its)
        self.assertFalse([d for d in its if d.where == "render pass initial transition"
                          and "earlier access: to present" in d.notes], its)
        self.assertTrue(any("load op LOAD" in d.where for d in its), its)
        self.assertEqual(checker_pairs(fr), spec_pairs(fr))
        self.assertTrue(spec_pairs(fr))

    def test_transitions_on_two_queues_still_race(self):
        a, b = self.ops(load=False)
        fr = self.frame(a, b, two_queues=True)
        self.assertTrue(self.items(fr))
        self.assertEqual(checker_pairs(fr), spec_pairs(fr))

    def test_earlier_transition_stays_live_for_later_accesses(self):
        # three transitions then a read: the read is checked against the first
        # transition too (nothing is ordered through the transition order)
        from covenant.timing import Barrier, Use
        F = frozenset
        a, b = self.ops(load=False)
        b.append(Barrier("sc", "color", "present", "final transition", self.CA, F({"COLOR_ATTACHMENT_WRITE"}),
                         self.BOP, F()))
        b.append(Use("sc", "read", None, "sample", "FRAGMENT_SHADER", "SHADER_SAMPLED_READ"))
        fr = self.frame(a, b)
        self.assertEqual(checker_pairs(fr), spec_pairs(fr))
        self.assertTrue(any(d.where == "sample" for d in self.items(fr)))

    def test_random_frames_agree_with_reference(self):
        # transitions are frequent in these frames; exact agreement with R-LT
        for seed in range(200):
            fr = timing_gen.random_frame(77_000 + seed, n_ops=30)
            with self.subTest(seed=seed):
                self.assertEqual(checker_pairs(fr), spec_pairs(fr))


if __name__ == "__main__":
    unittest.main()
