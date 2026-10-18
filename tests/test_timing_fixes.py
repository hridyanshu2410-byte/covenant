"""Timing-checker precision on synthetic model-driver records: subresource
granularity, dynamic offsets, matched ownership acquires, timeline initial
values, one barrier per subpass dependency, attachment load/store operations,
per-aspect layout matching (bench case 63), WAR through another resource's
barrier and same-command accesses. Each frame is also checked against the
reference semantics (covenant/timing_ref.py)."""
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from covenant import timing, timing_ref
from covenant.diagnostics import Report
from covenant.host_trace import _Loader, load_records
from covenant.timing import Barrier, Frame, Use

F = frozenset
TOP, BOT = ["VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT"], ["VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT"]
XFER = ["VK_PIPELINE_STAGE_TRANSFER_BIT"]
COMP = ["VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT"]


class Trace:
    def __init__(self):
        self.recs, self.seq = [], 0
        self.add("vkGetDeviceQueue", queue="queue#1", family=0, index=0)
        self.add("vkGetDeviceQueue", queue="queue#2", family=1, index=0)

    def add(self, fn, **kw):
        self.seq += 1
        r = dict(seq=self.seq, fn=fn, **kw)
        self.recs.append(r)
        return r

    def image(self, h, layers=1, mips=1, fmt="VK_FORMAT_R8G8B8A8_UNORM", sharing="VK_SHARING_MODE_CONCURRENT"):
        self.add("vkCreateImage", image=h, format=fmt, usage=[], extent=[4, 4, 1], mip_levels=mips,
                 array_layers=layers, sharing=sharing, initial_layout="VK_IMAGE_LAYOUT_UNDEFINED")

    def buffer(self, h, size=1024, sharing="VK_SHARING_MODE_CONCURRENT"):
        self.add("vkCreateBuffer", buffer=h, size=size, usage=[], sharing=sharing)

    def submit(self, cmds, queue="queue#1", cb=None, signal=(), wait=()):
        cb = cb or f"command_buffer#{self.seq}"
        self.add("vkAllocateCommandBuffers", pool="pool#1", command_buffers=[cb])
        self.add("vkBeginCommandBuffer", cb=cb)
        for fn, kw in cmds:
            self.add(fn, cb=cb, **kw)
        self.add("vkQueueSubmit", queue=queue, fence=None,
                 submits=[{"wait": list(wait), "command_buffers": [cb], "signal": list(signal)}])

    def model(self):
        return _Loader(self.recs).run()


def codes(frame):
    return sorted({d.code for d in timing.check(frame, Report()).items})


def agrees(test, frame):
    hz = []
    timing.check(frame, Report(), hazards=hz, exhaustive=True)
    test.assertEqual({frozenset((a, b)) for a, b, _ in hz},
                     {frozenset((a.key, b.key)) for a, b in timing_ref.Reference(frame, False).races()})


def copy_to_layer(layer, mip=0):
    return ("vkCmdCopyBufferToImage", dict(src="buffer#1", dst="image#1",
                                           dst_layout="VK_IMAGE_LAYOUT_GENERAL",
                                           regions=[{"buffer_offset": 0, "image": {
                                               "aspect": ["VK_IMAGE_ASPECT_COLOR_BIT"], "mip": mip,
                                               "base_layer": layer, "layer_count": 1}}]))


def fill(off, size, buf="buffer#2"):
    return ("vkCmdFillBuffer", dict(dst=buf, offset=off, size=size))


class Granularity(unittest.TestCase):
    """Array layers / aspects of images, byte ranges of buffers."""

    def frame(self, cmds, layers=2):
        t = Trace()
        t.buffer("buffer#1")
        t.buffer("buffer#2")
        t.image("image#1", layers=layers)
        t.submit(cmds)
        fr = t.model().frame
        fr.initial_layouts = {k: "general" for k in fr.initial_layouts}     # start in GENERAL
        return fr

    def test_copies_to_different_layers_do_not_race(self):
        fr = self.frame([copy_to_layer(0), copy_to_layer(1)])
        self.assertEqual(codes(fr), [])
        agrees(self, fr)

    def test_copies_to_the_same_layer_race(self):
        fr = self.frame([copy_to_layer(1), copy_to_layer(1)])
        self.assertEqual(codes(fr), ["timing.missing-barrier"])
        agrees(self, fr)

    def test_disjoint_byte_ranges_do_not_race(self):
        fr = self.frame([fill(0, 256), fill(256, 256)])
        self.assertEqual(codes(fr), [])
        fr = self.frame([fill(0, 256), fill(128, 256)])
        self.assertEqual(codes(fr), ["timing.missing-barrier"])
        agrees(self, fr)

    def test_buffer_barrier_covers_only_its_range(self):
        def bar(off, size):
            return ("vkCmdPipelineBarrier", dict(src_stage=XFER, dst_stage=XFER, memory_barriers=[],
                                                 image_barriers=[], buffer_barriers=[{
                                                     "buffer": "buffer#2", "src_stage": XFER, "dst_stage": XFER,
                                                     "src_access": ["VK_ACCESS_TRANSFER_WRITE_BIT"],
                                                     "dst_access": ["VK_ACCESS_TRANSFER_WRITE_BIT"],
                                                     "src_family": None, "dst_family": None,
                                                     "offset": off, "size": size}]))
        self.assertEqual(codes(self.frame([fill(0, 512), bar(0, 512), fill(0, 512)])), [])
        fr = self.frame([fill(0, 512), bar(0, 256), fill(0, 512)])
        self.assertEqual(codes(fr), ["timing.barrier-scope"])
        agrees(self, fr)

    def test_whole_resource_regions_are_none(self):
        fr = self.frame([copy_to_layer(0)], layers=1)
        uses = [op for op in fr.queues["graphics0"] if isinstance(op, Use)]
        self.assertTrue(uses and all(u.region is None for u in uses))


class DynamicOffsets(unittest.TestCase):
    def frame(self, off1, off2):
        t = Trace()
        t.buffer("buffer#1", size=4096)
        t.add("vkCreateDescriptorSetLayout", layout="dsl#1", bindings=[
            {"binding": 0, "type": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC", "count": 1,
             "stages": ["VK_SHADER_STAGE_COMPUTE_BIT"]}])
        t.add("vkAllocateDescriptorSets", sets=["set#1"], set_layouts=["dsl#1"])
        t.add("vkUpdateDescriptorSets", copies=[], writes=[{
            "set": "set#1", "binding": 0, "array_element": 0,
            "type": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC",
            "elements": [{"buffer": "buffer#1", "offset": 0, "range": 256}]}])

        def bind(off):
            return ("vkCmdBindDescriptorSets", dict(bind_point="VK_PIPELINE_BIND_POINT_COMPUTE",
                                                    layout="pl#1", first_set=0, sets=["set#1"],
                                                    dynamic_offsets=[off]))
        t.submit([bind(off1), ("vkCmdDispatch", {}), bind(off2), ("vkCmdDispatch", {})])
        return t.model().frame

    def test_dynamic_offsets_select_the_bytes(self):
        self.assertEqual(codes(self.frame(0, 256)), [])
        fr = self.frame(0, 128)
        self.assertEqual(codes(fr), ["timing.missing-barrier"])
        agrees(self, fr)


class HostWrites(unittest.TestCase):
    """Host writes in sub-allocated memory."""

    def frame(self, write, fresh=True, read=(0, 256)):
        t = Trace()
        t.buffer("buffer#1", size=2560)         # the GPU reads it; it ends inside page 0
        t.buffer("buffer#2", size=4096)
        t.buffer("buffer#3", size=4096)
        t.add("vkAllocateMemory", memory="memory#1", size=1 << 16, properties=[])
        t.add("vkBindBufferMemory", buffer="buffer#1", memory="memory#1", offset=0, size=2560)
        if not fresh:
            t.add("vkBindBufferMemory", buffer="buffer#2", memory="memory#1", offset=2560, size=4096)
        t.add("host_write", memory="memory#1", offset=0, page_size=4096, offsets=[0])   # fills buffer#1
        t.submit([("vkCmdCopyBuffer", dict(src="buffer#1", dst="buffer#3",
                                           regions=[{"src_offset": read[0], "dst_offset": 0,
                                                     "size": read[1]}]))])
        if fresh:                               # the next sub-allocation, right after buffer#1
            t.add("vkBindBufferMemory", buffer="buffer#2", memory="memory#1", offset=2560, size=4096)
        t.add("host_write", memory="memory#1", offset=0, page_size=4096, offsets=[0], **write)
        return t.model().frame

    def test_page_straddle_charges_the_fresh_sub_allocation(self):
        # page-straddle heuristic: only in the optional observed-modification capture mode
        with mock.patch.dict(os.environ, {"COVENANT_HOST_CAPTURE": "observed"}):
            self.assertEqual(codes(self.frame({})), [])
        # the default conservative capture charges every binding in the written page
        self.assertEqual(codes(self.frame({})), ["timing.cross-queue-race"])

    def test_page_shared_with_an_older_binding_is_charged_to_both(self):
        fr = self.frame({}, fresh=False)
        self.assertEqual(codes(fr), ["timing.cross-queue-race"])
        agrees(self, fr)

    def test_page_write_charges_only_the_bytes_of_the_page(self):
        # buffer#1 bytes [0, 2560) lie in page 0; a read of other bytes would not race,
        # but the page covers all of buffer#1 here, so pick a read in page 0
        self.assertEqual(codes(self.frame({}, fresh=False, read=(2048, 256))), ["timing.cross-queue-race"])

    def test_byte_diff_writes_are_exact(self):
        # byte runs are used only in the observed-modification capture mode
        with mock.patch.dict(os.environ, {"COVENANT_HOST_CAPTURE": "observed"}):
            self.assertEqual(codes(self.frame({"writes": [[2600, "00ff"]]}, fresh=False)), [])
            self.assertEqual(codes(self.frame({"writes": [[1000, "ff"]]}, fresh=False)), [])  # bytes not read
            fr = self.frame({"writes": [[100, "ff"]]}, fresh=False)
            self.assertEqual(codes(fr), ["timing.cross-queue-race"])
            agrees(self, fr)
            # a truncated diff falls back to pages
            self.assertEqual(codes(self.frame({"writes": [], "writes_truncated": True}, fresh=False)),
                             ["timing.cross-queue-race"])
        # conservative (default): the written page is charged whatever bytes changed
        for w in ([[2600, "00ff"]], [[1000, "ff"]], []):
            self.assertEqual(codes(self.frame({"writes": w}, fresh=False)), ["timing.cross-queue-race"])

    def test_indexed_draw_reads_only_its_indices(self):
        t = Trace()
        t.buffer("buffer#1", size=4096)
        t.add("vkAllocateMemory", memory="memory#1", size=4096, properties=[])
        t.add("vkBindBufferMemory", buffer="buffer#1", memory="memory#1", offset=0, size=4096)
        t.submit([("vkCmdBindIndexBuffer", dict(buffer="buffer#1", offset=64, index_type="VK_INDEX_TYPE_UINT16")),
                  ("vkCmdDrawIndexed", dict(index_count=6, instance_count=1, first_index=2,
                                            vertex_offset=0, first_instance=0))])
        uses = [op for op in t.model().frame.queues["graphics0"] if isinstance(op, Use) and "index" in op.where]
        self.assertEqual([u.region for u in uses], [((68, 80),)])


class Descriptors(unittest.TestCase):
    """Samplers and other descriptors that bind no resource, and push descriptors."""

    def trace(self, dtype, immutable=False):
        t = Trace()
        t.buffer("buffer#1")
        t.buffer("buffer#2")
        t.add("vkCreateDescriptorSetLayout", layout="dsl#1", create_flags=1, bindings=[
            {"binding": 0, "type": dtype, "count": 1, "stages": ["VK_SHADER_STAGE_COMPUTE_BIT"],
             "immutable_samplers": immutable}])
        t.add("vkCreatePipelineLayout", layout="pl#1", set_layouts=["dsl#1"], push_constants=[])
        return t

    def push(self, buf):
        return ("vkCmdPushDescriptorSetKHR", dict(
            bind_point="VK_PIPELINE_BIND_POINT_COMPUTE", layout="pl#1", set=0, writes=[{
                "set": None, "binding": 0, "array_element": 0, "type": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
                "elements": [{"buffer": buf, "offset": 0, "range": "whole"}]}]))

    def test_push_descriptors_bind_their_resources(self):
        t = self.trace("VK_DESCRIPTOR_TYPE_STORAGE_BUFFER")
        t.submit([self.push("buffer#1"), ("vkCmdDispatch", {}), self.push("buffer#1"), ("vkCmdDispatch", {})])
        m = t.model()
        self.assertEqual(m.draws[0].descriptors[(0, 0)].resource, "buffer#1")
        self.assertEqual(codes(m.frame), ["timing.missing-barrier"])
        t = self.trace("VK_DESCRIPTOR_TYPE_STORAGE_BUFFER")
        t.submit([self.push("buffer#1"), ("vkCmdDispatch", {}), self.push("buffer#2"), ("vkCmdDispatch", {})])
        self.assertEqual(codes(t.model().frame), [])

    def test_samplers_are_bound(self):
        t = self.trace("VK_DESCRIPTOR_TYPE_SAMPLER")
        t.add("vkAllocateDescriptorSets", sets=["set#1"], set_layouts=["dsl#1"])
        t.add("vkUpdateDescriptorSets", copies=[], writes=[{
            "set": "set#1", "binding": 0, "array_element": 0, "type": "VK_DESCRIPTOR_TYPE_SAMPLER",
            "elements": [{"sampler": "sampler#1", "view": None, "layout": "VK_IMAGE_LAYOUT_UNDEFINED"}]}])
        bind = ("vkCmdBindDescriptorSets", dict(bind_point="VK_PIPELINE_BIND_POINT_COMPUTE", layout="pl#1",
                                                first_set=0, sets=["set#1"], dynamic_offsets=[]))
        t.submit([bind, ("vkCmdDispatch", {})])
        self.assertEqual(t.model().draws[0].descriptors[(0, 0)].resource, "sampler#1")
        t = self.trace("VK_DESCRIPTOR_TYPE_SAMPLER", immutable=True)
        t.add("vkAllocateDescriptorSets", sets=["set#1"], set_layouts=["dsl#1"])
        t.submit([bind, ("vkCmdDispatch", {})])
        self.assertEqual(t.model().draws[0].descriptors[(0, 0)].resource, "immutable sampler")


OIT_GATHER = """#version 450
layout (early_fragment_tests) in;
struct Node { vec4 color; float depth; uint next; };
layout (set = 0, binding = 1) buffer GeometrySBO { uint count; uint maxNodeCount; };
layout (set = 0, binding = 2, r32ui) uniform coherent uimage2D headIndexImage;
layout (set = 0, binding = 3) buffer LinkedListSBO { Node nodes[]; };
layout (set = 0, binding = 4) buffer Plain { uint plain[]; };
void main() {
    uint nodeIdx = atomicAdd(count, 1);
    if (nodeIdx < maxNodeCount) {
        uint prev = imageAtomicExchange(headIndexImage, ivec2(gl_FragCoord.xy), nodeIdx);
        nodes[nodeIdx].color = vec4(1.0);
        nodes[nodeIdx].depth = gl_FragCoord.z;
        nodes[nodeIdx].next = prev;
    }
    plain[0] = 1u;
}
"""


@unittest.skipUnless(shutil.which("glslangValidator"), "needs glslangValidator")
class Atomics(unittest.TestCase):
    """Atomics and atomically allocated slots; image atomics through OpImageTexelPointer."""

    def test_binding_modes(self):
        from covenant import spirv
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "gather.frag")
            with open(src, "w") as f:
                f.write(OIT_GATHER)
            m = spirv.compile_glsl(src, d)
        acc = spirv.entry_bindings(m, "main")
        self.assertTrue(acc[(0, 2)], "image atomic through OpImageTexelPointer is a write")
        modes = spirv.entry_binding_modes(m, "main")
        self.assertEqual(modes[(0, 1)], frozenset({("atomic", 0), ("read", 1)}))
        self.assertEqual(modes[(0, 2)], frozenset({("atomic", None)}))
        self.assertEqual(modes[(0, 3)], frozenset({("slot", 0)}))
        self.assertNotIn((0, 4), modes)          # a plain store
        self.assertTrue(spirv.modes_compatible(modes[(0, 1)], modes[(0, 1)]))
        self.assertFalse(spirv.modes_compatible(modes[(0, 1)], frozenset({("read", 0)})))

    def test_atomic_draws_do_not_race(self):
        mode = frozenset({("atomic", 0), ("read", 1)})
        ops = [Use("b", "write", None, "draw 1", "FRAGMENT_SHADER", "SHADER_STORAGE_WRITE", cmd=1, atomic=mode),
               Use("b", "write", None, "draw 2", "FRAGMENT_SHADER", "SHADER_STORAGE_WRITE", cmd=2, atomic=mode),
               Use("b", "write", None, "plain", "FRAGMENT_SHADER", "SHADER_STORAGE_WRITE", cmd=3)]
        fr = Frame(queues={"q": ops})
        hz = []
        timing.check(fr, Report(), hazards=hz, exhaustive=True)
        self.assertEqual({frozenset(p[:2]) for p in hz}, {frozenset({("q", 1), ("q", 3)}),
                                                         frozenset({("q", 2), ("q", 3)})})
        agrees(self, fr)
        # on two queues the exemption does not apply
        fr = Frame(queues={"q": ops[:1], "r": [ops[1]]})
        self.assertEqual(codes(fr), ["timing.cross-queue-race"])
        agrees(self, fr)


class SignalScope(unittest.TestCase):
    def test_transition_is_ordered_before_a_later_signal(self):
        # sync2 present idiom: transition with dstStage BOTTOM_OF_PIPE, signal at ALL_GRAPHICS
        # (bench case 61; syncval HEAD accepts it)
        t = Trace()
        t.image("image#1")
        t.add("vkCreateSemaphore", semaphore="sem#1", type="binary")
        t.submit([("vkCmdPipelineBarrier2", dict(memory_barriers=[], buffer_barriers=[], image_barriers=[{
            "image": "image#1", "old_layout": "VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL",
            "new_layout": "VK_IMAGE_LAYOUT_PRESENT_SRC_KHR",
            "src_stage": ["VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT"],
            "src_access": ["VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT"],
            "dst_stage": ["VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT"], "dst_access": [],
            "src_family": None, "dst_family": None, "range": {"aspect": ["VK_IMAGE_ASPECT_COLOR_BIT"],
                                                              "base_mip": 0, "mip_count": 1,
                                                              "base_layer": 0, "layer_count": 1}}]))],
                 signal=[{"semaphore": "sem#1", "value": None,
                          "stage": ["VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT"]}])
        self.assertNotIn("timing.signal-scope", codes(t.model().frame))


class PipelineStages(unittest.TestCase):
    def test_stages_in_pipeline_order(self):
        t = Trace()
        t.add("vkCreatePipelineLayout", layout="pl#1", set_layouts=[], push_constants=[])
        t.add("vkCreateGraphicsPipelines", pipelines=[{"pipeline": "pipeline#1", "layout": "pl#1", "stages": [
            {"stage": ["VK_SHADER_STAGE_FRAGMENT_BIT"], "sha1": "f", "entry": "main"},
            {"stage": ["VK_SHADER_STAGE_VERTEX_BIT"], "sha1": "v", "entry": "main"}]}])
        self.assertEqual([x[1] for x in t.model().pipelines["pipeline#1"].stages], ["v", "f"])


class OwnershipAcquire(unittest.TestCase):
    """The acquire half of a matched transfer is no discard."""

    def test_matched_acquire_keeps_contents(self):
        t = Trace()
        t.image("image#1", sharing="VK_SHARING_MODE_EXCLUSIVE")
        qfot = {"image": "image#1", "old_layout": "VK_IMAGE_LAYOUT_GENERAL", "new_layout": "VK_IMAGE_LAYOUT_GENERAL",
                "src_stage": COMP, "dst_stage": COMP, "src_access": [], "dst_access": [],
                "src_family": 0, "dst_family": 1, "aspect": ["VK_IMAGE_ASPECT_COLOR_BIT"],
                "base_mip": 0, "level_count": 1, "base_layer": 0, "layer_count": 1}
        t.add("vkCreateSemaphore", semaphore="sem#1", type="binary")
        t.submit([("vkCmdPipelineBarrier", dict(src_stage=COMP, dst_stage=COMP, memory_barriers=[],
                                                buffer_barriers=[], image_barriers=[qfot]))],
                 signal=[{"semaphore": "sem#1"}])
        t.submit([("vkCmdPipelineBarrier", dict(src_stage=TOP, dst_stage=COMP, memory_barriers=[],
                                                buffer_barriers=[], image_barriers=[qfot]))],
                 queue="queue#2", wait=[{"semaphore": "sem#1", "stage": COMP}])
        fr = t.model().frame
        acq = [op for op in fr.queues["compute0"] if isinstance(op, Barrier) and op.res]
        self.assertEqual((acq[0].old, acq[0].new), ("general", None))

    def test_matched_acquire_does_not_repeat_the_transition(self):
        t = Trace()
        t.image("image#1", sharing="VK_SHARING_MODE_EXCLUSIVE")
        qfot = {"image": "image#1", "old_layout": "VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL",
                "new_layout": "VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL",
                "src_stage": XFER, "dst_stage": [], "src_access": ["VK_ACCESS_TRANSFER_WRITE_BIT"],
                "dst_access": [], "src_family": 1, "dst_family": 0, "aspect": ["VK_IMAGE_ASPECT_COLOR_BIT"],
                "base_mip": 0, "level_count": 1, "base_layer": 0, "layer_count": 1}
        acq = dict(qfot, src_stage=[], src_access=[], dst_stage=["VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT"],
                   dst_access=["VK_ACCESS_SHADER_READ_BIT"])
        t.add("vkCreateSemaphore", semaphore="sem#1", type="binary")
        t.submit([copy_to_layer(0), ("vkCmdPipelineBarrier", dict(src_stage=XFER, dst_stage=[], memory_barriers=[],
                                                                  buffer_barriers=[], image_barriers=[qfot]))],
                 queue="queue#2", signal=[{"semaphore": "sem#1"}])
        t.submit([("vkCmdPipelineBarrier", dict(src_stage=[], dst_stage=acq["dst_stage"], memory_barriers=[],
                                                buffer_barriers=[], image_barriers=[acq]))],
                 wait=[{"semaphore": "sem#1", "stage": ["VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT"]}])
        fr = t.model().frame
        fr.initial_layouts = {"image#1": "transfer_dst"}
        lts = [op for ops in fr.queues.values() for op in ops if isinstance(op, Barrier) and op.new]
        self.assertEqual(len(lts), 1)
        self.assertNotIn("timing.stale-barrier", codes(fr))


class TimelineInitialValue(unittest.TestCase):
    """A wait the initial value satisfies is not bound to a later signal."""

    def test_wait_for_initial_value(self):
        t = Trace()
        t.add("vkCreateSemaphore", semaphore="sem#1", type="timeline", initial_value=0)
        t.add("vkWaitSemaphores", semaphores=[{"semaphore": "sem#1", "value": 0}])
        t.submit([], signal=[{"semaphore": "sem#1", "value": 1}])
        fr = t.model().frame
        self.assertNotIn("timing.deadlock", codes(fr))


RP_ATT = {"format": "VK_FORMAT_R8G8B8A8_UNORM", "samples": 1, "load": "VK_ATTACHMENT_LOAD_OP_CLEAR",
          "store": "VK_ATTACHMENT_STORE_OP_STORE", "stencil_load": "VK_ATTACHMENT_LOAD_OP_DONT_CARE",
          "stencil_store": "VK_ATTACHMENT_STORE_OP_DONT_CARE",
          "initial_layout": "VK_IMAGE_LAYOUT_UNDEFINED", "final_layout": "VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL"}


class RenderPasses(unittest.TestCase):
    def model(self, deps):
        t = Trace()
        t.image("image#1")
        t.add("vkCreateImageView", view="view#1", image="image#1", format="VK_FORMAT_R8G8B8A8_UNORM",
              view_type="VK_IMAGE_VIEW_TYPE_2D", aspect=["VK_IMAGE_ASPECT_COLOR_BIT"], base_mip=0,
              level_count=1, base_layer=0, layer_count=1)
        t.add("vkCreateRenderPass", render_pass="rp#1", attachments=[RP_ATT], dependencies=deps,
              subpasses=[{"color": [{"attachment": 0, "layout": "VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL"}]}])
        t.add("vkCreateFramebuffer", framebuffer="fb#1", attachments=["view#1"])
        t.submit([("vkCmdBeginRenderPass", dict(render_pass="rp#1", framebuffer="fb#1", clear_values=[])),
                  ("vkCmdEndRenderPass", {})])
        return t.model()

    def test_one_barrier_per_dependency(self):
        # color -> BOTTOM and depth -> FRAGMENT_SHADER stay separate
        deps = [{"src_subpass": 0, "dst_subpass": -1, "src_stage": ["VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT"],
                 "dst_stage": BOT, "src_access": ["VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT"],
                 "dst_access": ["VK_ACCESS_MEMORY_READ_BIT"]},
                {"src_subpass": 0, "dst_subpass": -1, "src_stage": ["VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT"],
                 "dst_stage": ["VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT"],
                 "src_access": ["VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT"],
                 "dst_access": ["VK_ACCESS_SHADER_READ_BIT"]}]
        q = self.model(deps).frame.queues["graphics0"]
        ext = [op for op in q if isinstance(op, Barrier) and op.res is None and "EXTERNAL dependency" in op.where]
        self.assertEqual(len(ext), 2)
        self.assertEqual(len({op.call for op in ext}), 1)
        self.assertEqual(len({op.struct for op in ext}), 2)
        # the color attachment's final transition takes only the color dependency
        lt = [op for op in q if isinstance(op, Barrier) and op.res and "final layout" in op.where][0]
        self.assertEqual(lt.dst_stages, F({"BOTTOM_OF_PIPE"}))
        # a later sampled read is not ordered after the store / transition (T14)
        fr = self.model(deps).frame
        fr.queues["graphics0"].append(Use("image#1", "read", "shader_read", "sample", "FRAGMENT_SHADER",
                                          "SHADER_SAMPLED_READ"))
        self.assertIn("timing.barrier-scope", codes(fr))
        agrees(self, fr)

    def test_load_and_store_ops_are_accesses(self):
        q = self.model([]).frame.queues["graphics0"]
        ls = [(u.access, u.stage) for u in q if isinstance(u, Use) and u.pass_op]
        self.assertEqual(ls, [("write", "COLOR_ATTACHMENT_OUTPUT"), ("write", "COLOR_ATTACHMENT_OUTPUT")])

    def test_dynamic_rendering_load_store(self):
        t = Trace()
        t.image("image#1")
        t.add("vkCreateImageView", view="view#1", image="image#1", format="VK_FORMAT_R8G8B8A8_UNORM",
              aspect=["VK_IMAGE_ASPECT_COLOR_BIT"], base_mip=0, level_count=1, base_layer=0, layer_count=1)
        att = {"view": "view#1", "layout": "VK_IMAGE_LAYOUT_GENERAL", "load": "VK_ATTACHMENT_LOAD_OP_LOAD",
               "store": "VK_ATTACHMENT_STORE_OP_NONE"}
        t.submit([("vkCmdBeginRendering", dict(color=[att])), ("vkCmdEndRendering", {})])
        q = t.model().frame.queues["graphics0"]
        ls = [(u.access, u.flag) for u in q if isinstance(u, Use) and u.pass_op]
        self.assertEqual(ls, [("read", "COLOR_ATTACHMENT_READ")])


class Layouts(unittest.TestCase):
    def test_per_aspect_matching(self):
        # case 63: a depth-only use of a DEPTH_STENCIL_ATTACHMENT image
        self.assertTrue(timing.layouts_match("depth_attachment", "depth_stencil_attachment"))
        self.assertTrue(timing.layouts_match("attachment", "color_attachment"))
        self.assertTrue(timing.layouts_match("read_only", "shader_read"))
        self.assertFalse(timing.layouts_match("depth_read_only", "depth_stencil_attachment"))
        self.assertFalse(timing.layouts_match("shader_read", "depth_stencil_read"))
        self.assertFalse(timing.layouts_match("general", "color_attachment"))

    def test_layouts_tracked_per_region(self):
        ops = [Barrier("img", None, "transfer_dst", "t0", F({"TOP_OF_PIPE"}), F(), F({"TRANSFER"}),
                       F({"TRANSFER_WRITE"}), region=((0, 1),)),
               Use("img", "write", "transfer_dst", "copy0", "TRANSFER", "TRANSFER_WRITE", region=((0, 1),)),
               Use("img", "read", "general", "read1", "TRANSFER", "TRANSFER_READ", region=((1, 2),))]
        fr = Frame(queues={"gfx": ops}, initial_layouts={"img": "general"})
        self.assertEqual(codes(fr), [])
        ops[2].region = ((0, 2),)
        self.assertIn("timing.wrong-layout", codes(fr))


class Ordering(unittest.TestCase):
    def test_war_through_a_barrier_on_another_resource(self):
        # WAR needs only an execution dependency
        fr = Frame(queues={"q": [
            Use("a", "read", None, "d1", "COMPUTE_SHADER", "SHADER_STORAGE_READ"),
            Barrier("b", None, None, "bar", F({"COMPUTE_SHADER"}), F({"SHADER_WRITE"}), F({"COMPUTE_SHADER"}),
                    F({"SHADER_READ"})),
            Use("a", "write", None, "d2", "COMPUTE_SHADER", "SHADER_STORAGE_WRITE")]})
        self.assertEqual(codes(fr), [])
        agrees(self, fr)

    def test_discard_transition_kills_earlier_write(self):
        # R1': depth reuse across frames, UNDEFINED -> attachment with srcAccessMask 0
        FT = F({"EARLY_FRAGMENT_TESTS", "LATE_FRAGMENT_TESTS"})
        DSW = "DEPTH_STENCIL_ATTACHMENT_WRITE"
        fr = Frame(queues={"q": [
            Use("d", "write", None, "frame 1", "LATE_FRAGMENT_TESTS", DSW),
            Barrier("d", None, "attachment", "barrier", FT, F(), FT, F({DSW})),
            Use("d", "write", None, "frame 2", "EARLY_FRAGMENT_TESTS", DSW)]})
        # off by default: the frame-1 write is never made available and can land after
        # the frame-2 write (syncval WAW; bench case 64)
        self.assertEqual(codes(fr), ["timing.barrier-scope"])
        agrees(self, fr)
        hz = []
        # R1' is a closure of the opt-in heuristic R1 (COVENANT_HEURISTIC_DISCARD=1)
        timing.check(fr, Report(), hazards=hz, exhaustive=True, discard_relaxed=True, discard_kill=True)
        self.assertEqual({frozenset(p[:2]) for p in hz},
                         {frozenset((a.key, b.key)) for a, b in
                          timing_ref.Reference(fr, True, discard_kill=True).races()})
        self.assertEqual(hz, [])

    def test_accesses_of_one_command(self):
        # a draw's input-attachment read and its own attachment write
        fr = Frame(queues={"q": [
            Use("c", "write", None, "draw", "COLOR_ATTACHMENT_OUTPUT", "COLOR_ATTACHMENT_WRITE", cmd=1),
            Use("c", "read", None, "draw", "FRAGMENT_SHADER", "INPUT_ATTACHMENT_READ", cmd=1),
            Use("c", "read", None, "draw2", "FRAGMENT_SHADER", "INPUT_ATTACHMENT_READ", cmd=2)]})
        self.assertEqual(codes(fr), ["timing.missing-barrier"])
        agrees(self, fr)


class Window(unittest.TestCase):
    """Theorem 3: period, frames in flight and the w + N + 2 window."""

    def frame(self, frames=10, in_flight=2, wait=True):
        from covenant.timing import Present, Signal, Wait
        host, gpu = [], []
        for i in range(frames):
            if wait and i >= in_flight:
                host.append(Wait(f"fence{i % in_flight}@{i - in_flight}", "fence wait"))
            host.append(Use(f"ubo{i % in_flight}", "write", None, "memcpy", "HOST", "HOST_WRITE"))
            host.append(Signal(f"submit@{i}", "submit"))
            gpu.append(Wait(f"submit@{i}", "submit"))
            gpu.append(Use(f"ubo{i % in_flight}", "read", None, "draw", "VERTEX_SHADER", "UNIFORM_READ"))
            gpu.append(Signal(f"fence{i % in_flight}@{i}", "fence"))
            host.append(Present("swapchain", "present"))
        return Frame(queues={"host": host, "gfx": gpu})

    def test_period_and_frames_in_flight(self):
        from covenant.timing_window import window, window_frame
        info = window(self.frame())
        self.assertEqual((info.periodic, info.period, info.in_flight), (True, 2, 2))
        self.assertTrue(info.complete)
        self.assertEqual(info.needed, info.warmup + 4)
        r = Report()
        timing.check(self.frame(), r)
        self.assertEqual(r.items, [])
        self.assertIs(r.meta["timing.window"].complete, True)
        wf = window_frame(self.frame(), info)
        self.assertLess(sum(map(len, wf.queues.values())), sum(map(len, self.frame().queues.values())))

    def test_no_fence_wait_is_unbounded(self):
        from covenant.timing_window import window
        info = window(self.frame(wait=False))
        self.assertTrue(info.periodic)
        self.assertIsNone(info.in_flight)
        self.assertFalse(info.complete)
        self.assertIn("timing.cross-queue-race", codes(self.frame(wait=False)))


class Records(unittest.TestCase):
    def test_undecodable_line_is_reported(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"seq": 1, "fn": "vkCreateBuffer", "buffer": "b", "size": 4, "usage": []}) + "\n")
            f.write("{not json\n")
            f.write(json.dumps({"seq": 2, "fn": "vkCreateBuffer", "buffer": "c", "size": 4, "usage": []}) + "\n")
        try:
            recs = load_records(f.name)
        finally:
            os.unlink(f.name)
        self.assertEqual([r["fn"] for r in recs], ["vkCreateBuffer", "undecodable-record", "vkCreateBuffer"])
        self.assertTrue(_Loader(recs).run().warnings)


if __name__ == "__main__":
    unittest.main()
