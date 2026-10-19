"""Load a COVENANT model-driver trace (JSON lines) into a HostModel.

The model driver (model_driver/libcovvk.so) records every Vulkan call a host
program makes. This module rebuilds from that trace:

* the objects the program created (images, views, buffers, memory, swapchains,
  pipelines and their descriptor set layouts);
* every draw/dispatch that was submitted, with the pipeline and the descriptor
  bindings it saw, resolved to concrete resources and view formats;
* the fixed-function state the meaning rules read (shader_meaning docstring):
  the swapchain's imageColorSpace (SwapchainInfo.color_space), and per draw
  the first viewport, dynamic or static (DrawEvent.viewport), and the clear
  color of each color attachment the pass cleared (DrawEvent.color_clears);
  blend factors stay on PipelineInfo.record["blend"];
* a ``covenant.timing.Frame``: one op list per queue ("host", "graphics0",
  "compute0", ..., and "present:<image>" for the presentation engine, one per
  swapchain image) that the timing
  checker can explore for every schedule the recorded synchronization allows.

Mapping from Vulkan to timing ops
---------------------------------
* vkQueueSubmit: host Signal("submit@N") -> queue Wait("submit@N") (work
  starts after it is submitted), then for each batch: waits, the commands of
  its command buffers (as recorded at submit time), signals; then the fence.
* commands: resource reads/writes become Use ops (draws read their bound
  descriptors, vertex/index/indirect buffers and write their attachments with
  the attachment layout; dispatches write storage descriptors; copies, blits,
  fills and clears read/write); pipeline barriers become Barrier ops.
* binary semaphores: every signal is a distinct instance "<sem>:k"; a wait
  consumes the pending instance. Timeline semaphores: "<sem>=v"; a wait for v
  is matched to the first signal with a value >= v.
* fences: Signal at the end of the submit (or on "present" for an acquire
  fence); vkWaitForFences / a successful vkGetFenceStatus become host Waits;
  vkResetFences starts a new generation.
* vkAcquireNextImageKHR: host Acquire(swapchain); on "present:<image>": Wait for the
  previous present of that image, a read of the image (the display), then the
  Signal of the acquire semaphore/fence.
* vkQueuePresentKHR: on "present:<image>" (the presentation engine waits, not the
  presenting queue): Wait on the wait semaphores, a
  read of the image in layout "present", Signal("presented:..."); host Present.
* vkDeviceWaitIdle / vkQueueWaitIdle: each queue Signals, the host Waits.
* host_write records (page-protection traps) become host Use(write) of the
  written bytes of every buffer/image bound there. Default
  COVENANT_HOST_CAPTURE=conservative: every byte of every binding in each page
  the trap saw written in the interval (stores of unchanged bytes included).
  COVENANT_HOST_CAPTURE=observed: the changed byte runs
  when the record has them, else the written pages (see _Loader.host_write).

Acquire and Present ops are both placed on the host queue so that
timing.check_swapchains (which counts per queue) sees the host's acquire /
present order. Frame.swapchains[name] = (image_count, surface minImageCount),
the pair the Vulkan rule "at most image_count - minImageCount + 1 images held"
is stated with.
"""
import json
import os
import re
import struct
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

from . import desc_index as _desc_index
from . import ranges as _ranges
from . import spirv as _spirv
from .timing import _stage_in
from .timing import Acquire, Barrier, Frame, IndexGuard, Present, Signal, Use, Wait

# queue families of the model driver: 0 = graphics+compute+transfer, 1 = compute
FAMILY_NAMES = {0: "graphics", 1: "compute"}


# ------------------------------------------------------------------ data model
class Bound(NamedTuple):
    """One resolved descriptor: (descriptor type, resource name, view format)."""
    type: str
    resource: Optional[str]
    view_format: Optional[str]


@dataclass
class ImageInfo:
    id: str
    name: str
    format: str
    usage: list
    extent: tuple = ()
    mip_levels: int = 1
    array_layers: int = 1
    initial_layout: str = "undefined"
    swapchain: Optional[str] = None      # swapchain name if owned by a swapchain
    swapchain_index: Optional[int] = None
    memory: Optional[str] = None


@dataclass
class ViewInfo:
    id: str
    image: str            # image id
    format: str
    view_type: str = ""


@dataclass
class BufferInfo:
    id: str
    name: str
    size: int
    usage: list
    memory: Optional[str] = None
    offset: int = 0


@dataclass
class MemoryInfo:
    id: str
    size: int
    properties: list
    bindings: list = field(default_factory=list)     # [(kind, id, offset, size)]


@dataclass
class SwapchainInfo:
    id: str
    name: str
    image_count: int
    min_image_count: int            # what the program asked for
    surface_min_image_count: int    # VkSurfaceCapabilitiesKHR::minImageCount
    format: str
    extent: tuple
    images: list                    # image names, index order
    color_space: Optional[str] = None   # VkSwapchainCreateInfoKHR::imageColorSpace


_STAGE_ORDER = ["TASK", "MESH", "VERTEX", "TESSELLATION_CONTROL", "TESSELLATION_EVALUATION", "GEOMETRY",
                "FRAGMENT", "COMPUTE"]


def _stage_rank_name(st):
    """Position of a pipeline stage name ("VERTEX_SHADER", ...) in its pipeline."""
    from .timing import GRAPHICS_ORDER, MESH_ORDER
    for order in (GRAPHICS_ORDER, MESH_ORDER):
        if st in order:
            return order.index(st)
    return 99


def _stage_rank(stage):
    """Position of a shader stage ("VK_SHADER_STAGE_VERTEX_BIT", ...) in the
    pipeline, for linking stages in order; unknown stages (ray tracing) keep
    their relative order after these."""
    for i, k in enumerate(_STAGE_ORDER):
        if f"VK_SHADER_STAGE_{k}_BIT" in stage.split("|") or f"VK_SHADER_STAGE_{k}_BIT_EXT" in stage.split("|") \
                or f"VK_SHADER_STAGE_{k}_BIT_NV" in stage.split("|"):
            return i
    return len(_STAGE_ORDER)


@dataclass
class PipelineInfo:
    id: str
    kind: str                       # "graphics" | "compute"
    stages: list                    # [(stage, sha1, entry)]
    color_formats: list
    depth_format: Optional[str]
    stencil_format: Optional[str]
    layout: Optional[str]
    set_layouts: list               # per set index: [{binding,type,count,stages}]
    render_pass: Optional[str] = None
    vertex_bindings: list = field(default_factory=list)
    record: dict = field(default_factory=dict)


@dataclass
class DrawEvent:
    seq: int                        # seq of the vkCmdDraw*/vkCmdDispatch* record
    fn: str
    kind: str                       # "draw" | "dispatch"
    pipeline: Optional[str]
    queue: str
    submit_seq: int
    command_buffer: str
    descriptors: dict               # (set, binding) -> Bound (array element 0)
    descriptor_elements: dict       # (set, binding, element) -> Bound
    attachments: list               # [(role, image name, view format, layout)]
    vertex_buffers: list
    index_buffer: Optional[str] = None
    # depth state at the draw (for depth-convention checks such as reverse-Z)
    depth_test: Optional[bool] = None
    depth_write: Optional[bool] = None
    depth_compare: Optional[str] = None        # dynamic state if set, else the pipeline's
    depth_clear: Optional[float] = None        # clear value of the depth attachment, if cleared
    viewport_depth: Optional[tuple] = None     # (minDepth, maxDepth) of the first viewport
    viewport: Optional[tuple] = None           # (x, y, width, height) of the first viewport
    color_clears: dict = field(default_factory=dict)   # attachment image -> clear color, if cleared


@dataclass
class AccelInfo:
    """VK_KHR_acceleration_structure object: bytes [offset, offset+size) of its buffer."""
    id: str
    buffer: str                     # buffer id
    offset: int
    size: int
    type: str                       # "top" | "bottom" | "generic"
    refs: Optional[frozenset] = None    # TLAS: BLAS ids its last build referenced (None: unknown)


@dataclass
class HostModel:
    images: dict = field(default_factory=dict)       # id -> ImageInfo
    views: dict = field(default_factory=dict)        # id -> ViewInfo
    buffers: dict = field(default_factory=dict)      # id -> BufferInfo
    memories: dict = field(default_factory=dict)     # id -> MemoryInfo
    swapchains: dict = field(default_factory=dict)   # name -> SwapchainInfo
    pipelines: dict = field(default_factory=dict)    # id -> PipelineInfo
    set_layouts: dict = field(default_factory=dict)  # id -> [binding dicts]
    shader_modules: dict = field(default_factory=dict)  # id -> sha1
    draws: list = field(default_factory=list)        # [DrawEvent]
    frame: Frame = None
    names: dict = field(default_factory=dict)        # handle id -> display name
    queues: dict = field(default_factory=dict)       # queue id -> queue name
    warnings: list = field(default_factory=list)     # unimplemented calls etc.
    # commands and data the model does not cover, reported explicitly (never
    # skipped silently): fn or "what" -> {"count", "first", "effect"}; also
    # Frame.unsupported, which timing.check reports
    unsupported: dict = field(default_factory=dict)
    accel: dict = field(default_factory=dict)        # acceleration structure id -> AccelInfo
    records: list = field(default_factory=list)      # raw trace records

    def name(self, hid):
        return self.names.get(hid, hid)


# ------------------------------------------------------------------ helpers
_LAYOUT_SPECIAL = {
    "UNDEFINED": "undefined",
    "PRESENT_SRC_KHR": "present",
    "SHADER_READ_ONLY_OPTIMAL": "shader_read",
    "COLOR_ATTACHMENT_OPTIMAL": "color_attachment",
    "DEPTH_STENCIL_ATTACHMENT_OPTIMAL": "depth_stencil_attachment",
    "DEPTH_STENCIL_READ_ONLY_OPTIMAL": "depth_stencil_read",
    "TRANSFER_SRC_OPTIMAL": "transfer_src",
    "TRANSFER_DST_OPTIMAL": "transfer_dst",
}


def _region_union(a, b):
    """Union of two element regions (None = whole)."""
    if a is None or b is None:
        return None
    out = []
    for lo, hi in sorted(tuple(a) + tuple(b)):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return tuple(out)


MAX_HOST_INTERVALS = 8


def _region_coarsen(reg, k=MAX_HOST_INTERVALS):
    """At most k intervals, by closing the smallest gaps (a superset of reg).

    A host write record of the model driver can carry hundreds of thousands of exact
    byte runs (length-only past the content budget). Kept as they are, a buffer
    region of thousands of intervals makes every region intersection quadratic
    (a 10-frame Khronos multi_draw_indirect trace needed > 4.8 GB). Closing the
    smallest gaps keeps the region exact where it matters (which bindings and
    which ring halves the host wrote) and only adds unwritten gap bytes."""
    if reg is None or len(reg) <= k:
        return reg
    gaps = sorted(reg[i + 1][0] - reg[i][1] for i in range(len(reg) - 1))
    limit = gaps[len(reg) - k - 1]          # close every gap <= limit
    out = [reg[0]]
    for lo, hi in reg[1:]:
        if lo - out[-1][1] <= limit:
            out[-1] = (out[-1][0], hi)
        else:
            out.append((lo, hi))
    return tuple(out)


def _coarsen_spans(spans, boundaries, k=4 * MAX_HOST_INTERVALS):
    """Merge sorted, disjoint byte spans [(lo, hi)] of one host_write record
    down to about k spans by closing the smallest gaps, but never a gap that
    contains a binding boundary: every binding then gets the same set of
    written bindings as before, and within a binding only unwritten gap bytes
    are added (see _region_coarsen)."""
    if len(spans) <= k:
        return spans
    import bisect
    bnd = sorted(set(boundaries))

    def closable(a, b):             # no boundary strictly inside (a, b)
        i = bisect.bisect_right(bnd, a)
        return i >= len(bnd) or bnd[i] >= b
    gaps = sorted((spans[i + 1][0] - spans[i][1], i) for i in range(len(spans) - 1)
                  if closable(spans[i][1], spans[i + 1][0]))
    close = {i for _, i in gaps[:max(0, len(spans) - k)]}
    out = [spans[0]]
    for i in range(1, len(spans)):
        if i - 1 in close:
            out[-1] = (out[-1][0], spans[i][1])
        else:
            out.append(spans[i])
    return out


def _region_intersect(a, b):
    """Intersection of two element regions (None = whole)."""
    if a is None:
        return b
    if b is None:
        return a
    out = []
    for lo, hi in a:
        for lo2, hi2 in b:
            x, y = max(lo, lo2), min(hi, hi2)
            if x < y:
                out.append((x, y))
    return tuple(sorted(out))


def short_layout(name):
    """VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL -> 'shader_read' (None stays None)."""
    if name is None:
        return None
    s = str(name).replace("VK_IMAGE_LAYOUT_", "")
    if s in _LAYOUT_SPECIAL:
        return _LAYOUT_SPECIAL[s]
    for suf in ("_KHR", "_EXT", "_OPTIMAL"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s.lower()


# sync2 splits the transfer stage into COPY, BLIT, CLEAR and RESOLVE: they stay distinct
# stages (a COPY barrier does not order a blit); ALL_TRANSFER (= sync1 TRANSFER) holds all four
# (timing.STAGE_INCLUDES)
_STAGE_ALIAS = {"ALL_TRANSFER": "TRANSFER",
                "VERTEX_ATTRIBUTE_INPUT": "VERTEX_INPUT", "INDEX_INPUT": "VERTEX_INPUT"}
_PRE_RASTER = ("VERTEX_SHADER", "TESSELLATION_CONTROL_SHADER", "TESSELLATION_EVALUATION_SHADER",
               "GEOMETRY_SHADER", "TASK_SHADER", "MESH_SHADER")


def _strip(name, prefix):
    s = re.sub(prefix, "", str(name))
    return re.sub(r"_BIT(_[A-Z]+)?$", "", s)


IMPLICIT_ATT_ACCESS = ["VK_ACCESS_INPUT_ATTACHMENT_READ_BIT", "VK_ACCESS_COLOR_ATTACHMENT_READ_BIT",
                       "VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT",
                       "VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT",
                       "VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT"]


# how to read a subpass dependency whose flags an older driver did not record:
# framebuffer-local (default, conservative) or, for comparisons, framebuffer-global
_LEGACY_GLOBAL = os.environ.get("COVENANT_LEGACY_SUBPASS_FLAGS", "") == "global"
_BY_REGION = "VK_DEPENDENCY_BY_REGION_BIT"
_VIEW_LOCAL = ("VK_DEPENDENCY_VIEW_LOCAL_BIT", "VK_DEPENDENCY_VIEW_LOCAL_BIT_KHR")


def _fb_local(d):
    """The Barrier.fb_local value of a dependency record (a
    subpass dependency, or a vkCmdPipelineBarrier* / VkDependencyInfo record):
    False (framebuffer-global), True (BY_REGION / VIEW_LOCAL: framebuffer-local
    between framebuffer-space stages) or "none" (VIEW_LOCAL with a non-zero view
    offset: framebuffer-space pairs are at different views, ordered at no position).
    A record without the field (a subpass dependency in a trace of an older driver
    that did not record dependencyFlags) is taken as framebuffer-local (unknown: the weaker dependency)."""
    flags = d.get("dependency_flags")
    if flags is None:
        return not _LEGACY_GLOBAL
    if isinstance(flags, (int, float)):
        by, vl = bool(int(flags) & 1), bool(int(flags) & 2)
    else:
        flags = [str(f) for f in flags]
        by, vl = _BY_REGION in flags, any(f in flags for f in _VIEW_LOCAL)
    if vl and d.get("view_offset"):
        return "none"
    return bool(by or vl)


def _fb_join(vals):
    """fb_local of one barrier standing for several dependencies: the weakest."""
    vals = list(vals)
    if "none" in vals:
        return "none"
    return any(vals)


def _union_dep_masks(deps):
    """Masks of one barrier standing for the union of several subpass dependencies."""
    if not deps:
        return {}
    out = {}
    for k_in, k_out, fn in (("src_stage", "src_stages", stage_mask), ("src_access", "src_access", access_mask),
                            ("dst_stage", "dst_stages", stage_mask), ("dst_access", "dst_access", access_mask)):
        acc = frozenset()
        for d in deps:
            v = fn(d.get(k_in))
            if v is None:
                acc = None
                break
            acc |= v
        out[k_out] = acc
    return out


_ROLE_STAGES = {"color": ("COLOR_ATTACHMENT_OUTPUT",), "color_resolve": ("COLOR_ATTACHMENT_OUTPUT",),
                "depth": ("EARLY_FRAGMENT_TESTS", "LATE_FRAGMENT_TESTS"),
                "input": ("FRAGMENT_SHADER",)}


def _deps_for(deps, role, src):
    """The subpass dependencies whose src (src=True) or dst execution scope
    holds the stages an attachment of this role is used in; all of them if
    none does (the transition then takes the union, as before)."""
    from .timing import _stage_in
    stages = _ROLE_STAGES.get(role, ("COLOR_ATTACHMENT_OUTPUT",))
    out = []
    for d in deps:
        m = stage_mask(d.get("src_stage" if src else "dst_stage"))
        if m is None or any(_stage_in(st, m, src) for st in stages):
            out.append(d)
    return out or deps


def stage_mask(names):
    """Trace stage list (sync1 or sync2 names) -> frozenset of short names, or
    None (= all stages) when the trace holds bits this model does not know."""
    if names is None:
        return None
    out = set()
    for n in names:
        if isinstance(n, int):
            return None
        s = _strip(n, r"^VK_PIPELINE_STAGE_(2_)?")
        if s == "NONE":
            continue
        if s == "PRE_RASTERIZATION_SHADERS":
            out.update(_PRE_RASTER)
        else:
            out.add(_STAGE_ALIAS.get(s, s))
    return frozenset(out)


def access_mask(names):
    if names is None:
        return None
    out = set()
    for n in names:
        if isinstance(n, int):
            return None
        s = _strip(n, r"^VK_ACCESS_(2_)?")
        if s != "NONE":
            out.add(s)
    return frozenset(out)


_RT = ("RAY_TRACING_SHADER",)
_SHADER_STAGE = {"VERTEX": ("VERTEX_SHADER",), "FRAGMENT": ("FRAGMENT_SHADER",),
                 "COMPUTE": ("COMPUTE_SHADER",), "GEOMETRY": ("GEOMETRY_SHADER",),
                 "TESSELLATION_CONTROL": ("TESSELLATION_CONTROL_SHADER",),
                 "TESSELLATION_EVALUATION": ("TESSELLATION_EVALUATION_SHADER",),
                 "TASK": ("TASK_SHADER",), "MESH": ("MESH_SHADER",),
                 "RAYGEN": _RT, "ANY_HIT": _RT, "CLOSEST_HIT": _RT, "MISS": _RT, "INTERSECTION": _RT,
                 "CALLABLE": _RT,
                 "ALL_GRAPHICS": _PRE_RASTER + ("FRAGMENT_SHADER",),
                 "ALL": _PRE_RASTER + ("FRAGMENT_SHADER", "COMPUTE_SHADER") + _RT}
_BIND_POINT = {"draw": "VK_PIPELINE_BIND_POINT_GRAPHICS", "dispatch": "VK_PIPELINE_BIND_POINT_COMPUTE",
               "trace": "VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR"}
_MESH_DRAWS = ("vkCmdDrawMeshTasksEXT", "vkCmdDrawMeshTasksIndirectEXT", "vkCmdDrawMeshTasksIndirectCountEXT")
_DRAWS = ("vkCmdDraw", "vkCmdDrawIndexed", "vkCmdDrawIndirect", "vkCmdDrawIndexedIndirect",
          "vkCmdDrawIndirectCount", "vkCmdDrawIndexedIndirectCount") + _MESH_DRAWS
_DISPATCHES = ("vkCmdDispatch", "vkCmdDispatchIndirect", "vkCmdDispatchBase")
_TRACES = ("vkCmdTraceRaysKHR", "vkCmdTraceRaysIndirectKHR", "vkCmdTraceRaysIndirect2KHR")
_AS_CMDS = ("vkCmdBuildAccelerationStructuresKHR", "vkCmdBuildAccelerationStructuresIndirectKHR",
            "vkCmdCopyAccelerationStructureKHR", "vkCmdCopyAccelerationStructureToMemoryKHR",
            "vkCmdCopyMemoryToAccelerationStructureKHR", "vkCmdWriteAccelerationStructuresPropertiesKHR")
_AS_HOST = ("vkBuildAccelerationStructuresKHR", "vkCopyAccelerationStructureKHR",
            "vkCopyAccelerationStructureToMemoryKHR", "vkCopyMemoryToAccelerationStructureKHR",
            "vkWriteAccelerationStructuresPropertiesKHR")
AS_BUILD, AS_COPY = "ACCELERATION_STRUCTURE_BUILD", "ACCELERATION_STRUCTURE_COPY"
AS_READ, AS_WRITE = "ACCELERATION_STRUCTURE_READ", "ACCELERATION_STRUCTURE_WRITE"
# commands recorded only as "unimplemented" whose missing effect cannot hide a
# hazard: queries (results are not memory the model tracks), debug markers
_NO_MEMORY_EFFECT = {"vkCmdBeginQuery", "vkCmdEndQuery", "vkCmdResetQueryPool", "vkCmdWriteTimestamp",
                     "vkCmdWriteTimestamp2", "vkCmdBeginQueryIndexedEXT", "vkCmdEndQueryIndexedEXT",
                     "vkCmdDebugMarkerBeginEXT", "vkCmdDebugMarkerEndEXT", "vkCmdDebugMarkerInsertEXT",
                     "vkDebugMarkerSetObjectNameEXT", "vkDebugMarkerSetObjectTagEXT",
                     "vkCmdBeginDebugUtilsLabelEXT", "vkCmdEndDebugUtilsLabelEXT", "vkCmdInsertDebugUtilsLabelEXT",
                     "vkCmdWriteTimestamp2KHR"}
# host calls the model driver may only stub that touch no memory the model tracks
# and synchronize nothing (queries, object creation/destruction, debug annotations)
_HOST_NO_MEMORY_EFFECT = ("vkGet", "vkEnumerate", "vkCreate", "vkDestroy", "vkDebug", "vkSetDebug", "vkSubmitDebug",
                          "vkQueueBeginDebug", "vkQueueEndDebug", "vkQueueInsertDebug", "vkSetPrivateData",
                          "vkTrimCommandPool", "vkMergePipelineCaches", "vkAcquireProfilingLock",
                          "vkReleaseProfilingLock", "vkResetQueryPool", "vkSetHdrMetadata", "vkDeferredOperationJoin",
                          "vkCompileDeferred", "vkSetLatency", "vkLatencySleep", "vkQueueNotifyOutOfBand")


def shader_stages(names):
    """['VK_SHADER_STAGE_VERTEX_BIT', ...] -> pipeline stage names."""
    out = []
    for n in names or ():
        for st in _SHADER_STAGE.get(_strip(n, r"^VK_SHADER_STAGE_"), ()):
            if st not in out:
                out.append(st)
    return out


_DESC_FLAG = {
    "VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER": "UNIFORM_READ",
    "VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC": "UNIFORM_READ",
    "VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK": "UNIFORM_READ",
    "VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE": "SHADER_SAMPLED_READ",
    "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER": "SHADER_SAMPLED_READ",
    "VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER": "SHADER_SAMPLED_READ",
    "VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT": "INPUT_ATTACHMENT_READ",
}

_READ_TYPES = {"VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER", "VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC",
               "VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE", "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER",
               "VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER", "VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT"}
_STORAGE_TYPES = {"VK_DESCRIPTOR_TYPE_STORAGE_BUFFER", "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC",
                  "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE", "VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER"}
# descriptor-array indexing: descriptor types whose array elements may be narrowed to the decoded indices
# (read-only image descriptors; the widened reads then never touch a buffer, see timing.check)
_DYN_TYPES = {"VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE", "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER"}
_CHUNK = 4096                       # granularity of the rebuilt host-visible memory contents
_MAX_DECODE = 1 << 16               # most vertices / instances decoded per draw
_INT_FORMATS = {"VK_FORMAT_R32_UINT", "VK_FORMAT_R32_SINT"}


def spirv_readonly_bindings(code):
    """{(set, binding): read_only} for the resources a SPIR-V module declares;
    read_only means NonWritable on the variable or on every member of its block."""
    if len(code) < 20:
        return {}
    words = struct.unpack("<%dI" % (len(code) // 4), code[: len(code) // 4 * 4])
    if words[0] != 0x07230203:
        return {}
    deco, member_nw, members, ptr, arr, var = {}, {}, {}, {}, {}, {}
    i = 5
    while i < len(words):
        op, n = words[i] & 0xFFFF, words[i] >> 16
        if n == 0:
            break
        a = words[i + 1:i + n]
        if op == 71 and len(a) >= 2:                      # OpDecorate
            deco.setdefault(a[0], {})[a[1]] = a[2] if len(a) > 2 else True
        elif op == 72 and len(a) >= 3 and a[2] == 24:     # OpMemberDecorate NonWritable
            member_nw.setdefault(a[0], set()).add(a[1])
        elif op == 30:                                    # OpTypeStruct
            members[a[0]] = len(a) - 1
        elif op == 32:                                    # OpTypePointer
            ptr[a[0]] = a[2]
        elif op in (28, 29):                              # OpTypeArray / RuntimeArray
            arr[a[0]] = a[1]
        elif op == 59:                                    # OpVariable
            var[a[1]] = a[0]
        i += n
    out = {}
    for v, ty in var.items():
        d = deco.get(v, {})
        if 34 not in d or 33 not in d:
            continue
        ro = 24 in d
        if not ro:
            t = ptr.get(ty)
            while t in arr:
                t = arr[t]
            if t in members and members[t] and len(member_nw.get(t, ())) == members[t]:
                ro = True
        key = (d[34], d[33])
        out[key] = out.get(key, True) and ro
    return out


# ------------------------------------------------------------------ queue family ownership
class Ownership:
    """Queue family ownership of VK_SHARING_MODE_EXCLUSIVE resources, per
    element interval (bytes of a buffer; aspect-slot * layers + layer of one
    image mip), as resources.adoc defines it per buffer range / image
    subresource. Per element it keeps the family whose write last defined the
    contents (host writes and never-written elements: none) and the pending
    release halves.

    * a write on family f takes ownership of what it writes (an implicit
      acquire: the old contents are discarded, which the spec allows);
    * a read on family f of elements last written on another family g, with no
      release g->f / acquire in between, reads undefined contents: reported;
    * an acquire g->f of a range is matched when release halves g->f cover all
      of it; either way f then owns the range.
    covenant/timing_ref.py (ownership_ref) is the per-element reference."""
    BIG = 1 << 62

    def __init__(self):
        self.w = {}         # res -> sorted disjoint [lo, hi, family]
        self.rel = {}       # res -> sorted disjoint [lo, hi, (src family, dst family, where)]

    @classmethod
    def _iv(cls, region):
        return [(0, cls.BIG)] if region is None else [(lo, hi) for lo, hi in region if lo < hi]

    @staticmethod
    def _pieces(lst, lo, hi):
        return [(max(a, lo), min(b, hi), v) for a, b, v in lst if a < hi and lo < b]

    @staticmethod
    def _assign(lst, lo, hi, v):
        out = []
        for a, b, x in lst:
            if b <= lo or hi <= a:
                out.append([a, b, x])
                continue
            if a < lo:
                out.append([a, lo, x])
            if hi < b:
                out.append([hi, b, x])
        if v is not None:
            out.append([lo, hi, v])
        out.sort()
        return out

    def _set(self, m, res, region, v):
        lst = m.get(res, [])
        for lo, hi in self._iv(region):
            lst = self._assign(lst, lo, hi, v)
        m[res] = lst

    def use(self, res, region, fam, access):
        """-> [(lo, hi, family)] read elements last written on another family."""
        if access == "write":
            self._set(self.w, res, region, fam)
            return []
        bad = [p for lo, hi in self._iv(region) for p in self._pieces(self.w.get(res, []), lo, hi)
               if p[2] != fam]
        for lo, hi, _ in bad:               # reported once: the read's family now holds them
            self.w[res] = self._assign(self.w.get(res, []), lo, hi, fam)
        return bad

    def release(self, res, region, sf, df, where):
        self._set(self.rel, res, region, (sf, df, where))

    def released(self, res, region, sf, df):
        """Do release halves sf->df cover all of region?"""
        for lo, hi in self._iv(region):
            got = sorted(p for p in self._pieces(self.rel.get(res, []), lo, hi) if p[2][:2] == (sf, df))
            x = lo
            for a, b, _ in got:
                if a > x:
                    return False
                x = max(x, b)
            if x < hi:
                return False
        return True

    def acquire(self, res, region, sf, df):
        """-> matched; the range is then owned by df either way."""
        ok = self.released(res, region, sf, df)
        self._set(self.rel, res, region, None)
        self._set(self.w, res, region, df)
        return ok


# ------------------------------------------------------------------ loader
class _Loader:
    def __init__(self, records, spirv_dir=None):
        self.recs = records
        self.spirv_dir = spirv_dir
        self.m = HostModel(records=records)
        self.fr = Frame()
        self.fr.q("host")
        # object state
        self.view_image = {}
        self.buffer_views = {}
        self.pipeline_layouts = {}
        self.push_ranges = {}       # pipeline layout -> [(stages, offset, size)]
        self.push_checked = set()
        self.set_layout_of = {}
        self.set_contents = {}
        self.cb_cmds = {}
        self.cb_pool = {}
        self.render_passes = {}
        self.framebuffers = {}
        self.module_sha = {}
        self.access_cache = {}
        self.mode_cache = {}      # pipeline -> {(set, binding): spirv.entry_binding_modes value}
        self.stage_cache = {}     # pipeline -> {(set, binding): {stage: written}}, or None
        # sync state
        self.sem_type = {}
        self.sem_init = {}
        self.range_cache = {}     # pipeline -> (covenant.ranges exprs per binding, local size)
        self.module_cache = {}
        self.narrowed = 0         # descriptor accesses narrowed by covenant.ranges
        self.fresh_bind = {}      # memory -> objects bound since its last host_write record
        from . import modes as _modes
        self.capture = _modes.host_capture()      # "conservative" (default) or "observed"
        self.host_write_records = self.same_value_records = self.cross_thread_records = 0
        self.multi_thread_records = 0      # records made while the process had >1 thread
        self.next_rec = {r.get("seq"): r for r in records if r.get("fn") in
                         ("vkBindBufferMemory", "vkBindImageMemory")}   # seq -> bind record (page-straddle disambiguation)
        self.straddle = 0         # host-written pages attributed to fresh sub-allocations only
        self.bin_sem = {}         # sem -> [signal count, [pending names, oldest first], waits awaiting a future signal]
        self.tl_signals = {}      # sem -> [values]
        self.tl_waits = []        # (queue name, Wait op, sem, value)
        self.fences = {}          # fence -> dict(gen, signal, presignaled)
        self.last_present = {}    # image name -> signal name
        self.present_wait = {}    # binary sem -> (image, where) of the present that waited on it
        self.present_count = 0
        self.queue_used = {}      # queue name -> set of resources
        self.exclusive = set()    # resource names created VK_SHARING_MODE_EXCLUSIVE
        self.own = Ownership()    # exclusive resources: per-interval queue family ownership
        self.batch_releases = []  # (resource, where) of release barriers in the current batch
        self.mips = {}            # image id -> mip level count
        self.view_range = {}      # view id -> (base mip, level count)
        # subresource granularity (timing regions): images are tracked per mip
        # resource with elements aspect_slot * array_layers + layer; buffers by byte
        self.layers = {}          # image id -> array layer count
        self.img_format = {}      # image id -> format
        self.view_sub = {}        # view id -> (aspects, base layer, layer count)
        self.img_samples = {}     # image id -> sample count
        self.legacy_rp_flags = 0  # subpass dependencies without recorded dependencyFlags
        self.buf_size = {}        # buffer id -> size in bytes
        self.buffer_view_range = {}   # buffer view id -> (offset, range)
        # VK_KHR_acceleration_structure: TLAS instance references the driver read
        # from the instance buffer at submit, per (cb, build command seq, build index);
        # None = some reference could not be resolved
        self.tlas_rec = {}
        self.as_live = set()      # acceleration structures not yet destroyed
        self.caps_cache = {}      # shader sha1 -> set of SPIR-V capabilities
        self.host_events = set()  # events vkSetEvent has signalled from the host
        # in-use object rules: command buffers submitted, fences submitted with work,
        # binary semaphores waited on by a GPU queue
        self.cb_submitted, self.fence_used, self.sem_gpu_waited = set(), set(), set()
        # descriptor-array indexing: host-visible memory contents rebuilt from host_write records
        # (mapped memory starts zeroed in the model driver; "writes" are the changed runs)
        self.desc_refine = os.environ.get("COVENANT_NO_DESC_REFINE", "") in ("", "0")
        self.mem_chunks = {}      # memory -> {chunk index: bytearray(_CHUNK)}
        self.mem_hostw = {}       # memory -> set of chunk indices in pages the host wrote
        self.mem_unknown = {}     # memory -> [(lo, hi)] bytes whose content was not recorded
        self.mem_bad = {}         # memory -> why its recorded contents cannot be trusted
        self.capture_bad = None   # why no recorded contents can be trusted (capture warning)
        self.index_cache = {}     # pipeline -> desc_index.pipeline_index_terms result
        self.guards = []          # (IndexGuard, memory, lo, hi) of every decoded draw
        self.desc_refined = self.desc_widened = 0

    # -- names -----------------------------------------------------------
    def prepass(self):
        debug = {}
        for r in self.recs:
            if r["fn"] == "vkSetDebugUtilsObjectNameEXT" and r.get("object") and r.get("name"):
                debug[r["object"]] = r["name"]
        def group(h):
            # images and buffers share the resource namespace; views etc. have their own
            k = h.split("#")[0]
            return "resource" if k in ("image", "buffer") else k
        counts = {}
        for h, n in debug.items():
            counts[(group(h), n)] = counts.get((group(h), n), 0) + 1
        names = {}
        for h, n in debug.items():
            names[h] = n if counts[(group(h), n)] == 1 else f"{n}({h})"
        first_swapchain = True
        for r in self.recs:
            if r["fn"] == "vkCreateSwapchainKHR":
                sid = r["swapchain"]
                if sid not in names:
                    names[sid] = "swapchain" if first_swapchain else sid
                first_swapchain = False
                for i, img in enumerate(r["images"]):
                    if img not in names:
                        names[img] = f"{names[sid]}[{i}]"
        self.m.names = names

    def n(self, hid):
        return self.m.names.get(hid, hid)

    # -- frame helpers -----------------------------------------------------
    def q(self, name):
        return self.fr.q(name)

    def use(self, qn, res, access, layout=None, where="", stage=None, flag=None, scope=None,
            region=None, cmd=None, instance=None, pass_op=None, atomic=None, fb=None):
        """res: one resource name or a list (e.g. the mip levels of an image),
        or a list of (name, region) pairs. region: elements of each resource
        (None = all); cmd: the command the access belongs to (accesses of one
        command are not checked against each other)."""
        if not res:
            return
        for r in ([res] if isinstance(res, str) else res):
            reg = region
            if isinstance(r, tuple):
                r, reg = r
            self.q(qn).append(Use(r, access, layout, where, stage=stage, flag=flag, scope=scope,
                                  region=reg, cmd=cmd, instance=instance, pass_op=pass_op, atomic=atomic,
                                  fb=fb))
            self.queue_used.setdefault(qn, set()).add(r)
            self.check_owner(qn, r, access, where, reg)

    def check_owner(self, qn, r, access, where, region=None):
        """VK_SHARING_MODE_EXCLUSIVE: contents written on one queue family are
        defined on another only after a release/acquire pair (per range: Ownership)."""
        fam = self.queue_family.get(qn)
        if r not in self.exclusive or fam is None:
            return
        bad = self.own.use(r, region, fam, access)
        if bad:
            lo, hi, own = bad[0]
            part = "" if (lo, hi) == (0, Ownership.BIG) else f" (elements [{lo}, {hi}))"
            self.fr.findings.append((
                "timing", "timing.ownership-missing",
                f"'{r}'{part} was last written on queue family {own} but is read on '{qn}' "
                f"(family {fam}) with no ownership transfer, so its contents are undefined there.",
                where, ["release it on the owning queue and acquire it on this one, "
                        "or create it with VK_SHARING_MODE_CONCURRENT"]))

    # -- subresources: images with more than one mip level are tracked per level
    def img_res(self, img, base=0, count=None):
        """Resource names of mip levels [base, base+count) of image id `img`
        ("<image>.mip<k>"; plain "<image>" for single-mip images)."""
        if not img:
            return []
        name, levels = self.n(img), self.mips.get(img, 1)
        if levels <= 1:
            return [name]
        if count is None or count == 0 or count >= 0xFFFFFFFF:
            count = levels - base
        return [f"{name}.mip{k}" for k in range(base, min(levels, base + count))]

    def view_res(self, view):
        img, _ = self.resolve_image(view)
        base, count = self.view_range.get(view, (0, None))
        return self.img_res(img, base, count)

    def layers_res(self, img, regions, key):
        mips = sorted({(r.get(key) or {}).get("mip", 0) for r in regions or []}) if key else []
        if not mips or self.mips.get(img, 1) <= 1:
            return self.img_res(img)
        return [x for k in mips for x in self.img_res(img, k, 1)]

    def ranges_res(self, img, ranges):
        if not ranges:
            return self.img_res(img)
        out = []
        for r in ranges:
            for x in self.img_res(img, r.get("base_mip", 0), r.get("level_count")):
                if x not in out:
                    out.append(x)
        return out

    # -- subresource regions (timing.Use.region / Barrier.region)
    def _slots(self, img):
        """Number of aspect slots of an image (depth+stencil: 2; planes: n)."""
        f = str(self.img_format.get(img, ""))
        if "_S8_UINT" in f and f.startswith("VK_FORMAT_D"):
            return 2
        m = re.search(r"_([23])PLANE_", f)
        return int(m.group(1)) if m else 1

    def img_region(self, img, aspects=None, base_layer=0, layer_count=None):
        """Element region of every mip resource of `img` for these aspects and
        array layers (None = the whole mip)."""
        L, n = max(1, self.layers.get(img, 1)), self._slots(img)
        base_layer = base_layer or 0
        if layer_count in (None, 0, "remaining") or layer_count >= 0xFFFFFFFF:
            layer_count = L - base_layer
        lo, hi = max(0, base_layer), min(L, base_layer + layer_count)
        slots = set()
        for a in aspects or []:
            k = _strip(a, r"^VK_IMAGE_ASPECT_")
            if k in ("COLOR", "DEPTH", "PLANE_0", "MEMORY_PLANE_0"):
                slots.add(0)
            elif k in ("STENCIL", "PLANE_1", "MEMORY_PLANE_1"):
                slots.add(1 if n > 1 else 0)
            elif k in ("PLANE_2", "MEMORY_PLANE_2"):
                slots.add(2 if n > 2 else 0)
            else:
                slots = set(range(n))       # METADATA etc.: be conservative
                break
        if not slots:
            slots = set(range(n))
        if lo >= hi:
            return ()
        if lo == 0 and hi >= L and len(slots) >= n:
            return None
        out = []
        for sl in sorted(slots):
            a, b = sl * L + lo, sl * L + hi
            if out and out[-1][1] == a:
                out[-1] = (out[-1][0], b)
            else:
                out.append((a, b))
        return tuple(out)

    def img_parts(self, img, subs, mip_key="mip"):
        """[(mip resource name, region)] for a list of subresource dicts
        ({aspect, mip | base_mip+level_count, base_layer, layer_count})."""
        if not img:
            return []
        if not subs:
            return [(x, None) for x in self.img_res(img)]
        per = {}
        for r in subs:
            if mip_key == "mip":
                names = self.img_res(img, r.get("mip", 0), 1) if self.mips.get(img, 1) > 1 \
                    else self.img_res(img)
            else:
                names = self.img_res(img, r.get("base_mip", 0), r.get("level_count"))
            reg = self.img_region(img, r.get("aspect"), r.get("base_layer", 0), r.get("layer_count"))
            for x in names:
                per[x] = _region_union(per.get(x, ()), reg) if x in per else reg
        return list(per.items())

    def fb_token(self, view, input_attachment=False):
        """The framebuffer position token of an attachment access
        through `view` (Use.fb): the image and the view's base mip and layer, which fix
        how framebuffer (x, y, layer) coordinates map to texels. An input attachment
        read of a multisampled image may read any sample (subpassLoad with a sample
        index), which BY_REGION does not order: no token."""
        img, _ = self.resolve_image(view)
        if not img:
            return None
        if input_attachment and (self.img_samples.get(img, 1) or 1) > 1:
            return None
        return ("fb", self.n(img), self.view_range.get(view, (0, None))[0],
                self.view_sub.get(view, (None, 0, None))[1])

    def view_parts(self, view, aspects=None):
        """[(mip resource, region)] a view covers (aspects narrows the view's)."""
        img, _ = self.resolve_image(view)
        asp, bl, lc = self.view_sub.get(view, (None, 0, None))
        if aspects:
            asp = aspects
        reg = self.img_region(img, asp, bl, lc) if img else None
        return [(x, reg) for x in self.view_res(view)]

    def buf_region(self, buf, offset=0, size=None):
        """Byte region [offset, offset+size) of buffer id `buf` (None = whole)."""
        total = self.buf_size.get(buf)
        offset = offset or 0
        if size in (None, "whole") or (isinstance(size, int) and size >= 0xFFFFFFFFFFFFFFFF):
            if total is None:
                return None if offset == 0 else ((offset, 1 << 62),)
            size = total - offset
        if total is not None and offset <= 0 and offset + size >= total:
            return None
        if size <= 0:
            return ()
        return ((offset, offset + size),)

    def sem_signal(self, sem, value, where=""):
        if self.sem_type.get(sem) == "timeline":
            vals = self.tl_signals.setdefault(sem, [])
            v = value or 0
            name = f"{self.n(sem)}={value}"
            if vals and v <= max(vals) or v <= self.sem_init.get(sem, 0):
                # VUID-VkSubmitInfo2-semaphore-03882 / VUID-VkSemaphoreSignalInfo-value-03258:
                # a timeline signal must be greater than the current and every pending value.
                # A repeated value gets a name of its own, so a wait stays bound to the first
                # signal that reaches its value (resolve_timeline), the one that orders the least
                self.fr.findings.append((
                    "timing", "timing.timeline-value",
                    f"timeline semaphore '{self.n(sem)}' is signaled with value {v}, which is not greater "
                    f"than an earlier signaled value ({max(vals + [self.sem_init.get(sem, 0)])}): a wait for "
                    f"{v} may be satisfied by the earlier signal.", where,
                    ["timeline signal values must strictly increase (VUID-VkSubmitInfo2-semaphore-03882)"]))
                if v in vals:
                    name = f"{name}#{vals.count(v)}"
            vals.append(v)
            return name
        if sem in self.present_wait:
            # a present's wait finishes at an unknown time; only re-acquiring
            # the presented image proves the semaphore is free again
            img, pwhere = self.present_wait.pop(sem)
            self.fr.semaphore_reuse.append((self.n(sem), img, pwhere, where))
        st = self.bin_sem.setdefault(sem, [0, [], 0])
        st[0] += 1
        name = f"{self.n(sem)}:{st[0]}"
        if st[2]:
            st[2] -= 1              # an earlier wait already named this instance
            return name
        if st[1]:
            # VUID-vkQueueSubmit-pSignalSemaphores-00067 / VUID-vkQueueSubmit2-semaphore-03868 /
            # VUID-vkAcquireNextImageKHR-semaphore-01286: a binary semaphore must not be signaled
            # while a signal is pending. Which signal a later wait consumes is then undefined; the
            # wait is bound to the older one (sem_wait), the choice that orders the least.
            self.fr.findings.append((
                "timing", "timing.semaphore-resignal",
                f"binary semaphore '{self.n(sem)}' is signaled again while an earlier signal is still "
                "pending (nothing has waited on it), so a later wait may consume the stale signal.",
                where, [f"pending signal: {st[1][0]}",
                        "wait on (or drop) every binary semaphore signal before signaling it again; "
                        "VUID-vkQueueSubmit-pSignalSemaphores-00067, VUID-vkAcquireNextImageKHR-semaphore-01286"]))
        st[1].append(name)
        return name

    def sem_wait(self, qn, sem, value, where="", stages=None):
        if self.sem_type.get(sem) == "timeline":
            op = Wait(f"{self.n(sem)}>={value}", where, stages)
            self.q(qn).append(op)
            self.tl_waits.append((qn, op, sem, value or 0))
            return
        st = self.bin_sem.setdefault(sem, [0, [], 0])
        if st[1]:
            name = st[1].pop(0)     # FIFO: the oldest pending signal
        else:
            st[2] += 1              # wait before signal: the next unclaimed future instance
            name = f"{self.n(sem)}:{st[0] + st[2]}"
        self.q(qn).append(Wait(name, where, stages))
        if qn != "host" and not qn.startswith("present:"):
            # the wait operation is pending until it executes: an acquire may reuse the
            # semaphore only after the host knows it has (VUID-vkAcquireNextImageKHR-semaphore-01779)
            self.sem_gpu_waited.add(sem)
            self.use(qn, f"semaphore:{self.n(sem)}", "read", None, f"{where} (the wait operation)",
                     None, "SEM_WAIT")

    def cb_reset(self, cb, where):
        if cb in self.cb_submitted:
            self.use("host", f"cmdbuf:{self.n(cb)}", "write", None, where, "HOST", "HOST_CB_RESET")

    def fence_signal_name(self, fence):
        f = self.fences.setdefault(fence, {"gen": 0, "signal": None, "presignaled": False})
        f["signal"] = f"{self.n(fence)}@{f['gen']}"
        return f["signal"]

    # -- object records ------------------------------------------------------
    def resolve_image(self, view):
        return self.view_image.get(view, (None, None))

    def bound_of(self, desc_type, el):
        """-> (Bound, resource name(s) to use, layout)

        Samplers, inline uniform blocks and acceleration structures are bound
        (Bound.resource names them) but are no memory the timing checker
        tracks: the inline block lives in the set (the "set:" use covers it)."""
        if desc_type == "VK_DESCRIPTOR_TYPE_SAMPLER":
            smp = el.get("sampler")
            return Bound(desc_type, self.n(smp) if smp else "immutable sampler", None), None, None
        if desc_type == "VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK":
            return Bound(desc_type, "inline uniform block", None), None, None
        if (desc_type or "").startswith("VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE"):
            acc = el.get("acceleration_structure")
            return Bound(desc_type, self.n(acc) if acc else "acceleration structure", None), None, None
        if "buffer" in el:
            b = el["buffer"]
            return Bound(desc_type, self.n(b) if b else None, None), (self.n(b) if b else None), None
        if "texel_view" in el:
            bv = self.buffer_views.get(el["texel_view"])
            res = self.n(bv[0]) if bv else None
            return Bound(desc_type, res, bv[1] if bv else None), res, None
        if "view" in el:
            img, fmt = self.resolve_image(el.get("view"))
            res = self.n(img) if img else None
            return Bound(desc_type, res, fmt), self.view_res(el.get("view")), short_layout(el.get("layout"))
        return Bound(desc_type, None, None), None, None

    def shader_gap(self, pipeline, why, what="shader code not available"):
        """A pipeline whose shader code is not available, or whose accesses the
        provenance analysis cannot attribute to descriptors: fail closed. Its storage
        descriptors are taken as written by every stage, no access is atomic, and the
        gap is recorded with effect "untracked" (the code may also reach memory through
        device addresses), so the check is not guarantee-eligible."""
        self.note_unsupported(what, f"pipeline {self.n(pipeline)}", "untracked",
                              f"{why}: storage accesses taken as writes; the verdict does not cover "
                              "memory the shader may reach otherwise")

    def access_for(self, pipeline):
        """{(set, binding): written} over the pipeline's entry points, from the
        dumped SPIR-V; bindings no entry reaches are absent. None if unknown."""
        if pipeline in self.access_cache:
            return self.access_cache[pipeline]
        out, modes = None, {}
        per_stage = {}          # (set, binding) -> {pipeline stage: written}
        p = self.m.pipelines.get(pipeline)
        if p and not self.spirv_dir:
            self.shader_gap(pipeline, "no SPIR-V directory was given")
        if p and self.spirv_dir:
            out = {}
            for stg_names, sha, entry in p.stages:
                path = os.path.join(self.spirv_dir, f"{sha}.spv")
                got, why, what = None, f"module {sha[:12]} ({stg_names}) is missing", "shader code not available"
                if os.path.exists(path):
                    try:
                        mod = _spirv.parse(path)
                        got = _spirv.entry_bindings(mod, entry)
                        md = _spirv.entry_binding_modes(mod, entry, spans=True) if got else {}
                    except _spirv.AnalysisIncomplete as ex:
                        got, what = None, "shader accesses not attributed"
                        why = f"module {sha[:12]} ({stg_names}): {ex}"
                    except Exception as ex:     # unknown opcodes, truncated module, ...
                        got, why = None, f"module {sha[:12]} ({stg_names}) cannot be read: {type(ex).__name__}"
                if got is None:
                    self.shader_gap(pipeline, why, what)
                    out, modes, per_stage = None, {}, {}
                    break
                for k, w in got.items():
                    out[k] = out.get(k, False) or w
                    for ps in shader_stages(stg_names.split("|")):
                        d = per_stage.setdefault(k, {})
                        d[ps] = d.get(ps, False) or w
                    # atomic-only / slot bindings: every stage touching it must qualify
                    if k in modes and modes[k] is None or k not in md:
                        modes[k] = None
                    else:
                        modes[k] = modes.get(k, frozenset()) | md[k]
        self.access_cache[pipeline] = out
        self.mode_cache[pipeline] = {k: v for k, v in modes.items() if v}
        self.stage_cache[pipeline] = per_stage if out is not None else None
        return out

    # -- command interpretation ------------------------------------------------
    def expand(self, cb, qn, submit_seq, st=None, depth=0):
        if depth > 4:
            return
        if st is None:
            st = {"pipeline": {}, "sets": {}, "vb": {}, "ib": None, "att": [], "rp": None, "labels": [],
                  "depth_compare": None, "depth_test": None, "depth_write": None, "depth_clear": None,
                  "viewport_depth": None}
        family = None
        cbname = self.n(cb)
        for c in self.cb_cmds.get(cb, []):
            fn = c["fn"]
            label = f" [{st['labels'][-1]}]" if st["labels"] else ""
            where = f"{fn} @seq {c['seq']}{label} ({cbname}, submitted @seq {submit_seq} on {qn})"
            if fn == "vkCmdBindPipeline":
                st["pipeline"][c["bind_point"]] = c["pipeline"]
            elif fn == "vkCmdBindDescriptorSets":
                d = st["sets"].setdefault(c["bind_point"], {})
                bl = st.setdefault("set_bind_layout", {}).setdefault(c["bind_point"], {})
                for i, s in enumerate(c["sets"]):
                    d[c["first_set"] + i] = s
                    bl[c["first_set"] + i] = (c.get("layout"), where)
                self.bind_dynamic(st, c)
            elif fn == "vkCmdPushDescriptorSetKHR":
                # push descriptors: a set whose contents live in the command buffer
                sid = f"push descriptors @seq {c['seq']}"
                self.m.names.setdefault(sid, sid)
                lays = self.pipeline_layouts.get(c.get("layout"), [])
                if c["set"] < len(lays):
                    self.set_layout_of[sid] = lays[c["set"]]
                contents = self.set_contents[sid] = {}
                for w in c.get("writes", []):
                    for i, el in enumerate(w["elements"]):
                        contents[(w["binding"], w["array_element"] + i)] = (w["type"], el)
                st["sets"].setdefault(c["bind_point"], {})[c["set"]] = sid
                st.setdefault("set_bind_layout", {}).setdefault(c["bind_point"], {})[c["set"]] = (c.get("layout"), where)
                st.setdefault("dyn", {}).setdefault(c["bind_point"], {})
                for key in [x for x in st["dyn"][c["bind_point"]] if x[0] == c["set"]]:
                    del st["dyn"][c["bind_point"]][key]
            elif fn == "vkCmdBindVertexBuffers":
                offs = c.get("offsets") or []
                for i, b in enumerate(c["buffers"]):
                    st["vb"][c["first_binding"] + i] = b
                    st.setdefault("vbo", {})[c["first_binding"] + i] = offs[i] if i < len(offs) else 0
            elif fn == "vkCmdBindIndexBuffer":
                st["ib"] = c["buffer"]
                st["ibo"] = c.get("offset", 0)
                st["ibt"] = {"VK_INDEX_TYPE_UINT8_EXT": 1, "VK_INDEX_TYPE_UINT8_KHR": 1, "VK_INDEX_TYPE_UINT8": 1,
                             "VK_INDEX_TYPE_UINT16": 2}.get(c.get("index_type"), 4)
            elif fn == "vkCmdSetDepthCompareOp":
                st["depth_compare"] = c.get("op")
            elif fn == "vkCmdSetDepthTestEnable":
                st["depth_test"] = c.get("enable")
            elif fn == "vkCmdSetDepthWriteEnable":
                st["depth_write"] = c.get("enable")
            elif fn == "vkCmdSetStencilTestEnable":
                st["stencil_test"] = c.get("enable")
            elif fn == "vkCmdSetViewport":
                if c.get("viewports"):
                    st["viewport_depth"] = tuple(c["viewports"][0][4:6])
                    st["viewport"] = tuple(c["viewports"][0][:4])
            elif fn == "vkCmdBeginDebugUtilsLabelEXT":
                st["labels"].append(c.get("label") or "")
            elif fn == "vkCmdEndDebugUtilsLabelEXT":
                if st["labels"]:
                    st["labels"].pop()
            elif fn == "vkCmdBeginRendering":
                att = []
                st["catt"] = {}
                for k, a in enumerate(c.get("color") or []):
                    n0 = len(att)
                    self._att(att, "color", a)
                    if len(att) > n0:
                        st["catt"][k] = att[n0]     # for vkCmdClearAttachments
                for role in ("depth", "stencil"):
                    if c.get(role):
                        self._att(att, role, c[role])
                d = c.get("depth") or {}
                st["depth_clear"] = (d.get("clear") or {}).get("depth") if d.get("view") else None
                st["clears"] = {self.n(self.resolve_image(a["view"])[0]): (a.get("clear") or {}).get("color")
                                for a in c.get("color") or [] if a.get("view") and self.resolve_image(a["view"])[0]
                                and a.get("load") == "VK_ATTACHMENT_LOAD_OP_CLEAR"}
                st["att"] = att
                st["scope"] = object()
                st["dr"] = self.rendering_atts(c)
                st["in_pass"] = True
                self.load_store(qn, st["dr"], True, where, st["scope"])
            elif fn == "vkCmdEndRendering":
                self.load_store(qn, st.get("dr") or [], False, where, st.get("scope"))
                st["att"] = []
                st["dr"] = []
                st["in_pass"] = False
            elif fn == "vkCmdBeginRenderPass":
                self.begin_render_pass(c, st, qn, where)
            elif fn == "vkCmdNextSubpass":
                self.next_subpass(st, qn, where)
            elif fn == "vkCmdEndRenderPass":
                self.end_render_pass(st, qn, where)
            elif fn in _DRAWS:
                self.draw(c, st, qn, submit_seq, cbname, where, "draw")
            elif fn in _DISPATCHES:
                self.draw(c, st, qn, submit_seq, cbname, where, "dispatch")
            elif fn in _TRACES:
                self.draw(c, st, qn, submit_seq, cbname, where, "trace")
            elif fn in _AS_CMDS:
                self.accel_cmd(c, cb, qn, where)
            elif fn in ("vkCmdCopyBuffer", "vkCmdCopyBufferToImage", "vkCmdCopyImageToBuffer",
                        "vkCmdCopyImage", "vkCmdBlitImage"):
                skey = {"vkCmdCopyImageToBuffer": "image", "vkCmdCopyImage": "src", "vkCmdBlitImage": "src"}.get(fn)
                dkey = {"vkCmdCopyBufferToImage": "image", "vkCmdCopyImage": "dst", "vkCmdBlitImage": "dst"}.get(fn)
                cmd = object()
                src = self.copy_parts(fn, c, "src", skey)
                dst = self.copy_parts(fn, c, "dst", dkey)
                xst = "BLIT" if fn == "vkCmdBlitImage" else "COPY"      # sync2 stage of the command
                self.use(qn, src, "read", short_layout(c.get("src_layout")), where,
                         stage=xst, flag="TRANSFER_READ", cmd=cmd)
                self.use(qn, dst, "write", short_layout(c.get("dst_layout")), where,
                         stage=xst, flag="TRANSFER_WRITE", cmd=cmd)
            elif fn in ("vkCmdFillBuffer", "vkCmdUpdateBuffer"):
                # "Clear Commands" chapter: fill and update run in VK_PIPELINE_STAGE_2_CLEAR_BIT
                self.use(qn, self.n(c["dst"]), "write", None, where, stage="CLEAR", flag="TRANSFER_WRITE",
                         region=self.buf_region(c["dst"], c.get("offset", 0), c.get("size")))
            elif fn == "vkCmdClearAttachments":
                # "Clearing Attachments": a write of the current subpass's attachments, color in
                # COLOR_ATTACHMENT_OUTPUT, depth/stencil in the fragment test stages, like a draw's. The
                # clear rectangles are not tracked: the whole attachment view is taken to be written.
                cmd = object()
                for a in c.get("attachments") or []:
                    asp = " ".join(a.get("aspect") or [])
                    if "COLOR" in asp:
                        e = (st.get("catt") or {}).get(a.get("color"))
                        ents = [e] if e else [x for x in st["att"] if x[0] == "color"]
                        for role, img, fmt, lay, resl, parts, tok in ents:
                            self.use(qn, parts, "write", lay, f"{where}: {role} attachment (clear)",
                                     "COLOR_ATTACHMENT_OUTPUT", "COLOR_ATTACHMENT_WRITE", scope=st.get("scope"),
                                     cmd=cmd, fb=tok)
                        continue
                    for role, img, fmt, lay, resl, parts, tok in st["att"]:
                        if (role == "depth" and ("DEPTH" in asp or "STENCIL" in asp)) or \
                                (role == "stencil" and "STENCIL" in asp):
                            self.use(qn, parts, "write", lay, f"{where}: {role} attachment (clear)",
                                     "EARLY_FRAGMENT_TESTS", "DEPTH_STENCIL_ATTACHMENT_WRITE",
                                     scope=st.get("scope"), cmd=cmd, fb=tok)
            elif fn == "vkCmdCopyQueryPoolResults" and "dst" not in c:
                pass  # a record from an older driver (marked unimplemented, no arguments): premise warning only
            elif fn == "vkCmdCopyQueryPoolResults":
                # the results are written to the buffer by a copy (sync2 COPY stage); the size of one
                # query's results depends on the pool, so `stride` bytes per query are taken to be written
                n, stride = c.get("count") or 0, c.get("stride") or 0
                self.use(qn, self.n(c["dst"]), "write", None, where, stage="COPY", flag="TRANSFER_WRITE",
                         region=self.buf_region(c["dst"], c.get("offset", 0), n * stride if stride else None))
            elif fn in ("vkCmdClearColorImage", "vkCmdClearDepthStencilImage"):
                self.use(qn, self.img_parts(c["dst"], c.get("ranges"), "range"), "write",
                         short_layout(c.get("dst_layout")), where, stage="CLEAR", flag="TRANSFER_WRITE")
            elif fn == "vkCmdPushConstants":
                self.check_push(c, where)
                # the bytes, for the buffer ranges shaders select with them
                pc = st.setdefault("pc", bytearray(256))
                mask = st.setdefault("pc_mask", bytearray(256))
                try:
                    data = bytes.fromhex(c.get("data") or "")
                except ValueError:
                    data = b""
                o = c.get("offset", 0) or 0
                for k in range(min(len(data), max(0, 256 - o))):
                    pc[o + k] = data[k]
                    mask[o + k] = 1
            elif fn in ("vkCmdPipelineBarrier", "vkCmdPipelineBarrier2"):
                if family is None:
                    family = self.queue_family.get(qn)
                self.barrier(c, qn, family, where)
            elif fn in ("vkCmdSetEvent", "vkCmdSetEvent2", "vkCmdResetEvent", "vkCmdResetEvent2"):
                # split barrier: the position whose earlier commands an event wait orders
                kind = "set" if "Set" in fn else "reset"
                self.q(qn).append(Barrier(None, None, None, where, frozenset(), frozenset(), frozenset(),
                                          frozenset(), event_mark=(kind, c.get("event"))))
            elif fn in ("vkCmdWaitEvents", "vkCmdWaitEvents2"):
                if family is None:
                    family = self.queue_family.get(qn)
                for i, ev in enumerate(c.get("events") or []):
                    dep = c if fn == "vkCmdWaitEvents" else (c.get("dependencies") or [{}] * (i + 1))[i]
                    if ev in self.host_events:
                        self.note_unsupported("event signalled by the host (vkSetEvent)", where, "approximated",
                                              "a device wait on a host-set event is taken to order nothing")
                    self.barrier(dep, qn, family, f"{where} (event {self.n(ev)})", event=ev)
            elif fn == "vkCmdExecuteCommands":
                for sec in c["command_buffers"]:
                    self.expand(sec, qn, submit_seq, st, depth + 1)
            elif c.get("warning") and fn not in _NO_MEMORY_EFFECT:
                # a command the model driver only stubs: its accesses and barriers are
                # unknown, so it is reported, never dropped silently
                self.note_unsupported(fn, where, "memory", "command not modelled (no accesses or barriers)",
                                      kind="command")

    def _att(self, att, role, a):
        asp = {"depth": ["VK_IMAGE_ASPECT_DEPTH_BIT"], "stencil": ["VK_IMAGE_ASPECT_STENCIL_BIT"]}.get(role)
        img, fmt = self.resolve_image(a.get("view"))
        if img:
            att.append((role, self.n(img), fmt, short_layout(a.get("layout")), self.view_res(a.get("view")),
                        self.view_parts(a.get("view"), asp), self.fb_token(a.get("view"))))
        if a.get("resolve_view"):
            rimg, rfmt = self.resolve_image(a["resolve_view"])
            if rimg:
                att.append((role + "_resolve", self.n(rimg), rfmt, short_layout(a.get("resolve_layout")),
                            self.view_res(a["resolve_view"]), self.view_parts(a["resolve_view"], asp),
                            self.fb_token(a["resolve_view"])))

    # -- attachment load/store operations (modelled as accesses)
    def rendering_atts(self, c):
        """[(role, view, load op, store op, layout)] of a vkCmdBeginRendering."""
        out = []
        for a in c.get("color") or []:
            if a and a.get("view") and self.resolve_image(a["view"])[0]:
                out.append(("color", a["view"], a.get("load"), a.get("store"), short_layout(a.get("layout"))))
        for role in ("depth", "stencil"):
            a = c.get(role)
            if a and a.get("view") and self.resolve_image(a["view"])[0]:
                out.append((role, a["view"], a.get("load"), a.get("store"), short_layout(a.get("layout"))))
        return out

    def load_store(self, qn, atts, load, where, scope):
        """Uses for the load (load=True) or store operations of attachments
        [(role, view, load op, store op, layout)]; role color, depth or stencil.
        Color: COLOR_ATTACHMENT_OUTPUT. Depth/stencil: loads in
        EARLY_FRAGMENT_TESTS, stores in LATE_FRAGMENT_TESTS."""
        for role, view, lop, sop, lay in atts:
            op = str(lop if load else sop or "")
            if op.endswith(("_NONE", "_NONE_KHR", "_NONE_EXT", "_NONE_QCOM")) or not op or op == "None":
                continue
            acc = "read" if load and op.endswith("LOAD_OP_LOAD") else "write"
            color = role == "color"
            stage = "COLOR_ATTACHMENT_OUTPUT" if color else ("EARLY_FRAGMENT_TESTS" if load else "LATE_FRAGMENT_TESTS")
            flag = ("COLOR_ATTACHMENT_" if color else "DEPTH_STENCIL_ATTACHMENT_") + acc.upper()
            asp = {"depth": ["VK_IMAGE_ASPECT_DEPTH_BIT"], "stencil": ["VK_IMAGE_ASPECT_STENCIL_BIT"]}.get(role)
            vasp = self.view_sub.get(view, (None,))[0]
            if asp and vasp and not any(role.upper() in str(x) for x in vasp):
                continue                        # the view does not hold this aspect
            img = self.n(self.resolve_image(view)[0])
            self.use(qn, self.view_parts(view, asp), acc, lay,
                     f"{where} ({'load' if load else 'store'} op {_strip(op, r'^VK_ATTACHMENT_(LOAD|STORE)_OP_')} "
                     f"of {role} attachment {img})",
                     stage, flag, scope=scope, instance=scope, pass_op="load" if load else "store",
                     fb=self.fb_token(view))

    def rp_load_store_atts(self, r, i, layout):
        """load_store entries for attachment i of a render pass instance."""
        a = r["rp"]["attachments"][i]
        view = r["views"][i] if i < len(r["views"]) else None
        if view is None:
            return []
        fmt = str(a.get("format", ""))
        # X8_D24_UNORM_PACK32 is a depth format
        if fmt.startswith(("VK_FORMAT_D", "VK_FORMAT_S8", "VK_FORMAT_X8_D24")):
            out = []
            if not fmt.startswith("VK_FORMAT_S8"):
                out.append(("depth", view, a.get("load"), a.get("store"), layout))
            if "S8" in fmt:
                out.append(("stencil", view, a.get("stencil_load"), a.get("stencil_store"), layout))
            return out
        return [("color", view, a.get("load"), a.get("store"), layout)]

    def bind_dynamic(self, st, c):
        """Dynamic offsets of the bound sets: (set, binding, element) -> offset,
        in the order Vulkan consumes them (set, binding, element)."""
        bp = c["bind_point"]
        dyn = st.setdefault("dyn", {}).setdefault(bp, {})
        offs = list(c.get("dynamic_offsets") or [])
        k = 0
        for i, sid in enumerate(c["sets"]):
            si = c["first_set"] + i
            for key in [x for x in dyn if x[0] == si]:
                del dyn[key]
            lay = self.m.set_layouts.get(self.set_layout_of.get(sid), [])
            for b in sorted(lay, key=lambda b: b.get("binding", 0)):
                if "DYNAMIC" not in str(b.get("type", "")):
                    continue
                for el in range(b.get("count", 1) or 0):
                    dyn[(si, b.get("binding"), el)] = offs[k] if k < len(offs) else None
                    k += 1
        if k != len(offs):
            for key in list(dyn):
                dyn[key] = None             # the offsets cannot be matched: whole buffers

    def desc_region(self, st, bp, si, binding, el, dtype, entry):
        """Elements a descriptor covers: the buffer range (plus its dynamic
        offset), the texel view's range, or the image view's subresources."""
        if "buffer" in entry and entry.get("buffer"):
            off = entry.get("offset", 0) or 0
            if "DYNAMIC" in str(dtype):
                d = st.get("dyn", {}).get(bp, {}).get((si, binding, el))
                if d is None:
                    return None
                off += d
            return self.buf_region(entry["buffer"], off, entry.get("range"))
        if "texel_view" in entry and entry.get("texel_view") in self.buffer_view_range:
            bv = self.buffer_views.get(entry["texel_view"])
            o, rg = self.buffer_view_range[entry["texel_view"]]
            return self.buf_region(bv[0], o, rg) if bv else None
        if "view" in entry and entry.get("view"):
            asp, bl, lc = self.view_sub.get(entry["view"], (None, 0, None))
            img, _ = self.resolve_image(entry["view"])
            return self.img_region(img, asp, bl, lc) if img else None
        return None

    def copy_parts(self, fn, c, side, key):
        """Resources and regions one side of a copy/blit touches."""
        h = c[side]
        regions = c.get("regions") or []
        if key:                                 # an image side
            return self.img_parts(h, [r.get(key) or {} for r in regions] if regions else None)
        name = self.n(h)
        if fn == "vkCmdCopyBuffer" and regions:
            reg = ()
            for r in regions:
                o = r.get("src_offset" if side == "src" else "dst_offset", 0)
                reg = _region_union(reg, self.buf_region(h, o, r.get("size")))
            return [(name, reg)]
        if regions and all("buffer_offset" in r for r in regions):
            # buffer <-> image: from the lowest offset on (row pitch not modelled)
            return [(name, self.buf_region(h, min(r["buffer_offset"] for r in regions), None))]
        return [(name, None)]

    def rp_views(self, c):
        fb = self.framebuffers.get(c.get("framebuffer"), {})
        return c.get("attachments") or fb.get("attachments") or []

    def begin_render_pass(self, c, st, qn, where):
        rp = self.render_passes.get(c["render_pass"])
        if not rp:
            return
        views = self.rp_views(c)
        imgs = [self.resolve_image(v) for v in views]
        cur = {}
        for i, a in enumerate(rp["attachments"]):
            if i < len(imgs) and imgs[i][0]:
                cur[i] = short_layout(a["initial_layout"])
        st["rp"] = {"rp": rp, "imgs": imgs, "views": views, "subpass": 0, "cur": cur,
                    "used": {}, "where": where}
        st["in_pass"] = True
        st["depth_clear"] = None
        cv = c.get("clear_values") or []
        st["clears"] = {self.n(imgs[i][0]): cv[i].get("color") for i, a in enumerate(rp["attachments"])
                        if a.get("load") == "VK_ATTACHMENT_LOAD_OP_CLEAR" and i < len(cv) and i < len(imgs)
                        and imgs[i][0] and isinstance(cv[i], dict)}
        sp0 = rp["subpasses"][0] if rp["subpasses"] else {}
        di = (sp0.get("depth") or {}).get("attachment", -1)
        cv = c.get("clear_values") or []
        if 0 <= di < len(rp["attachments"]) and rp["attachments"][di]["load"] == "VK_ATTACHMENT_LOAD_OP_CLEAR" \
                and di < len(cv):
            st["depth_clear"] = cv[di]["depth"]
        self.enter_subpass(st, qn, where, first=True)

    def enter_subpass(self, st, qn, where, first=False):
        r = st["rp"]
        sp = r["rp"]["subpasses"][r["subpass"]] if r["subpass"] < len(r["rp"]["subpasses"]) else None
        att = []
        if sp is None:
            st["att"] = att
            st["scope"] = object()
            return
        # subpass dependencies into this subpass order it after earlier work (and
        # carry its layout transitions); without one from EXTERNAL the spec's
        # implicit dependency waits on nothing (srcStage TOP_OF_PIPE)
        sub = r["subpass"]
        deps = [d for d in r["rp"].get("dependencies", []) if d.get("dst_subpass") == sub
                and d.get("src_subpass") != sub]
        if first and not any(d.get("src_subpass") == -1 for d in deps):
            deps.append({"src_stage": ["VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT"], "src_access": [],
                         "dst_stage": ["VK_PIPELINE_STAGE_ALL_COMMANDS_BIT"],
                         "dst_access": IMPLICIT_ATT_ACCESS, "implicit": True})
        call = ("render pass", where, sub)
        if deps:
            dwhere = where + (" (render pass: implicit dependency from EXTERNAL)"
                              if all(d.get("implicit") for d in deps)
                              else f" (subpass dependency into subpass {sub})")
            # one barrier per dependency, all structs of one call (the
            # union of their masks would invent src_i -> dst_j dependencies)
            for k, d in enumerate(deps):
                self.q(qn).append(Barrier(None, None, None, dwhere, **self._masks(d), call=call, struct=k,
                                          fb_local=self._rp_fb_local(d)))
        refs = [("color", x) for x in sp.get("color", [])] + [("color_resolve", x) for x in sp.get("resolve", [])]
        if sp.get("depth"):
            refs.append(("depth", sp["depth"]))
        refs += [("input", x) for x in sp.get("input", [])]
        colk = {id(x): k for k, x in enumerate(sp.get("color", []))}
        st["catt"] = {}
        scope = object()
        loads = []
        for role, ref in refs:
            i = ref["attachment"]
            if i < 0 or i >= len(r["imgs"]) or not r["imgs"][i][0]:
                continue
            img, fmt = r["imgs"][i]
            resl = self.view_res(r["views"][i])
            parts = self.view_parts(r["views"][i])
            lay = short_layout(ref["layout"])
            if r["cur"].get(i) != lay:
                old = r["cur"].get(i)
                # the transition happens-after the availability operations of
                # every dependency into this subpass and before the visibility
                # operations of those whose dst scope holds the attachment's use
                sel = _deps_for(deps, role, src=False)
                masks = _union_dep_masks(sel)
                fbl = _fb_join(self._rp_fb_local(d) for d in sel)
                for x, reg in parts:
                    self.q(qn).append(Barrier(x, None if old == "undefined" else old, lay,
                                              where + " (render pass layout transition)",
                                              **masks, call=call, struct=1000 + i,
                                              region=reg, fb_local=fbl, fb=self.fb_token(r["views"][i]),
                                              auto=True))
                r["cur"][i] = lay
            if i not in r["used"]:
                loads += self.rp_load_store_atts(r, i, lay)
            r["used"][i] = (scope, lay)
            if role == "input":
                self.use(qn, parts, "read", lay, where + " (input attachment)",
                         "FRAGMENT_SHADER", "INPUT_ATTACHMENT_READ", instance=scope,
                         fb=self.fb_token(r["views"][i], True))
            else:
                att.append((role, self.n(img), fmt, lay, resl, parts, self.fb_token(r["views"][i])))
                if role == "color":
                    st["catt"][colk[id(ref)]] = att[-1]
        self.load_store(qn, loads, True, where, scope)
        st["att"] = att
        st["scope"] = scope

    def next_subpass(self, st, qn, where):
        if st["rp"]:
            st["rp"]["subpass"] += 1
            self.enter_subpass(st, qn, where)

    def end_render_pass(self, st, qn, where):
        r = st["rp"]
        if not r:
            return
        last = len(r["rp"]["subpasses"]) - 1
        deps = [d for d in r["rp"].get("dependencies", []) if d.get("dst_subpass") == -1]
        if not deps:
            # implicit dependency to EXTERNAL: ends at BOTTOM_OF_PIPE, makes nothing visible
            deps = [{"src_stage": ["VK_PIPELINE_STAGE_ALL_COMMANDS_BIT"], "src_access": IMPLICIT_ATT_ACCESS,
                     "dst_stage": ["VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT"], "dst_access": [],
                     "implicit": True}]
        # store operations, in the scope of the last subpass using each attachment
        for i in sorted(r["used"]):
            scope, lay = r["used"][i]
            self.load_store(qn, self.rp_load_store_atts(r, i, lay), False, where, scope)
        call = ("render pass", where, "end")
        roles = {}
        for sp in r["rp"]["subpasses"]:
            for x in sp.get("color", []) + sp.get("resolve", []):
                roles.setdefault(x["attachment"], "color")
            if sp.get("depth"):
                roles.setdefault(sp["depth"]["attachment"], "depth")
            for x in sp.get("input", []):
                roles.setdefault(x["attachment"], "input")
        for i, a in enumerate(r["rp"]["attachments"]):
            if i in r["cur"]:
                fin = short_layout(a["final_layout"])
                if r["cur"][i] != fin:
                    sel = _deps_for(deps, roles.get(i, "color"), src=True)
                    masks = _union_dep_masks(sel)
                    fbl = _fb_join(self._rp_fb_local(d) for d in sel)
                    for x, reg in self.view_parts(r["views"][i]):
                        # the final-layout transition starts from the attachment's
                        # current layout; None ("from UNDEFINED") only if that is undefined
                        cur = r["cur"][i]
                        self.q(qn).append(Barrier(x, None if cur in (None, "undefined") else cur, fin,
                                                  where + " (render pass final layout)",
                                                  **masks, call=call, struct=1000 + i, region=reg,
                                                  fb_local=fbl, fb=self.fb_token(r["views"][i]), auto=True))
                    r["cur"][i] = fin
        dwhere = where + (" (render pass: implicit dependency to EXTERNAL)" if deps[0].get("implicit")
                          else f" (subpass {last} -> EXTERNAL dependency)")
        for k, d in enumerate(deps):
            self.q(qn).append(Barrier(None, None, None, dwhere, **self._masks(d), call=call, struct=k,
                                      fb_local=self._rp_fb_local(d)))
        st["rp"] = None
        st["att"] = []
        st["in_pass"] = False

    # -- ray tracing / acceleration structures ------------------------------
    def note_unsupported(self, what, where, effect="memory", detail="", kind="data"):
        """Record something the model does not cover (reported by timing.check
        as report.meta["model.unsupported"]; effect "memory" also as a warning:
        timing.unmodelled-command for kind "command", else timing.unmodelled)."""
        e = self.m.unsupported.setdefault(what, {"count": 0, "first": where, "effect": effect, "detail": detail,
                                                 "kind": kind})
        e["count"] += 1

    def as_res(self, a):
        """(buffer name, byte region) an acceleration structure lives in, or None."""
        info = self.m.accel.get(a)
        if info is None:
            return None
        return (self.n(info.buffer), self.buf_region(info.buffer, info.offset, info.size))

    def dev_res(self, d, where, what):
        """(buffer name, byte region) of a device-address range the driver resolved."""
        if not d:
            return None
        if d.get("unresolved") or not d.get("buffer"):
            self.note_unsupported(f"unresolved device address ({what})", where, "memory",
                                  "the address is in no live buffer; the access is not tracked")
            return None
        return (self.n(d["buffer"]), self.buf_region(d["buffer"], d.get("offset", 0), d.get("size")))

    def blas_refs(self, a):
        """BLASes a TLAS's last build referenced (a BLAS or unknown: itself only)."""
        info = self.m.accel.get(a)
        if info is None or info.type == "bottom":
            return []
        refs = info.refs if info.refs is not None else frozenset()
        return [x for x in sorted(refs) if x in self.m.accel]

    def accel_cmd(self, c, cb, qn, where):
        fn = c["fn"]
        if fn in ("vkCmdBuildAccelerationStructuresKHR", "vkCmdBuildAccelerationStructuresIndirectKHR"):
            for i, b in enumerate(c.get("builds") or []):
                # builds of one call are unordered with each other: each its own command
                cmd, w = object(), f"{where}: build {i}"
                if b.get("mode") == "update" and b.get("src"):
                    self.use(qn, [self.as_res(b["src"])] if self.as_res(b["src"]) else None, "read", None,
                             f"{w} source", AS_BUILD, AS_READ, cmd=cmd)
                sc = self.dev_res(b.get("scratch"), w, "scratch")
                if sc:
                    self.use(qn, [sc], "read", None, f"{w} scratch", AS_BUILD, AS_READ, cmd=cmd)
                    self.use(qn, [sc], "write", None, f"{w} scratch", AS_BUILD, AS_WRITE, cmd=cmd)
                for gi, g in enumerate(b.get("geometries") or []):
                    for key in ("vertex", "index", "transform", "data"):
                        r = self.dev_res(g.get(key), w, f"build input {key}")
                        if r:
                            self.use(qn, [r], "read", None, f"{w} geometry {gi} {key}", AS_BUILD,
                                     "SHADER_READ", cmd=cmd)
                    if g.get("array_of_pointers"):
                        self.note_unsupported("TLAS instance array of pointers", w, "memory",
                                              "the instances the pointers name are not tracked")
                dst = self.m.accel.get(b.get("dst"))
                if b.get("type") == "top" or (dst and dst.type == "top"):
                    refs = self.tlas_rec.get((cb, c["seq"], i))
                    if refs is None:
                        if (cb, c["seq"], i) not in self.tlas_rec:
                            self.note_unsupported("unresolved TLAS instance reference", w, "approximated",
                                                  "the build is taken to read every live BLAS")
                        refs = frozenset(x for x in self.as_live if self.m.accel[x].type != "top")
                    for x in sorted(refs):
                        r = self.as_res(x)
                        if r:
                            self.use(qn, [r], "read", None, f"{w}: instance BLAS {self.n(x)}", AS_BUILD,
                                     AS_READ, cmd=cmd)
                    if dst:
                        dst.refs = frozenset(refs)
                if fn.endswith("IndirectKHR"):
                    ind = (c.get("indirect") or [])
                    r = self.dev_res(ind[i].get("range") if i < len(ind) else None, w, "indirect build range")
                    if r:
                        self.use(qn, [r], "read", None, f"{w} indirect range", AS_BUILD,
                                 "INDIRECT_COMMAND_READ", cmd=cmd)
                r = self.as_res(b.get("dst"))
                if r:
                    self.use(qn, [r], "write", None, f"{w} destination {self.n(b.get('dst'))}", AS_BUILD,
                             AS_WRITE, cmd=cmd)
        elif fn == "vkCmdCopyAccelerationStructureKHR":
            cmd = object()
            src, dst = self.as_res(c.get("src")), self.as_res(c.get("dst"))
            self.use(qn, [src] if src else None, "read", None, f"{where}: source", AS_COPY, AS_READ, cmd=cmd)
            self.use(qn, [dst] if dst else None, "write", None, f"{where}: destination", AS_COPY, AS_WRITE, cmd=cmd)
            s_i, d_i = self.m.accel.get(c.get("src")), self.m.accel.get(c.get("dst"))
            if s_i and d_i:
                d_i.refs = s_i.refs
        elif fn == "vkCmdCopyAccelerationStructureToMemoryKHR":
            cmd = object()
            src = self.as_res(c.get("src"))
            self.use(qn, [src] if src else None, "read", None, f"{where}: source", AS_COPY, AS_READ, cmd=cmd)
            r = self.dev_res(c.get("dst"), where, "serialization destination")
            self.use(qn, [r] if r else None, "write", None, f"{where}: destination", AS_COPY, "TRANSFER_WRITE", cmd=cmd)
        elif fn == "vkCmdCopyMemoryToAccelerationStructureKHR":
            cmd = object()
            r = self.dev_res(c.get("src"), where, "deserialization source")
            self.use(qn, [r] if r else None, "read", None, f"{where}: source", AS_COPY, "TRANSFER_READ", cmd=cmd)
            dst = self.as_res(c.get("dst"))
            self.use(qn, [dst] if dst else None, "write", None, f"{where}: destination", AS_COPY, AS_WRITE, cmd=cmd)
            d_i = self.m.accel.get(c.get("dst"))
            if d_i:
                d_i.refs = None
        elif fn == "vkCmdWriteAccelerationStructuresPropertiesKHR":
            cmd = object()
            for a in c.get("acceleration_structures") or []:
                r = self.as_res(a)
                self.use(qn, [r] if r else None, "read", None, f"{where}: {self.n(a)}", AS_BUILD, AS_READ, cmd=cmd)

    def host_accel(self, r):
        """accelerationStructureHostCommands: the host builds/copies in place.
        Modelled as host accesses of the structures; build inputs are host
        addresses the model does not track."""
        fn, where = r["fn"], f"{r['fn']} @seq {r['seq']}"
        self.note_unsupported("host acceleration-structure command inputs", where, "approximated",
                              "host-address build inputs are not tracked; the structures are")
        if fn == "vkBuildAccelerationStructuresKHR":
            for i, b in enumerate(r.get("builds") or []):
                if b.get("mode") == "update" and self.as_res(b.get("src")):
                    self.use("host", [self.as_res(b["src"])], "read", None, f"{where}: build {i} source",
                             "HOST", "HOST_READ")
                d = self.m.accel.get(b.get("dst"))
                if d:
                    if d.type == "top":
                        d.refs = None
                    self.use("host", [self.as_res(b["dst"])], "write", None, f"{where}: build {i} destination",
                             "HOST", "HOST_WRITE")
        else:
            for key, acc in (("src", "read"), ("dst", "write")):
                if self.as_res(r.get(key)):
                    self.use("host", [self.as_res(r[key])], acc, None, f"{where}: {key}", "HOST",
                             "HOST_READ" if acc == "read" else "HOST_WRITE")
            for a in r.get("acceleration_structures") or []:
                if self.as_res(a):
                    self.use("host", [self.as_res(a)], "read", None, f"{where}: {self.n(a)}", "HOST", "HOST_READ")

    def shader_caps(self, sha):
        """OpCapability operands of a dumped module (empty if unknown)."""
        if sha in self.caps_cache:
            return self.caps_cache[sha]
        caps = set()
        path = os.path.join(self.spirv_dir or "", f"{sha}.spv")
        if self.spirv_dir and os.path.exists(path):
            with open(path, "rb") as f:
                code = f.read()
            words = struct.unpack("<%dI" % (len(code) // 4), code[: len(code) // 4 * 4])
            i = 5
            while i < len(words):
                op, n = words[i] & 0xFFFF, words[i] >> 16
                if n == 0 or op == 54:            # OpFunction: capabilities come first
                    break
                if op == 17 and n >= 2:
                    caps.add(words[i + 1])
                i += n
        self.caps_cache[sha] = caps
        return caps

    # -- dynamically used descriptor-array elements ----------------------------------
    def _tag(self, qn, n0, cond=None, guard=None, hseq=None, where=None, n1=None):
        """Set the IndexGuard fields on the Uses at positions n0..n1 of queue qn (appended
        by use() just before; kept out of use() so the change stays local)."""
        for u in self.q(qn)[n0:n1]:
            if cond is not None:
                u.cond = cond
            if guard is not None:
                u.guard = guard
            if hseq is not None:
                u.hseq = hseq
            if where is not None:
                u.where = where

    def _record_contents(self, r, page, offs):
        """Rebuild host-visible memory contents from a host_write record: the pages the
        trap saw written are host-written; "writes" runs carry the changed bytes (a run
        without content, or no runs at all, leaves those bytes unknown)."""
        mem = r.get("memory")
        hw = self.mem_hostw.setdefault(mem, set())
        for o in offs:
            hw.update(range(o // _CHUNK, (o + page - 1) // _CHUNK + 1))
        if r.get("writes_truncated"):
            self.mem_bad.setdefault(mem, "host_write record truncated")
        if r.get("writes") is None:
            unk = self.mem_unknown.setdefault(mem, [])
            unk.extend((o, o + page) for o in offs)
            return
        chunks = self.mem_chunks.setdefault(mem, {})
        for w in r["writes"]:
            off = int(w[0])
            if len(w) < 2 or not w[1]:
                n = w[2] if len(w) >= 3 and isinstance(w[2], int) else page
                self.mem_unknown.setdefault(mem, []).append((off, off + n))
                continue
            b = bytes.fromhex(w[1])
            i = 0
            while i < len(b):
                ci, co = divmod(off + i, _CHUNK)
                n = min(len(b) - i, _CHUNK - co)
                ch = chunks.get(ci)
                if ch is None:
                    ch = chunks[ci] = bytearray(_CHUNK)
                ch[co:co + n] = b[i:i + n]
                i += n

    def _read_u32(self, mem, addr):
        """The 4 bytes at addr of memory mem as rebuilt now, or None if they are not
        host-written with recorded content."""
        hw = self.mem_hostw.get(mem)
        if not hw or addr // _CHUNK not in hw or (addr + 3) // _CHUNK not in hw:
            return None
        for lo, hi in self.mem_unknown.get(mem, ()):
            if lo < addr + 4 and addr < hi:
                return None
        out = bytearray(4)
        chunks = self.mem_chunks.get(mem, {})
        for k in range(4):
            ci, co = divmod(addr + k, _CHUNK)
            ch = chunks.get(ci)
            out[k] = ch[co] if ch is not None else 0
        return struct.unpack("<I", bytes(out))[0]

    def index_terms(self, pipe, p):
        if pipe in self.index_cache:
            return self.index_cache[pipe]
        res = None
        if p is not None and p.kind == "graphics" and self.spirv_dir:
            def load(sha):
                path = os.path.join(self.spirv_dir, f"{sha}.spv")
                if not os.path.exists(path):
                    return None
                mod = self.module_cache.get(path)
                if mod is None:
                    mod = self.module_cache[path] = _spirv.parse(path)
                return mod
            try:
                res = _desc_index.pipeline_index_terms(p.stages, load)
            except Exception:
                res = None
        self.index_cache[pipe] = res
        return res

    def dynamic_elements(self, c, st, p, pipe, kind, submit_seq, where):
        """(IndexGuard, {(set, binding): elements | None}, vertex slots read)
        for a direct non-indexed draw whose descriptor-array indices are a flat integer vertex
        attribute (desc_index), decoded from the host-written vertex / instance bytes as of
        this submit; None when nothing can be decoded. A binding maps to None (every written
        element is charged) when any of its indices cannot be decoded."""
        if not self.desc_refine or kind != "draw" or c["fn"] != "vkCmdDraw" or p is None:
            return None
        terms = self.index_terms(pipe, p)
        if not terms or not any(v for v in terms.values()):
            return None
        rec = p.record or {}
        dyn_states = " ".join(str(x) for x in rec.get("dynamic_states") or [])
        if "VERTEX_INPUT" in dyn_states:            # VERTEX_INPUT_EXT, VERTEX_INPUT_BINDING_STRIDE
            return None
        attrs = {a.get("location"): a for a in rec.get("vertex_attributes") or []}
        vbind = {b.get("binding"): b for b in rec.get("vertex_bindings") or []}
        guard = IndexGuard(submit_seq, where)
        out, slots, span = {}, set(), {}
        decoded = {}                                # attribute location -> set of 32-bit values

        def values(loc):
            if loc in decoded:
                return decoded[loc]
            a = attrs.get(loc)
            vals = None
            if a is not None and a.get("format") in _INT_FORMATS:
                b = vbind.get(a.get("binding"))
                buf = st["vb"].get(a.get("binding"))
                info = self.m.buffers.get(buf) if buf else None
                if b is not None and info is not None and info.memory is not None and \
                        b.get("divisor") in (None, 1) and b.get("stride") is not None:
                    inst = "INSTANCE" in str(b.get("rate"))
                    first = (c.get("first_instance") if inst else c.get("first_vertex")) or 0
                    n = c.get("instance_count") if inst else c.get("vertex_count")
                    stride, aoff = b["stride"], a.get("offset", 0) or 0
                    voff = st.get("vbo", {}).get(a.get("binding"), 0) or 0
                    if isinstance(n, int) and 0 <= n <= _MAX_DECODE and isinstance(first, int):
                        vals = set()
                        for i in range(first, first + n):
                            boff = voff + i * stride + aoff
                            if boff < 0 or boff + 4 > info.size:
                                vals = None
                                break
                            v = self._read_u32(info.memory, info.offset + boff)
                            if v is None:
                                vals = None
                                break
                            vals.add(v)
                        if vals is not None and n > 0:
                            lo = info.offset + voff + first * stride + aoff
                            hi = info.offset + voff + (first + n - 1) * stride + aoff + 4
                            m0 = span.get(info.memory)
                            span[info.memory] = (min(lo, m0[0]), max(hi, m0[1])) if m0 else (lo, hi)
                            slots.add(a.get("binding"))
                        elif vals is not None:
                            slots.add(a.get("binding"))
            decoded[loc] = vals
            return vals

        for key, ts in terms.items():
            if ts is None:
                out[key] = None
                continue
            els = set()
            for t in ts:
                if t[0] == "c":
                    els.add(t[1])
                    continue
                vals = values(t[1])
                if vals is None:
                    els = None
                    break
                els.update(_desc_index.apply_ops(t[2], v) for v in vals)
            if els is not None and any(e >= 1 << 31 for e in els):
                els = None                          # negative / out-of-range index: undefined
            out[key] = els
        if not any(v is not None for v in out.values()) or not slots:
            return None
        for mem, (lo, hi) in span.items():
            self.guards.append((guard, mem, lo, hi))
        if not span:
            self.guards.append((guard, None, 0, 0))
        return guard, out, slots

    def finish_guards(self):
        """Fail (statically) every IndexGuard whose decoded bytes are not determined by
        the host alone: a device write or an image bound anywhere on those bytes of the memory,
        an untrusted capture of that memory, or a model gap that may hide accesses."""
        if not self.guards:
            return
        gap = next((k for k, u in self.m.unsupported.items() if u.get("effect") in ("memory", "untracked")), None)
        dev = {}                                    # memory -> [(lo, hi)] bytes a device command writes
        by_name = {self.n(b): info for b, info in self.m.buffers.items()}
        for qn, ops in self.fr.queues.items():
            if qn == "host":
                continue
            for op in ops:
                if not isinstance(op, Use) or op.access != "write":
                    continue
                info = by_name.get(op.res)
                if info is None or info.memory is None:
                    continue
                for lo, hi in (op.region if op.region is not None else ((0, info.size),)):
                    dev.setdefault(info.memory, []).append((info.offset + lo, info.offset + min(hi, info.size)))
        for g, mem, lo, hi in self.guards:
            why = self.capture_bad or (f"model gap: {gap}" if gap else None)
            if why is None and mem is not None:
                why = self.mem_bad.get(mem)
                if why is None and any(a < hi and lo < b for a, b in dev.get(mem, ())):
                    why = "device writes to the vertex/instance memory"
                mi = self.m.memories.get(mem)
                if why is None and mi is not None and any(
                        k == "image" and off < hi and lo < off + size for k, _, off, size in mi.bindings):
                    why = "an image is bound on the vertex/instance memory"
            if why is not None and g.static is None:
                g.static = why

    def draw(self, c, st, qn, submit_seq, cbname, where, kind):
        bp = _BIND_POINT[kind]
        pipe = st["pipeline"].get(bp)
        p = self.m.pipelines.get(pipe)
        if p and any(5347 in self.shader_caps(sha) for _, sha, _ in p.stages):
            # PhysicalStorageBufferAddresses: loads/stores through raw pointers
            self.note_unsupported("shader buffer-device-address accesses", where, "untracked",
                                  f"pipeline {self.n(pipe)} reaches memory through device addresses")
        nsets = len(p.set_layouts) if p else None
        acc_map = self.access_for(pipe) if pipe else None
        if p and p.layout and acc_map:
            self.check_set_compat(p, {k[0] for k in acc_map}, st.get("set_bind_layout", {}).get(bp, {}), where)
        descs, elems = {}, {}
        cmd = object()
        inst = st.get("scope") if st.get("in_pass") else None   # the subpass the command is in
        dyn = self.dynamic_elements(c, st, p, pipe, kind, submit_seq, where)   # descriptor-array indexing
        wheres, marks = {}, []     # (first Use, cond, shared where) per element, tagged below
        for si, sid in sorted(st["sets"].get(bp, {}).items()):
            if nsets is not None and si >= nsets:
                continue
            if acc_map is None or any(k[0] == si for k in acc_map):
                # the set object itself must outlive the work that reads it
                self.use(qn, f"set:{self.n(sid)}", "read", None, f"{where}: set {si}",
                         None, "DESCRIPTOR_READ")
            for decl in self.m.set_layouts.get(self.set_layout_of.get(sid), []):
                if decl.get("immutable_samplers") and decl.get("type") == "VK_DESCRIPTOR_TYPE_SAMPLER":
                    descs.setdefault((si, decl.get("binding")), Bound(decl["type"], "immutable sampler", None))
            for (binding, el), (dtype, entry) in sorted(self.set_contents.get(sid, {}).items()):
                b, res, lay = self.bound_of(dtype, entry)
                elems[(si, binding, el)] = b
                if el == 0 or (si, binding) not in descs:
                    descs[(si, binding)] = b
                if (dtype or "").startswith("VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE"):
                    # trace / ray query: reads the TLAS and the BLASes its last build referenced
                    a = entry.get("acceleration_structure")
                    if not a or (acc_map is not None and (si, binding) not in acc_map):
                        continue
                    ress = [x for x in [self.as_res(a)] + [self.as_res(y) for y in self.blas_refs(a)] if x]
                    if a not in self.m.accel:
                        self.note_unsupported("acceleration structure descriptor of unknown object", where,
                                              "memory", "the structure's memory is not known")
                    for stg in self.binding_stages(sid, binding, kind, p, self._used_at(pipe, si, binding)):
                        self.use(qn, ress, "read", None, f"{where}: set {si} binding {binding}", stg, AS_READ,
                                 cmd=cmd, instance=inst)
                    continue
                if res is None or dtype not in _READ_TYPES | _STORAGE_TYPES:
                    continue
                if acc_map is not None and (si, binding) not in acc_map:
                    continue                    # bound but no entry point touches it
                if dtype in _STORAGE_TYPES:
                    # from the SPIR-V when we have it; otherwise a write (fail closed:
                    # shader_gap has recorded the missing code)
                    if acc_map is not None:
                        acc = "write" if acc_map[(si, binding)] else "read"
                    else:
                        acc = "write"
                    flag = "SHADER_STORAGE_WRITE" if acc == "write" else "SHADER_STORAGE_READ"
                else:
                    acc, flag = "read", _DESC_FLAG.get(dtype)
                reg = self.desc_region(st, bp, si, binding, el, dtype, entry)
                if reg != () and "BUFFER" in str(dtype) and "TEXEL" not in str(dtype) and entry.get("buffer"):
                    reg = self.refine_region(pipe, kind, si, binding, st, c, bp, el, dtype, entry, reg)
                atom = None
                if dtype in _STORAGE_TYPES and (si, binding) in self.mode_cache.get(pipe, {}):
                    # member spans -> buffer bytes (descriptor offset, plus the dynamic offset)
                    base = (entry.get("offset", 0) or 0) if entry.get("buffer") else None
                    if base is not None and "DYNAMIC" in str(dtype):
                        d = st.get("dyn", {}).get(bp, {}).get((si, binding, el))
                        base = None if d is None else base + d
                    atom = _spirv.physical_modes(self.mode_cache[pipe][(si, binding)], res, base)
                used = self._used_at(pipe, si, binding)
                if dyn is not None:     # this element's Uses are tagged after the loop
                    bwhere = wheres.get((si, binding))     # one string per binding, not per element
                    if bwhere is None:
                        bwhere = wheres[(si, binding)] = f"{where}: set {si} binding {binding}"
                    cond = None
                    if dtype in _DYN_TYPES and dyn[1].get((si, binding)) is not None:
                        if el in dyn[1][(si, binding)]:
                            self.desc_refined += 1
                        else:
                            cond = dyn[0]       # not dynamically used unless the guard fails
                            self.desc_widened += 1
                    marks.append((len(self.q(qn)), cond, bwhere))
                for stg in self.binding_stages(sid, binding, kind, p, used):
                    a_s, f_s = acc, flag
                    if dtype in _STORAGE_TYPES and used and stg in used and acc == "write" and not used[stg]:
                        # this stage's entry point only reads it (another stage writes)
                        a_s, f_s = "read", "SHADER_STORAGE_READ"
                    self.use(qn, res, a_s, lay, f"{where}: set {si} binding {binding}", stg, f_s,
                             region=reg, cmd=cmd, instance=inst, atomic=atom,
                             fb=self.fb_token(entry.get("view"), True)
                             if dtype == "VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT" else None)
        vbs = []
        if kind == "draw" and c["fn"] not in _MESH_DRAWS:
            slots = [b["binding"] for b in p.vertex_bindings] if p and p.vertex_bindings else sorted(st["vb"])
            for i, (n0, cond, bw) in enumerate(marks):     # tag the descriptor Uses
                self._tag(qn, n0, cond=cond, where=bw, n1=marks[i + 1][0] if i + 1 < len(marks) else None)
            for s in slots:
                if s in st["vb"]:
                    vbs.append(self.n(st["vb"][s]))
                    n0 = len(self.q(qn))
                    self.use(qn, self.n(st["vb"][s]), "read", None, f"{where}: vertex buffer {s}",
                             "VERTEX_INPUT", "VERTEX_ATTRIBUTE_READ", cmd=cmd, instance=inst,
                             region=self.buf_region(st["vb"][s], st.get("vbo", {}).get(s, 0), None))
                    if dyn is not None and s in dyn[2]:
                        self._tag(qn, n0, guard=dyn[0])     # the read the guard is about
            if c["fn"] in ("vkCmdDrawIndexed", "vkCmdDrawIndexedIndirect", "vkCmdDrawIndexedIndirectCount") \
                    and st["ib"]:
                self.use(qn, self.n(st["ib"]), "read", None, f"{where}: index buffer",
                         "VERTEX_INPUT", "INDEX_READ", cmd=cmd, instance=inst,
                         region=self.index_region(st, c))
        if c.get("buffer"):
            self.use(qn, self.n(c["buffer"]), "read", None, f"{where}: indirect buffer",
                     "DRAW_INDIRECT", "INDIRECT_COMMAND_READ", cmd=cmd, instance=inst,
                     region=self.indirect_region(c))
        if c.get("count_buffer"):
            self.use(qn, self.n(c["count_buffer"]), "read", None, f"{where}: count buffer",
                     "DRAW_INDIRECT", "INDIRECT_COMMAND_READ", cmd=cmd, instance=inst,
                     region=self.buf_region(c["count_buffer"], c.get("count_offset", 0), 4))
        if kind == "trace":
            if c.get("indirect"):
                r = self.dev_res(c["indirect"], where, "trace indirect")
                self.use(qn, [r] if r else None, "read", None, f"{where}: indirect parameters",
                         "DRAW_INDIRECT", "INDIRECT_COMMAND_READ", cmd=cmd)
            if c.get("sbt_unknown"):
                self.note_unsupported("shader binding table of vkCmdTraceRaysIndirect2KHR", where, "memory",
                                      "the table addresses live in device memory; their reads are not tracked")
            for key in ("raygen", "miss", "hit", "callable"):
                r = self.dev_res(c.get(key), where, f"{key} shader binding table")
                self.use(qn, [r] if r else None, "read", None, f"{where}: {key} shader binding table",
                         "RAY_TRACING_SHADER", "SHADER_BINDING_TABLE_READ", cmd=cmd)
        rec = p.record if p else {}
        if kind == "draw":
            dw = self.ds_state(st, rec, p, "depth_write", "DEPTH_WRITE_ENABLE")
            dt = self.ds_state(st, rec, p, "depth_test", "DEPTH_TEST_ENABLE")
            stt = self.ds_state(st, rec, p, "stencil_test", "STENCIL_TEST_ENABLE")
            for role, img, fmt, lay, resl, parts, tok in st["att"]:
                if role.startswith("color"):
                    self.use(qn, parts, "write", lay, f"{where}: {role} attachment",
                             "COLOR_ATTACHMENT_OUTPUT", "COLOR_ATTACHMENT_WRITE", scope=st.get("scope"), cmd=cmd, instance=inst, fb=tok)
                    continue
                # a depth/stencil attachment is written only with writes enabled and a
                # writable layout; otherwise the draw only tests against it
                ro_layout = lay is not None and "read" in str(lay)
                if role in ("depth", "stencil"):
                    # "Depth writes are always disabled when depthTestEnable
                    # is VK_FALSE" (fragops "Depth Test"); with the depth test off the
                    # depth aspect is not accessed at all. The stencil aspect is
                    # accessed only with the stencil test on (its ops may write).
                    has_stencil = "S8" in str(fmt or "")
                    if role == "depth":
                        acc = ("write" if dw is not False and not ro_layout else "read") \
                            if dt is not False else None
                    else:
                        acc = None
                    if (role == "stencil" or has_stencil) and stt is not False \
                            and (role == "stencil" or stt is True):
                        sacc = "read" if ro_layout else "write"
                        acc = "write" if "write" in (acc, sacc) else sacc
                    if acc == "write":
                        self.use(qn, parts, "write", lay, f"{where}: {role} attachment",
                                 "EARLY_FRAGMENT_TESTS", "DEPTH_STENCIL_ATTACHMENT_WRITE", scope=st.get("scope"),
                                 cmd=cmd, instance=inst, fb=tok)
                    elif acc == "read":
                        self.use(qn, parts, "read", lay, f"{where}: {role} attachment (read-only)",
                                 "EARLY_FRAGMENT_TESTS", "DEPTH_STENCIL_ATTACHMENT_READ", scope=st.get("scope"),
                                 cmd=cmd, instance=inst, fb=tok)
                    continue
                if dw is not False and not ro_layout:
                    self.use(qn, parts, "write", lay, f"{where}: {role} attachment",
                             "EARLY_FRAGMENT_TESTS", "DEPTH_STENCIL_ATTACHMENT_WRITE", scope=st.get("scope"),
                             cmd=cmd, instance=inst, fb=tok)
                elif dt is not False:
                    self.use(qn, parts, "read", lay, f"{where}: {role} attachment (read-only)",
                             "EARLY_FRAGMENT_TESTS", "DEPTH_STENCIL_ATTACHMENT_READ", scope=st.get("scope"),
                             cmd=cmd, instance=inst, fb=tok)
        self.m.draws.append(DrawEvent(
            seq=c["seq"], fn=c["fn"], kind=kind, pipeline=pipe, queue=qn, submit_seq=submit_seq,
            command_buffer=cbname, descriptors=descs, descriptor_elements=elems,
            attachments=[a[:4] for a in st["att"]] if kind == "draw" else [], vertex_buffers=vbs,
            index_buffer=self.n(st["ib"]) if (kind == "draw" and st["ib"]) else None,
            depth_test=self.ds_state(st, rec, p, "depth_test", "DEPTH_TEST_ENABLE"),
            depth_write=self.ds_state(st, rec, p, "depth_write", "DEPTH_WRITE_ENABLE"),
            depth_compare=st["depth_compare"] or rec.get("depth_compare"),
            depth_clear=st["depth_clear"] if kind == "draw" else None,
            viewport_depth=st["viewport_depth"] if kind == "draw" else None,
            viewport=(tuple(rec["viewports"][0][:4]) if rec.get("viewports") else st.get("viewport"))
            if kind == "draw" else None,
            color_clears=dict(st.get("clears") or {}) if kind == "draw" else {}))

    def range_exprs(self, pipeline):
        """{(set, binding): [(offset expr, size)] | None} over the pipeline's
        stages, plus the compute local size (covenant.ranges); None if unknown."""
        if pipeline in self.range_cache:
            return self.range_cache[pipeline]
        out, lsize = None, None
        p = self.m.pipelines.get(pipeline)
        if p and self.spirv_dir and os.environ.get("COVENANT_NO_RANGES", "") in ("", "0"):
            out = {}
            for stage, sha, entry in p.stages:
                path = os.path.join(self.spirv_dir, f"{sha}.spv")
                try:
                    mod = self.module_cache.get(path) or _spirv.parse(path)
                    self.module_cache[path] = mod
                    got = _ranges.binding_exprs(mod, entry)
                    if p.kind == "compute":
                        lsize = _ranges.local_size(mod, entry)
                except Exception:
                    out = None
                    break
                for k, v in got.items():
                    out[k] = None if (v is None or (k in out and out[k] is None)) else out.get(k, []) + v
        self.range_cache[pipeline] = (out, lsize)
        return self.range_cache[pipeline]

    def refine_region(self, pipe, kind, si, binding, st, c, bp, el, dtype, entry, reg):
        """The descriptor region narrowed to the bytes this command's
        shaders can access (push constants, dispatch / draw parameters)."""
        if pipe is None or el != 0:
            return reg
        exprs, lsize = self.range_exprs(pipe)
        if not exprs or (si, binding) not in exprs or exprs[(si, binding)] is None:
            return reg
        bi = {}
        fn = c["fn"]
        if kind == "dispatch" and fn in ("vkCmdDispatch", "vkCmdDispatchBase") and c.get("groups"):
            g = list(c["groups"])[:3]
            base = list(c.get("base") or [0, 0, 0])[:3]
            for a in range(3):
                bi[("WorkgroupId", a)] = (base[a], base[a] + g[a] - 1)
                bi[("NumWorkgroups", a)] = (g[a], g[a])
                if lsize and all(isinstance(x, int) and x > 0 for x in lsize):
                    bi[("LocalInvocationId", a)] = (0, lsize[a] - 1)
                    bi[("GlobalInvocationId", a)] = (base[a] * lsize[a], (base[a] + g[a]) * lsize[a] - 1)
            if lsize and all(isinstance(x, int) and x > 0 for x in lsize):
                bi[("LocalInvocationIndex", None)] = (0, lsize[0] * lsize[1] * lsize[2] - 1)
        elif kind == "draw" and fn in ("vkCmdDraw", "vkCmdDrawIndexed"):
            ic, fi = c.get("instance_count"), c.get("first_instance", 0) or 0
            if isinstance(ic, int) and ic > 0:
                bi[("InstanceIndex", None)] = (fi, fi + ic - 1)
            if fn == "vkCmdDraw" and isinstance(c.get("vertex_count"), int) and c["vertex_count"] > 0:
                fv = c.get("first_vertex", 0) or 0
                bi[("VertexIndex", None)] = (fv, fv + c["vertex_count"] - 1)
        ctx = _ranges.Ctx(st.get("pc"), st.get("pc_mask"), bi)
        rr = _ranges.ranges(exprs[(si, binding)], ctx)
        if rr is None:
            return reg
        off = entry.get("offset", 0) or 0
        if "DYNAMIC" in str(dtype):
            d = st.get("dyn", {}).get(bp, {}).get((si, binding, el))
            if d is None:
                return reg
            off += d
        narrow = ()
        for lo, hi in rr:
            r = self.buf_region(entry["buffer"], off + int(lo), None if hi == _ranges.INF else int(hi - lo))
            narrow = _region_union(narrow, r)
        out = _region_intersect(reg, narrow)
        if out != reg:
            self.narrowed += 1
        return out

    @staticmethod
    def ds_state(st, rec, p, key, dyn_name):
        """Depth/stencil enable in effect for a draw: a pipeline's static
        value, unless the pipeline declares the state dynamic (then the last
        vkCmdSet*Enable); shader objects / pipelines without the state use the
        dynamic value. None = unknown."""
        stv, pv = st.get(key), rec.get(key)
        if p is None or pv is None:
            return stv if stv is not None else pv
        if any(dyn_name in str(d) for d in rec.get("dynamic_states") or []):
            return stv
        return pv

    def indirect_region(self, c):
        """Bytes an indirect draw/dispatch reads: draw_count records of
        `stride` bytes (to the end of the buffer when the count is a GPU value)."""
        size = {"vkCmdDrawIndirect": 16, "vkCmdDrawIndexedIndirect": 20, "vkCmdDispatchIndirect": 12,
                "vkCmdDrawIndirectCount": 16, "vkCmdDrawIndexedIndirectCount": 20,
                "vkCmdDrawMeshTasksIndirectEXT": 12, "vkCmdDrawMeshTasksIndirectCountEXT": 12}.get(c["fn"])
        off = c.get("offset", 0) or 0
        n = c.get("max_draw_count") if c.get("count_buffer") else c.get("draw_count")
        if size is None or (n is not None and not isinstance(n, int)):
            return self.buf_region(c["buffer"], off, None)
        n = 1 if n is None else n
        if n <= 0:
            return ()
        stride = c.get("stride") or size
        return self.buf_region(c["buffer"], off, (n - 1) * stride + size)

    def _used_at(self, pipe, si, binding):
        """{stage: written} of the entry points that statically use (si, binding),
        or None when the pipeline's SPIR-V is unknown."""
        sc = self.stage_cache.get(pipe)
        if sc is None:
            return None
        return sc.get((si, binding))

    def binding_stages(self, set_id, binding, kind, p, used=None):
        """Pipeline stages a descriptor access is charged at.

        With the pipeline's SPIR-V known (``used``: the stages whose entry
        point statically uses the binding, spirv.entry_bindings over its call graph),
        the access is charged at exactly those stages: a shader invocation's accesses
        are performed by the stage of its entry point. Charging the access at every
        stage the set layout's stageFlags grant and the pipeline contains (binding-use
        detection per pipeline, the union over entry points) would be sound but
        over-approximate (GPUBugBench case 94's fixed variant).
        Without SPIR-V the layout-based charge stays (sound fallback)."""
        if kind == "dispatch":
            return ["COMPUTE_SHADER"]
        if kind == "trace":
            return ["RAY_TRACING_SHADER"]
        if used:
            st = [x for x in used if x not in ("COMPUTE_SHADER", "RAY_TRACING_SHADER")]
            if st:
                return sorted(st, key=lambda x: _stage_rank_name(x))
        lay = self.m.set_layouts.get(self.set_layout_of.get(set_id), [])
        stages = []
        for b in lay:
            if b.get("binding") == binding:
                stages = shader_stages(b.get("stages"))
        stages = [x for x in stages if x not in ("COMPUTE_SHADER", "RAY_TRACING_SHADER")]
        if p:
            used = set(shader_stages([n for st in p.stages for n in st[0].split("|")]))
            stages = [x for x in stages if x in used] or stages
        return stages or [None]

    @staticmethod
    def _masks(b):
        return dict(src_stages=stage_mask(b.get("src_stage")), src_access=access_mask(b.get("src_access")),
                    dst_stages=stage_mask(b.get("dst_stage")), dst_access=access_mask(b.get("dst_access")))

    def transfer(self, res, sf, df, family, where, region=None):
        """Queue family ownership transfer halves, matched in submission order, per range."""
        if res not in self.exclusive or sf is None or df is None or sf == df:
            return
        if family == sf:
            self.own.release(res, region, sf, df, where)
            self.batch_releases.append((res, where))
        elif family == df:
            if not self.own.acquire(res, region, sf, df):
                self.fr.findings.append((
                    "timing", "timing.ownership-unmatched-acquire",
                    f"this barrier acquires '{res}' for queue family {df} from family {sf}, "
                    f"but family {sf} never released it to {df} before; without the release "
                    "the acquire has no defined effect and the contents are undefined.",
                    where, ["record a matching release barrier (same families) on the "
                            "owning queue and order it before this acquire with a semaphore"]))

    def barrier(self, c, qn, family, where, event=None):
        """event: the barriers are a vkCmdWaitEvents* dependency on that event."""
        q = self.q(qn)
        start = len(q)
        self._barrier(c, qn, family, where)
        if event is not None:
            for op in q[start:]:
                op.event = event

    def _rp_fb_local(self, d):
        """fb_local of a subpass dependency (implicit ones: framebuffer-global)."""
        if d.get("implicit"):
            return False
        if d.get("dependency_flags") is None:
            self.legacy_rp_flags += 1
        return _fb_local(d)

    def _barrier(self, c, qn, family, where):
        # vkCmdPipelineBarrier* / VkDependencyInfo dependencyFlags
        # (vkCmdWaitEvents has none: framebuffer-global)
        fbl = _fb_local(c) if "dependency_flags" in c else False
        k = 0                       # struct index: structs of one call never chain
        for b in c.get("image_barriers", []):
            k += 1
            old, new = short_layout(b["old_layout"]), short_layout(b["new_layout"])
            sf, df = b.get("src_family"), b.get("dst_family")
            acquire = sf is not None and df is not None and sf != df and family == df
            names = self.img_res(b["image"], b.get("base_mip", 0), b.get("level_count"))
            reg = self.img_region(b["image"], b.get("aspect"), b.get("base_layer", 0), b.get("layer_count"))
            if acquire:
                if all(x in self.exclusive and self.own.released(x, reg, sf, df) for x in names):
                    # matched acquire half: the contents survive (no discard) and
                    # the layout change was done once, by the release half: no transition
                    old = new
                else:
                    old = None                  # acquire half of a queue family transfer
            if old == "undefined":
                old = None
            for img in names:
                self.transfer(img, sf, df, family, where, reg)
                self.q(qn).append(Barrier(img, old, new if new != old or old is None else None, where,
                                          **self._masks(b), call=where, struct=k, region=reg, fb_local=fbl))
                self.queue_used.setdefault(qn, set()).add(img)
        for b in c.get("buffer_barriers", []):
            k += 1
            buf = self.n(b["buffer"])
            breg = self.buf_region(b["buffer"], b.get("offset", 0), b.get("size"))
            self.transfer(buf, b.get("src_family"), b.get("dst_family"), family, where, breg)
            self.q(qn).append(Barrier(buf, None, None, where, **self._masks(b), call=where, struct=k,
                                      region=breg, fb_local=fbl))
            self.queue_used.setdefault(qn, set()).add(buf)
        for b in c.get("memory_barriers", []):
            k += 1
            self.q(qn).append(Barrier(None, None, None, where + " (global memory barrier)", **self._masks(b),
                                      call=where, struct=k, fb_local=fbl))
        if not (c.get("memory_barriers") or c.get("buffer_barriers") or c.get("image_barriers")) \
                and "src_stage" in c:
            # execution-only dependency (sync1): orders, but makes no memory visible
            self.q(qn).append(Barrier(None, None, None, where + " (execution dependency)",
                                      stage_mask(c["src_stage"]), frozenset(),
                                      stage_mask(c["dst_stage"]), frozenset(), fb_local=fbl))

    # -- main pass ---------------------------------------------------------------
    def run(self):
        self.prepass()
        m, host = self.m, self.q("host")
        self.queue_family = {}
        for r in self.recs:
            fn, seq = r["fn"], r["seq"]
            if r.get("warning"):
                m.warnings.append(f"seq {seq}: {fn}: {r['warning']}")
                if fn.startswith("vkCmd") and "cb" not in r and fn not in _NO_MEMORY_EFFECT:
                    # a trace from a driver that stubbed commands without their
                    # command buffer: where it ran is unknown
                    self.note_unsupported(fn, f"{fn} @seq {seq} (command buffer not recorded)", "memory",
                                          "command not modelled; its command buffer and submission are unknown",
                                          kind="command")
                elif not fn.startswith("vkCmd") and not fn.startswith(_HOST_NO_MEMORY_EFFECT) \
                        and fn not in _NO_MEMORY_EFFECT:
                    self.note_unsupported(fn, f"{fn} @seq {seq} (host)", "memory",
                                          "host call not modelled (no accesses or synchronization)",
                                          kind="command")
            if fn == "vkGetDeviceQueue":
                if r["queue"] not in m.queues:
                    qn = f"{FAMILY_NAMES.get(r['family'], 'family%d_' % r['family'])}{r['index']}"
                    m.queues[r["queue"]] = qn
                    self.queue_family[qn] = r["family"]
            elif fn == "vkCreateImage":
                ext = tuple(r.get("extent", ()))
                m.images[r["image"]] = ImageInfo(r["image"], self.n(r["image"]), r["format"], r["usage"], ext,
                                                 r.get("mip_levels", 1), r.get("array_layers", 1),
                                                 short_layout(r.get("initial_layout")))
                self.mips[r["image"]] = r.get("mip_levels", 1)
                self.layers[r["image"]] = r.get("array_layers", 1)
                self.img_samples[r["image"]] = r.get("samples", 1) or 1
                self.img_format[r["image"]] = r.get("format")
                if "EXCLUSIVE" in str(r.get("sharing", "")):
                    self.exclusive.update(self.img_res(r["image"]))
            elif fn == "vkCreateSwapchainKHR":
                sid, name = r["swapchain"], self.n(r["swapchain"])
                imgs = []
                for i, img in enumerate(r["images"]):
                    m.images[img] = ImageInfo(img, self.n(img), r["format"], r["usage"], tuple(r["extent"]),
                                              swapchain=name, swapchain_index=i)
                    self.img_format[img] = r["format"]
                    imgs.append(self.n(img))
                m.swapchains[name] = SwapchainInfo(sid, name, r["image_count"], r["min_image_count"],
                                                   r.get("surface_min_image_count", 2), r["format"],
                                                   tuple(r["extent"]), imgs, r.get("color_space"))
                self.fr.swapchains[name] = (r["image_count"], r.get("surface_min_image_count", 2))
            elif fn == "vkCreateImageView":
                m.views[r["view"]] = ViewInfo(r["view"], r["image"], r["format"], r.get("view_type", ""))
                self.view_image[r["view"]] = (r["image"], r["format"])
                self.view_range[r["view"]] = (r.get("base_mip", 0), r.get("level_count"))
                self.view_sub[r["view"]] = (r.get("aspect"), r.get("base_layer", 0), r.get("layer_count"))
            elif fn == "vkCreateBufferView":
                self.buffer_views[r["view"]] = (r["buffer"], r["format"])
                self.buffer_view_range[r["view"]] = (r.get("offset", 0), r.get("range"))
            elif fn == "vkCreateBuffer":
                m.buffers[r["buffer"]] = BufferInfo(r["buffer"], self.n(r["buffer"]), r["size"], r["usage"])
                self.buf_size[r["buffer"]] = r["size"]
                if "EXCLUSIVE" in str(r.get("sharing", "")):
                    self.exclusive.add(self.n(r["buffer"]))
            elif fn == "vkAllocateMemory":
                m.memories[r["memory"]] = MemoryInfo(r["memory"], r["size"], r.get("properties", []))
            elif fn in ("vkBindBufferMemory", "vkBindImageMemory"):
                kind = "buffer" if fn == "vkBindBufferMemory" else "image"
                obj = r[kind]
                mem = m.memories.get(r["memory"])
                if mem:
                    mem.bindings.append((kind, obj, r["offset"], r.get("size") or mem.size))
                    self.fresh_bind.setdefault(r["memory"], set()).add(obj)
                tgt = (m.buffers if kind == "buffer" else m.images).get(obj)
                if tgt:
                    tgt.memory = r["memory"]
                    if kind == "buffer":
                        tgt.offset = r["offset"]
            elif fn == "vkCreateShaderModule":
                m.shader_modules[r["module"]] = r["sha1"]
            elif fn == "vkCreateDescriptorSetLayout":
                m.set_layouts[r["layout"]] = r["bindings"]
            elif fn == "vkCreatePipelineLayout":
                self.pipeline_layouts[r["layout"]] = r["set_layouts"]
                self.push_ranges[r["layout"]] = [(frozenset(x["stages"]), x["offset"], x["size"])
                                                 for x in r.get("push_constants") or []]
            elif fn == "vkAllocateDescriptorSets":
                for s, l in zip(r["sets"], r["set_layouts"]):
                    self.set_layout_of[s] = l
                    self.set_contents[s] = {}
            elif fn == "vkUpdateDescriptorSets":
                for w in r["writes"]:
                    d = self.set_contents.setdefault(w["set"], {})
                    for i, el in enumerate(w["elements"]):
                        # consecutive elements may roll over into the next binding; kept simple
                        d[(w["binding"], w["array_element"] + i)] = (w["type"], el)
                for cp in r["copies"]:
                    src = self.set_contents.get(cp["src_set"], {})
                    dst = self.set_contents.setdefault(cp["dst_set"], {})
                    for i in range(cp["count"]):
                        k = (cp["src_binding"], cp["src_array_element"] + i)
                        if k in src:
                            dst[(cp["dst_binding"], cp["dst_array_element"] + i)] = src[k]
            elif fn == "vkCreateRenderPass":
                self.render_passes[r["render_pass"]] = r
            elif fn == "vkCreateFramebuffer":
                self.framebuffers[r["framebuffer"]] = r
            elif fn in ("vkCreateGraphicsPipelines", "vkCreateComputePipelines"):
                kind = "graphics" if fn == "vkCreateGraphicsPipelines" else "compute"
                for p in r["pipelines"]:
                    sl = [m.set_layouts.get(x, []) for x in self.pipeline_layouts.get(p["layout"], [])]
                    color = p.get("color_formats", [])
                    depth, stencil = p.get("depth_format"), p.get("stencil_format")
                    if p.get("render_pass") and p["render_pass"] in self.render_passes:
                        rp = self.render_passes[p["render_pass"]]
                        sp = rp["subpasses"][p.get("subpass", 0)]
                        atts = rp["attachments"]
                        color = [atts[x["attachment"]]["format"] if x["attachment"] >= 0 else "VK_FORMAT_UNDEFINED"
                                 for x in sp["color"]]
                        if sp.get("depth") and sp["depth"]["attachment"] >= 0:
                            depth = atts[sp["depth"]["attachment"]]["format"]
                    m.pipelines[p["pipeline"]] = PipelineInfo(
                        p["pipeline"], kind,
                        sorted([("|".join(s["stage"]), s["sha1"], s["entry"]) for s in p["stages"]],
                               key=lambda x: _stage_rank(x[0])),     # pipeline order, not pStages order
                        color, depth, stencil, p["layout"], sl, p.get("render_pass"),
                        p.get("vertex_bindings", []), p)
            elif fn == "vkCreateAccelerationStructureKHR":
                a = r.get("acceleration_structure")     # absent in traces of drivers that only stubbed it
                if a is not None:
                    m.accel[a] = AccelInfo(a, r["buffer"], r.get("offset", 0), r.get("size", 0),
                                           r.get("type", "generic"))
                    self.as_live.add(a)
            elif fn == "vkDestroyAccelerationStructureKHR":
                self.as_live.discard(r.get("handle"))
            elif fn == "covenant_tlas_refs":
                refs = r.get("refs") or []
                bad = [x for x in refs if x not in m.accel]
                if bad or not r.get("resolved", True):
                    self.note_unsupported("unresolved TLAS instance reference",
                                          f"TLAS build @seq {r.get('cmd_seq')} (submitted after seq {seq})",
                                          "approximated", "the build is taken to read every live BLAS")
                self.tlas_rec[(r["cb"], r["cmd_seq"], r.get("build", 0))] = \
                    None if bad or not r.get("resolved", True) else frozenset(refs)
            elif fn == "vkCreateRayTracingPipelinesKHR":
                for p in r.get("pipelines") or []:
                    sl = [m.set_layouts.get(x, []) for x in self.pipeline_layouts.get(p.get("layout"), [])]
                    stages = [("|".join(x["stage"]), x["sha1"], x["entry"]) for x in p.get("stages", [])]
                    for lib in p.get("libraries") or []:       # VK_KHR_pipeline_library: linked stages
                        if lib in m.pipelines:
                            stages += [x for x in m.pipelines[lib].stages if x not in stages]
                    m.pipelines[p["pipeline"]] = PipelineInfo(p["pipeline"], "ray_tracing", stages, [], None, None,
                                                              p.get("layout"), sl, None, [], p)
            elif fn in _AS_HOST:
                self.host_accel(r)
            elif fn == "vkSetEvent":
                self.host_events.add(r.get("event"))
            elif fn == "vkAllocateCommandBuffers":
                for cb in r["command_buffers"]:
                    self.cb_pool[cb] = r["pool"]
                    self.cb_cmds[cb] = []
            elif fn in ("vkBeginCommandBuffer", "vkResetCommandBuffer"):
                self.cb_cmds[r["cb"]] = []
                self.cb_reset(r["cb"], f"{fn} @seq {seq} on {self.n(r['cb'])}")
            elif fn == "vkResetCommandPool":
                for cb, pool in self.cb_pool.items():
                    if pool == r["pool"]:
                        self.cb_cmds[cb] = []
                        self.cb_reset(cb, f"{fn} @seq {seq} resets {self.n(cb)}")
            elif fn.startswith("vkCmd") and "cb" in r:
                self.cb_cmds.setdefault(r["cb"], []).append(r)
            # ---- synchronization
            elif fn == "vkCreateSemaphore":
                self.sem_type[r["semaphore"]] = r.get("type", "binary")
                self.sem_init[r["semaphore"]] = r.get("initial_value", 0)
            elif fn == "vkCreateFence":
                self.fences[r["fence"]] = {"gen": 0, "signal": None, "presignaled": bool(r.get("signaled"))}
            elif fn == "vkResetFences":
                for f in r["fences"]:
                    if f in self.fence_used:
                        # VUID-vkResetFences-pFences-01123: not while its signal is pending
                        self.use("host", f"fence:{self.n(f)}", "write", None,
                                 f"vkResetFences @seq {seq} on {self.n(f)}", "HOST", "HOST_FENCE_RESET")
                    st = self.fences.setdefault(f, {"gen": 0, "signal": None, "presignaled": False})
                    st.update(gen=st["gen"] + 1, signal=None, presignaled=False)
            elif fn in ("vkWaitForFences", "vkGetFenceStatus"):
                fences = r["fences"] if fn == "vkWaitForFences" else [r["fence"]]
                if fn == "vkWaitForFences" and not r.get("wait_all") and len(fences) > 1:
                    continue        # cannot know which fence released the wait
                for f in fences:
                    st = self.fences.setdefault(f, {"gen": 0, "signal": None, "presignaled": False})
                    if st["signal"]:
                        host.append(Wait(st["signal"], f"{fn} @seq {seq} on {self.n(f)}"))
                    elif st["presignaled"]:
                        continue
                    elif fn == "vkWaitForFences" and r.get("timeout") == "infinite":
                        host.append(Wait(f"{self.n(f)}@{st['gen']}",
                                         f"{fn} @seq {seq} on {self.n(f)}, which nothing will signal"))
            elif fn == "vkFreeDescriptorSets":
                for ds in r.get("sets") or []:
                    self.use("host", f"set:{self.n(ds)}", "write", None,
                             f"{fn} @seq {seq} frees {self.n(ds)}", "HOST", "HOST_FREE")
            elif fn == "vkSignalSemaphore":
                host.append(Signal(self.sem_signal(r["semaphore"], r["value"]), f"{fn} @seq {seq}"))
            elif fn == "vkGetSemaphoreCounterValue":
                # a counter query that returns v orders every signal of value <= v before the
                # host's later work, like a host wait for v (the model driver records v; in traces
                # without it the driver returned UINT64_MAX >> 1, i.e. every signal submitted so far)
                vals = self.tl_signals.get(r["semaphore"], [])
                v = r.get("value")
                if v is None:
                    v = max(vals) if vals else None
                elif vals:
                    v = min(v, max(vals))
                if self.sem_type.get(r["semaphore"]) == "timeline" and v:
                    self.sem_wait("host", r["semaphore"], v, f"{fn} @seq {seq} (counter query)")
            elif fn == "vkWaitSemaphores":
                sems = r["semaphores"]
                if r.get("wait_any") and len(sems) > 1:
                    continue
                for s in sems:
                    self.sem_wait("host", s["semaphore"], s["value"], f"{fn} @seq {seq}")
            elif fn == "vkQueueSubmit":
                self.submit(r)
            elif fn == "vkAcquireNextImageKHR":
                self.acquire(r)
            elif fn == "vkQueuePresentKHR":
                self.present(r)
            elif fn in ("vkDeviceWaitIdle", "vkQueueWaitIdle"):
                qs = [n for n in self.fr.queues if n != "host"] if fn == "vkDeviceWaitIdle" \
                    else [m.queues.get(r["queue"], r["queue"])]
                for qn in qs:
                    if self.fr.queues.get(qn):
                        name = f"idle@{seq}:{qn}"
                        self.q(qn).append(Signal(name, f"{fn} @seq {seq}"))
                        host.append(Wait(name, f"{fn} @seq {seq}: wait for {qn}"))
            elif fn == "host_write":
                self.host_write(r)
            elif fn == "covenant_capture_warning":
                # the driver reports that the page-protection handler was replaced (capture precondition P1)
                self.capture_bad = "host-write capture warning"
                self.note_unsupported("host-write capture: " + str(r.get("reason", "warning")),
                                      f"@seq {r.get('seq')}", "memory", "host writes may be missed after this point")
            if fn == "vkMapMemory" and r.get("tracked") is False:
                # the driver ran out of trap regions (MAXREG) or tracking is off: host
                # writes to this mapping are invisible (capture precondition P1)
                self.mem_bad.setdefault(r.get("memory"), "mapping without host-write capture")
                self.note_unsupported("mapped memory without host-write capture", f"vkMapMemory @seq {r.get('seq')}",
                                      "memory", "writes through this mapping are not recorded")
        self.resolve_timeline()
        self.check_pipeline_layouts()
        if self.legacy_rp_flags:
            # a trace from an older driver has no subpass dependencyFlags
            self.note_unsupported("subpass dependency flags not recorded", "vkCreateRenderPass*", "approximated",
                                  "trace from an older model driver: every explicit subpass dependency "
                                  "is taken as framebuffer-local (BY_REGION), the weaker reading"
                                  if not _LEGACY_GLOBAL else
                                  "trace from an older model driver: subpass dependencies taken as "
                                  "framebuffer-global (COVENANT_LEGACY_SUBPASS_FLAGS=global, not conservative)")
            m.unsupported["subpass dependency flags not recorded"]["count"] = self.legacy_rp_flags
        self.fr.unsupported = m.unsupported
        self.fr.host_capture = self.capture
        for img in m.images.values():
            for x in self.img_res(img.id):
                self.fr.initial_layouts[x] = img.initial_layout or "undefined"
                # the finite element range of the whole image
                self.fr.extents[x] = ((0, max(1, self.layers.get(img.id, 1)) * self._slots(img.id)),)
        m.frame = self.fr
        self.finish_guards()        # static IndexGuard failures (needs the whole frame)
        # how host writes were captured
        m.capture = {"mode": self.capture, "host_write_records": self.host_write_records,
                     "same_value_records": self.same_value_records,
                     "cross_thread_records": self.cross_thread_records,
                     "multi_thread_records": self.multi_thread_records}
        return m

    def qname(self, qid):
        return self.m.queues.get(qid, self.n(qid))

    def submit(self, r):
        seq, qn = r["seq"], self.qname(r["queue"])
        host, q = self.q("host"), self.q(qn)
        host.append(Signal(f"submit@{seq}", f"vkQueueSubmit @seq {seq} to {qn}"))
        q.append(Wait(f"submit@{seq}", f"vkQueueSubmit @seq {seq}"))
        for b in r["submits"]:
            for w in b["wait"]:
                self.sem_wait(qn, w["semaphore"], w.get("value"), f"submit @seq {seq} waits {self.n(w['semaphore'])}",
                              stage_mask(w.get("stage")))
            start, self.batch_releases = len(q), []
            for cb in b["command_buffers"]:
                self.expand(cb, qn, seq)
            for cb in b["command_buffers"]:
                # the command buffer is pending until the batch completes: a host
                # reset / re-record must be ordered after this (VUID-vkBeginCommandBuffer-commandBuffer-00049)
                self.cb_submitted.add(cb)
                self.use(qn, f"cmdbuf:{self.n(cb)}", "read", None,
                         f"vkQueueSubmit @seq {seq} executes {self.n(cb)} on {qn}", None, "CB_EXECUTE")
            for s in b["signal"]:
                self.check_signal_scope(s, q[start:], seq)
                sw = f"submit @seq {seq} signals {self.n(s['semaphore'])}"
                q.append(Signal(self.sem_signal(s["semaphore"], s.get("value"), sw), sw,
                                stages=stage_mask(s.get("stage"))))
        if r.get("fence"):
            self.fence_used.add(r["fence"])
            self.use(qn, f"fence:{self.n(r['fence'])}", "read", None,
                     f"vkQueueSubmit @seq {seq} signals {self.n(r['fence'])}", None, "FENCE_SIGNAL")
            q.append(Signal(self.fence_signal_name(r["fence"]), f"submit @seq {seq} signals {self.n(r['fence'])}"))

    def _layout_compatible(self, a, b, n):
        """Pipeline layouts a and b are compatible for set n: the same set
        layouts (by definition) for sets 0..n and the same push constant ranges."""
        if a == b:
            return True
        la, lb = self.pipeline_layouts.get(a), self.pipeline_layouts.get(b)
        if la is None or lb is None or len(la) <= n or len(lb) <= n:
            return False
        defn = lambda h: json.dumps(self.m.set_layouts.get(h), sort_keys=True)
        if any(defn(x) != defn(y) for x, y in zip(la[:n + 1], lb[:n + 1])):
            return False
        return sorted(map(str, self.push_ranges.get(a, []))) == sorted(map(str, self.push_ranges.get(b, [])))

    def check_set_compat(self, p, sets, bound, where):
        for si in sorted(sets):
            if si not in bound or bound[si][0] is None:
                continue
            lay, bwhere = bound[si]
            key = ("compat", p.id, si, lay)
            if key in self.push_checked:
                continue
            self.push_checked.add(key)
            if not self._layout_compatible(lay, p.layout, si):
                self.fr.findings.append((
                    "structure", "structure.incompatible-set-layout",
                    f"set {si} was bound with pipeline layout '{self.n(lay)}', which is not "
                    f"compatible for set {si} with '{self.n(p.layout)}' of pipeline "
                    f"'{self.n(p.id)}'; the binding is disturbed and the shader reads an "
                    "undefined set (VUID-vkCmdDraw-None-08600).", where,
                    [f"bound at: {bwhere}", "re-bind the set with this pipeline's layout, or make "
                     "the layouts share set layouts and push constant ranges"]))

    def check_pipeline_layouts(self):
        """Every binding a shader entry point uses must be declared in the
        pipeline layout's set layout, visible to that stage."""
        if not self.spirv_dir:
            return
        for pid, p in self.m.pipelines.items():
            for stage, sha, entry in p.stages:
                path = os.path.join(self.spirv_dir, f"{sha}.spv")
                if not os.path.exists(path):
                    continue
                try:
                    used = _spirv.entry_bindings(_spirv.parse(path), entry) or {}
                except Exception:
                    continue
                for (si, b) in sorted(used):
                    decl = None
                    if si < len(p.set_layouts):
                        decl = next((x for x in p.set_layouts[si] if x.get("binding") == b), None)
                    stages = set(stage.split("|"))
                    if decl is None:
                        why = f"set {si} of its layout has no binding {b}"
                    elif not stages & set(decl.get("stages") or []) and \
                            "VK_SHADER_STAGE_ALL" not in (decl.get("stages") or []):
                        why = f"binding {b} of set {si} is not visible to {stage}"
                    else:
                        continue
                    self.fr.findings.append((
                        "structure", "structure.binding-not-in-layout",
                        f"entry point '{entry}' of pipeline '{self.n(pid)}' uses set {si} binding "
                        f"{b}, but {why} (VUID-VkGraphicsPipelineCreateInfo-layout-07988); "
                        "whatever the shader reads there is undefined.",
                        f"pipeline '{self.n(pid)}'",
                        ["declare the binding in the set layout with this stage, or fix the "
                         "binding number on the host or in the shader"]))

    def check_push(self, c, where):
        """vkCmdPushConstants must match the layout's ranges byte by byte
        (VUID-vkCmdPushConstants-offset-01795 and -01796)."""
        key = (c["layout"], tuple(c["stages"]), c["offset"], c["size"])
        if key in self.push_checked or c["layout"] not in self.push_ranges:
            return
        self.push_checked.add(key)
        # the driver writes a flag word as the known VK_SHADER_STAGE_* names
        # plus one integer for the bits it has no name for (VK_SHADER_STAGE_ALL = 0x7FFFFFFF
        # leaves such a remainder); sorting a set of both would raise TypeError and abort the
        # check (VulkanSceneGraph programs use such masks). Both rules are applied to
        # the named stages only: bits without a name are no shader stage this checker knows
        # (VUID-vkCmdPushConstants-offset-01795/-01796 quantify over shader stages).
        named = lambda st: frozenset(x for x in st if isinstance(x, str))
        stages, lo, hi = named(c["stages"]), c["offset"], c["offset"] + c["size"]
        ranges = [(named(rs), o, n, None) for rs, o, n in self.push_ranges[c["layout"]]]
        short = lambda st: "|".join(sorted(x.replace("VK_SHADER_STAGE_", "").replace("_BIT", "")
                                           for x in st))
        problems = []
        for st in sorted(stages):
            bad = [b for b in range(lo, hi)
                   if not any(st in rs and o <= b < o + n for rs, o, n, _ in ranges)]
            if bad:
                problems.append(f"stage {short([st])} has no range covering bytes "
                                f"{bad[0]}..{bad[-1]}")
        for rs, o, n, rem in ranges:
            if o < hi and lo < o + n and not rs <= stages:
                problems.append(f"bytes {max(lo, o)}..{min(hi, o + n) - 1} overlap the "
                                f"{short(rs)} range ({o}..{o + n - 1}) but the call's stages "
                                f"{short(stages)} do not include all of {short(rs)}")
        if problems:
            self.fr.findings.append((
                "structure", "structure.push-constant-range",
                f"vkCmdPushConstants({short(stages)}, offset {lo}, size {c['size']}) does not match "
                f"the pipeline layout's push constant ranges: " + "; ".join(problems) + ".",
                where, ["push each stage's range with its own call, using exactly the "
                        "stages its range declares"]))

    def check_signal_scope(self, sig, ops, seq):
        """A semaphore signal with a stage mask orders only work in those stages.
        The release half of an ownership transfer ignores its dst stages, so
        only ALL_COMMANDS covers it. A layout transition earlier in submission
        order is ordered before the signal whatever the barrier's dst stages
        (timing_ref: LT => b through a later Signal on its queue; syncval
        agrees on its PRESENT_AFTER_WRITE check), so a
        transition to PRESENT with dstStage NONE / BOTTOM_OF_PIPE is fine."""
        S = stage_mask(sig.get("stage"))
        # BOTTOM_OF_PIPE in a first scope "is equivalent to ALL_COMMANDS with
        # VkAccessFlags set to 0" (synchronization.adoc:525-528, :744-747); a
        # semaphore signal's access scope is fixed, so it covers a release too
        if S is None or "ALL_COMMANDS" in S or "BOTTOM_OF_PIPE" in S:
            return
        name = self.n(sig["semaphore"])
        released = {r for r, _ in self.batch_releases}
        seen = set()
        for op in ops:
            if not isinstance(op, Barrier) or op.res is None or op.res in seen:
                continue
            if op.res in released:
                why = "it is the release half of an ownership transfer, whose dst stages are ignored"
            elif not S:
                why = "the signal's stage mask is empty, so its first synchronization scope holds no command"
            else:
                continue
            seen.add(op.res)
            self.fr.findings.append((
                "timing", "timing.signal-scope",
                f"semaphore '{name}' is signaled with stages {', '.join(sorted(S))}, which does not "
                f"cover the barrier on '{op.res}' ({why}); a queue waiting on '{name}' is not "
                "ordered after that layout transition.",
                f"vkQueueSubmit @seq {seq} signals {name}",
                [f"barrier: {op.where}", "signal with ALL_COMMANDS, or with a stage the "
                                         "barrier's dst stages include"]))

    def acquire(self, r):
        seq = r["seq"]
        sc = self.n(r["swapchain"])
        info = self.m.swapchains.get(sc)
        idx = r["image_index"]
        img = info.images[idx] if info and idx < len(info.images) else f"{sc}[{idx}]"
        where = f"vkAcquireNextImageKHR @seq {seq} ({img})"
        self.q("host").append(Acquire(sc, r.get("timeout") == "infinite", where))
        pq = self.q(f"present:{img}")
        for sem, (pimg, _) in list(self.present_wait.items()):
            if pimg == img:
                del self.present_wait[sem]      # the earlier present of img is done
        prev = self.last_present.get(img)
        if prev:
            pq.append(Wait(prev, where))
            self.use(f"present:{img}", img, "read", None, f"presentation engine shows {img} until {where}")
        elif os.environ.get("COVENANT_FIRST_ACQUIRE", "1") not in ("", "0"):
            # an image that was never presented has no presentation-engine
            # read pending, but "after acquiring a presentable image and before
            # modifying it, the application must use a synchronization primitive to
            # ensure that the presentation engine has finished reading from the image"
            # (wsi.adoc) holds for every acquire. Modelled as a read that ends at the
            # acquire's signal and reported as timing.acquire-first-use (a warning).
            self.use(f"present:{img}", img, "read", None,
                     f"{img} is owned by the presentation engine until {where} signals (first acquire)",
                     None, "PRESENT_FIRST_ACQUIRE")
        if r.get("semaphore"):
            if r["semaphore"] in self.sem_gpu_waited:
                self.use("host", f"semaphore:{self.n(r['semaphore'])}", "write", None,
                         f"{where} reuses {self.n(r['semaphore'])}", "HOST", "HOST_ACQUIRE_SEM")
            pq.append(Signal(self.sem_signal(r["semaphore"], None, where), where))
        if r.get("fence"):
            pq.append(Signal(self.fence_signal_name(r["fence"]), where))

    def present(self, r):
        # The presentation engine, not the presenting queue, waits on the
        # semaphores: model it as one pseudo-queue per swapchain image, so a
        # present orders nothing on the GPU queue and an acquire is ordered
        # only after the earlier present of the same image.
        seq, qn = r["seq"], self.qname(r["queue"])
        host = self.q("host")
        where = f"vkQueuePresentKHR @seq {seq} on {qn}"
        host.append(Signal(f"submit@{seq}", where))
        first = None
        for s in r["swapchains"]:
            sc = self.n(s["swapchain"])
            info = self.m.swapchains.get(sc)
            idx = s["image_index"]
            img = info.images[idx] if info and idx < len(info.images) else f"{sc}[{idx}]"
            pn = f"present:{img}"
            pq = self.q(pn)
            pq.append(Wait(f"submit@{seq}", where))
            if first is None:
                for w in r["wait"]:
                    self.sem_wait(pn, w["semaphore"], w.get("value"), where)
                pq.append(Signal(f"present-waits@{seq}", where))
                first = pn
            else:
                pq.append(Wait(f"present-waits@{seq}", where))
            self.use(pn, img, "read", "present", f"{where}: presents {img}")
            self.present_count += 1
            name = f"presented:{img}:{self.present_count}"
            pq.append(Signal(name, where))
            self.last_present[img] = name
            for w in r["wait"]:
                if self.sem_type.get(w["semaphore"]) != "timeline":
                    self.present_wait[w["semaphore"]] = (img, where)
            host.append(Present(sc, where))

    def index_region(self, st, c):
        """Bytes of the index buffer a draw reads: [first_index, first_index + index_count)
        indices after the bound offset (vkCmdDrawIndexed); the rest of the buffer otherwise."""
        off = st.get("ibo", 0) or 0
        if c["fn"] == "vkCmdDrawIndexed" and isinstance(c.get("index_count"), int):
            sz = st.get("ibt", 4)
            return self.buf_region(st["ib"], off + (c.get("first_index") or 0) * sz, c["index_count"] * sz)
        return self.buf_region(st["ib"], off, None)

    def host_write(self, r):
        """Host Use(write) of the bytes of every binding in the written pages.

        Page-straddle disambiguation: page protection only says which
        page was written. When a written page holds several bindings and some
        of them were bound since this memory's previous host_write record, the
        write is charged to those fresh bindings only: a bump sub-allocator
        (vkb BufferPool, VMA linear pools) places the buffer the host is about
        to fill right after the previous one, so the neighbour's tail sharing
        the page is not what the host wrote. This is a heuristic: a host write
        to the old neighbour's tail in the same page and the same interval is
        not seen. Every other page is charged to every binding in it.

        Records from a driver with shadow copies carry "writes" (the byte runs
        that changed); those are used instead of pages and need no heuristic."""
        mem = self.m.memories.get(r["memory"])
        fresh = self.fresh_bind.pop(r["memory"], set())
        if not mem:
            return
        # page-straddle disambiguation: a binding made by the very next call is fresh too. The driver
        # emits a host_write record at the start of the next call, so a bump
        # allocator's "write the data, then bind the buffer there" puts the bind
        # right after the record. Such bindings are not in mem.bindings yet; a page
        # they overlap is charged to fresh bindings only (none: to nothing).
        ahead = []
        nxt = self.next_rec.get(r["seq"] + 1)
        if nxt and nxt.get("fn") in ("vkBindBufferMemory", "vkBindImageMemory") and nxt.get("memory") == r["memory"]:
            kind = "buffer" if nxt["fn"] == "vkBindBufferMemory" else "image"
            ahead = [(kind, nxt[kind], nxt["offset"], nxt.get("size") or mem.size)]
        page = r.get("page_size", 4096)
        offs = r.get("offsets") or [r.get("offset", 0)]
        self.host_write_records += 1
        self._capture_threads(r)
        self._record_contents(r, page, offs)
        if r.get("writes") is not None and not r["writes"] and not r.get("writes_truncated"):
            # pages the trap saw written, no byte changed (a same-value store or a write
            # restored within the interval): counted in both modes; the observed mode emits
            # no access for them, the conservative mode does
            self.same_value_records += 1
        if self.capture == "conservative":
            # every page the protection trap saw written in this
            # capture interval is a write of every byte of every binding it overlaps,
            # changed or not (a store of the bytes already there, or a write restored
            # before the interval ended, leaves no diff but is an access). No
            # page-straddle heuristic: a neighbour's tail sharing the page is charged too.
            spans = [(o, o + page) for o in offs]
            hit = {}
            for o, end in spans:
                for kind, obj, off, size in list(mem.bindings) + ahead:
                    if not (off < end and o < off + size):
                        continue
                    if kind == "image":
                        hit[obj] = ("image", None)
                        continue
                    lo, hi = max(o, off), min(end, off + size)
                    hit.setdefault(obj, ("buffer", []))[1].append(self.buf_region(obj, lo - off, hi - lo))
            self._emit_host_writes(r, hit, offs)
            return

        def run_len(w):
            if len(w) >= 2 and w[1]:
                return len(w[1]) // 2
            if len(w) >= 3 and isinstance(w[2], int):
                return w[2]                 # length-only run past the driver's content budget
            return 0
        if r.get("writes") is not None:
            # byte runs that changed since the last flush (driver shadow copy): exact,
            # so no page attribution is needed (a store of the bytes already there is not seen)
            spans = sorted((int(w[0]), int(w[0]) + run_len(w)) for w in r["writes"] if run_len(w) > 0)
            spans = _coarsen_spans(spans, [x for b in list(mem.bindings) + ahead for x in (b[2], b[2] + b[3])])
            exact = [True] * len(spans)
            if r.get("writes_truncated"):
                # the runs are exact up to the truncation point; only the
                # bytes past it fall back to whole pages
                cut = r.get("truncated_at")
                if cut is None:
                    cut = max((e for _, e in spans), default=0)
                for o in offs:
                    if o + page > cut:
                        spans.append((max(o, cut), o + page))
                        exact.append(False)
        else:
            spans = [(o, o + page) for o in offs]
            exact = [False] * len(spans)
        hit = {}                    # obj -> (kind, region union)
        for (o, end), ex in zip(spans, exact):
            cands = [b for b in mem.bindings if b[2] < end and o < b[2] + b[3]]
            nxt_c = [b for b in ahead if b[2] < end and o < b[2] + b[3]]
            if not ex and len(cands) + len(nxt_c) > 1 and (fresh or nxt_c):
                new = [b for b in cands if b[1] in fresh]
                if (new or nxt_c) and len(new) < len(cands):
                    self.straddle += 1
                    cands = new
            for kind, obj, off, size in cands:
                if kind == "image":
                    hit[obj] = ("image", None)
                    continue
                lo, hi = max(o, off), min(end, off + size)
                hit.setdefault(obj, ("buffer", []))[1].append(self.buf_region(obj, lo - off, hi - lo))
        self._emit_host_writes(r, hit, offs)

    def _emit_host_writes(self, r, hit, offs):
        for obj, (kind, regs) in list(hit.items()):
            if kind == "buffer":        # one union of all runs (pairwise unions are quadratic)
                hit[obj] = (kind, None if any(x is None for x in regs) else
                            _region_coarsen(_region_union((), tuple(x for rg in regs for x in rg))))
        for obj, (kind, reg) in hit.items():
            res = self.n(obj)
            resl = self.img_res(obj) if kind == "image" else [(res, reg)]
            n0 = len(self.q("host"))
            self.use("host", resl, "write", None,
                     f"host write to {res} (mapped {self.n(r['memory'])} +0x{offs[0]:x}), "
                     f"before call seq {r['seq'] + 1}", "HOST", "HOST_WRITE")
            self._tag("host", n0, hseq=r["seq"])        # IndexGuard snapshot test

    def _capture_threads(self, r):
        """The capture-interval placement is exact when the thread that wrote is the
        thread of the Vulkan calls before and after the interval (capture precondition P3).
        The model driver records the faulting threads and the two calls' threads; a write from
        another thread is placed in trace order, which host happens-before need not
        follow, so it is outside the theorem and reported as a model limitation."""
        ft, ct = r.get("fault_tids"), r.get("call_tids")
        if ft is None or ct is None:
            return
        if (r.get("threads") or 0) > 1:
            # other threads existed: P3 is then checked only through the faulting threads
            # (a write by another thread to a page already unprotected in this interval
            # does not fault, so it is not attributed); counted, not reported
            self.multi_thread_records += 1
        if r.get("fault_tids_overflow") or any(t != ct[0] for t in ft) or (ct[1] is not None and ct[1] != ct[0]):
            self.cross_thread_records += 1
            self.mem_bad.setdefault(r.get("memory"), "host writes from another thread")
            self.note_unsupported("host writes from a thread other than the one making the surrounding "
                                  "Vulkan calls", f"host_write @seq {r['seq']}", "memory",
                                  "placed in trace order (capture precondition P3); host happens-before "
                                  "across threads is not recorded")

    def resolve_timeline(self):
        for qn, op, sem, v in self.tl_waits:
            vals = [x for x in self.tl_signals.get(sem, []) if x >= v]
            if self.sem_init.get(sem, 0) < v and vals:
                op.sem = f"{self.n(sem)}={min(vals)}"
            elif self.sem_init.get(sem, 0) >= v:
                ops = self.fr.queues[qn]
                for i, o in enumerate(ops):
                    if o is op:
                        del ops[i]
                        break


def load_records(path):
    out = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # a truncated last line (program crashed mid-write), or a corrupt record
                    out.append({"fn": "undecodable-record", "seq": out[-1]["seq"] if out else 0,
                                "warning": f"line {n} of the trace is not valid JSON; the record is skipped"})
    return out


def load_trace(path, spirv_dir=None) -> HostModel:
    """Read a model-driver trace and build the HostModel (with .frame).

    spirv_dir: directory of <sha1>.spv files (the driver writes them when
    COVENANT_SPIRV_DIR is set); used to treat storage bindings a compute shader
    declares NonWritable as reads. Defaults to $COVENANT_SPIRV_DIR."""
    if spirv_dir is None:
        spirv_dir = os.environ.get("COVENANT_SPIRV_DIR")
    return _Loader(load_records(path), spirv_dir).run()
