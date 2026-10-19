"""Pillar 3: whole-frame timing checks.

The frame is a set of queues (graphics, compute, transfer, the CPU as
"host", and one presentation-engine queue per swapchain image), each an
ordered list of operations, linked by semaphores and fences. The checker
computes happens-before with vector clocks and barrier chains and requires:

* two conflicting accesses (at least one write) to overlapping parts of one
  resource are ordered: by a chain of barriers on the same queue, by a
  semaphore/fence chain across queues (whose wait stage mask, or a barrier
  chained from it, must reach the later access), or by host program order.
  Read-after-write and write-after-write need a memory dependency (the
  earlier write in the first access scope of the chain's first dependency,
  the later access in the second access scope of its last one);
  write-after-read needs only an execution dependency, and an execution
  dependency on any resource orders it;
* a layout transition is a write of its subresources: it must be ordered
  after earlier accesses (by its own barrier's first scope or a chain into
  it) and later accesses after it (by a chain that starts at its own
  barrier, or a semaphore signalled after it);
* R-LT (synchronization.adoc:262-269, "synchronization-image-barrier-layout-
  transition-order"): "Layout transitions that are performed via image memory
  barriers execute in their entirety in submission order, relative to other
  image layout transitions submitted to the same queue, including those
  performed by render passes", so such a pair is never a race. The pair must
  include a transition performed by an image memory
  barrier; two automatic render-pass transitions (Barrier.auto) are checked
  like any other pair, since the text does not state their order. The ordering
  is not transitive: nothing else is ordered through it ("an application must
  not rely on ordering of image layout transitions to influence ordering of
  other commands"), so an access after the later transition must still be
  ordered after the earlier one by a dependency chain;
* framebuffer-local dependencies (Barrier.fb_local,
  dependencyFlags BY_REGION / VIEW_LOCAL) order a pair of framebuffer-space
  operations only at the same framebuffer position: the model orders such a
  pair only for two accesses with the same Use.fb token (attachment accesses
  through one image / view base), never storage, sampled or uniform accesses;
  pairs with a non-framebuffer-space stage are ordinary (see _Chain);
* every image is in the layout an access needs, and each barrier's "from"
  layout is the layout the image is really in;
* a loop is accepted when one iteration returns every resource to the state
  it started in (a fixed point). RCGP rejects any barrier inside a loop.

The rules are those of covenant/timing_ref.py, the executable reference
semantics; tests/test_timing_ref.py checks Haz_spec ⊆ Haz_C
on random frames and on recorded traces (Theorem 1).

The default is spec-strict. A layout transition is a write, and every
earlier write must be made available before it, whatever the old layout
(synchronization.adoc "Image Layout Transitions": "applications must ensure
that all memory writes have been made available before a layout transition is
executed"; UNDEFINED only lets the contents be discarded; the Khronos
synchronization examples keep a memory dependency for depth reuse through an
UNDEFINED transition). The relaxation R1 (``discard_relaxed``) is an opt-in heuristic, COVENANT_HEURISTIC_DISCARD=1
(covenant/modes.py), outside the theorem: a transition from UNDEFINED discards
the contents, so an earlier write needs only an execution dependency before
it, not availability. Its closure R1' (``discard_kill``, off by default):
the discarded write is also not checked against later accesses ordered after
the transition. R1' is off because it hides a real write-after-write: the
earlier write, never made available, can still land in memory after the
transition (the later clear / attachment write); syncval reports it
(bench case 64, SaschaWillems/Vulkan#1291).

Algorithm (FastTrack-style epochs). Each resource is an interval map over
its elements (bytes of a buffer; aspect-slot * layers + layer of one mip
level of an image). Every segment keeps the writes not yet proven ordered
before a later write, and per (queue, stage, render-pass scope) the latest
read since. A new access is checked only against those entries. An entry is
dropped once a later access is proven ordered after it, or a later access of
the same queue, stage, access flag and scope supersedes it; this is sound
because spec happens-before is transitive through an access that lies in
both scopes. Barrier-chain queries use per-entry chain states advanced
incrementally over the queue's barrier list (the barrier-frontier index),
so each barrier is visited once per live entry. ``exhaustive=True`` keeps
every entry (pairwise checking, used by the property tests).

Frames in flight are handled by unrolling several frames; per-frame copies
of a contract become separate resource instances.
"""
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from functools import lru_cache

from .diagnostics import Report


@dataclass(slots=True)     # ~5x smaller Use objects (long GSK traces hold 10M of them)
class Use:
    res: str
    access: str               # "read" | "write"
    layout: str = None        # required image layout, if an image
    where: str = ""
    stage: str = None         # pipeline stage, e.g. "FRAGMENT_SHADER" (None = any)
    flag: str = None          # access flag, e.g. "SHADER_SAMPLED_READ" (None = any)
    scope: object = None      # render pass instance / subpass, for attachment accesses
    call: object = None       # for a layout transition modelled as a Use: its barrier call
    struct: object = None     # ... and the struct within that call
    region: object = None     # None = whole resource; else ((lo, hi), ...) half-open element
                              # intervals: bytes of a buffer, aspect*layers+layer of an image mip
    cmd: object = None        # accesses of one command (same cmd) are never checked pairwise
    # the subpass / rendering instance the access is recorded in, and whether it
    # is an attachment load or store operation ("load" / "store"): a load
    # happens-before and a store happens-after every command of its subpass that
    # accesses the attachment, so such pairs are never checked
    instance: object = None
    pass_op: str = None
    # IndexGuard (descriptor-array indexing): cond: this access happens only if the guard fails (a
    # descriptor-array element the draw's decoded indices exclude); guard: this access (the
    # draw's vertex/instance read) carries the indices' decoding; hseq: trace seq of the
    # host_write record a host write comes from
    cond: object = None
    guard: object = None
    hseq: int = None
    # shader storage accesses that are only atomics / atomically allocated
    # slots / plain reads: a frozenset of (class, location), the location a
    # buffer byte range (spirv.physical_modes) or None (anywhere). Two accesses on
    # one queue with compatible modes (spirv.modes_compatible) are never checked
    # pairwise
    atomic: frozenset = None
    # the framebuffer position of an access performed by a
    # framebuffer-space operation at its own pixel (attachment reads/writes, input
    # attachment reads, load/store/resolve operations): a token naming the image
    # and the view base (mip, layer) that maps framebuffer coordinates to texels.
    # None for every other access (storage/sampled/uniform ... reads and writes reach
    # any texel or byte). A framebuffer-local dependency (BY_REGION / VIEW_LOCAL)
    # orders a pair of framebuffer-space accesses only if both carry the same token.
    fb: object = None


def atomic_exempt(a, b, slots=False):
    """Atomic accesses of two commands on one queue: no data race in the Vulkan
    memory model. Accesses through atomically allocated slots count only with the
    opt-in heuristic (slots=True, COVENANT_HEURISTIC_SLOTS=1), outside the verified
    configuration."""
    if a.atomic is None or b.atomic is None or a.queue != b.queue:
        return False
    from .spirv import modes_compatible
    return modes_compatible(a.atomic, b.atomic, slots)


@dataclass
class Barrier:
    """A pipeline barrier on one resource (res=None: a global memory barrier).
    Stage/access masks are sets of Vulkan names without prefixes, e.g.
    {"COLOR_ATTACHMENT_OUTPUT"}; None means "covers everything" (masks
    unknown)."""
    res: str
    old: str = None
    new: str = None
    where: str = ""
    src_stages: frozenset = None
    src_access: frozenset = None
    dst_stages: frozenset = None
    dst_access: frozenset = None
    # The structs of one vkCmdPipelineBarrier* call (or one subpass dependency
    # set) share one position: they never chain into each other, and a layout
    # transition is ordered only by its own struct. call=None: a call of its own.
    call: object = None
    struct: object = None
    region: object = None     # subresource region the memory barrier covers (None = whole)
    # Events (split barriers). event: the barrier is (part of) a vkCmdWaitEvents*
    # dependency on that event: its first synchronization scope holds only what
    # precedes the latest event_mark ("set", event) before it on the queue (none
    # if a ("reset", event) came after that set, or there was no set).
    # event_mark: this op is a vkCmdSetEvent* / vkCmdResetEvent* position marker
    # (all masks empty: it orders nothing itself)
    event: str = None
    event_mark: tuple = None
    # dependencyFlags has BY_REGION or VIEW_LOCAL. Then each
    # pair of a first-scope and a second-scope operation that are both in
    # framebuffer-space stages is ordered only framebuffer-locally (synchronization.adoc,
    # "Framebuffer Region Dependencies"): the model orders such a pair only when both
    # carry the same Use.fb token. A pair with a non-framebuffer-space stage on either
    # side is an ordinary dependency. fb: for a layout transition of such a barrier,
    # the token of the attachment it transitions (None: ordered only globally).
    # fb_local "none": VIEW_LOCAL with a non-zero viewOffset, whose framebuffer-space
    # pairs relate different views: they are not ordered at any position.
    fb_local: object = False
    fb: object = None
    # a render pass's automatic layout transition (initialLayout,
    # subpass, finalLayout). R-LT exempts a pair of transitions on one queue only if at
    # least one of them is performed by an image memory barrier.
    auto: bool = False


# ---------------------------------------------------------------- stage algebra
# Logical order of stages per pipeline (src scopes include earlier stages,
# dst scopes later ones; TOP/BOTTOM_OF_PIPE are handled separately).
GRAPHICS_ORDER = ["TOP_OF_PIPE", "DRAW_INDIRECT", "VERTEX_INPUT", "VERTEX_SHADER",
                  "TESSELLATION_CONTROL_SHADER", "TESSELLATION_EVALUATION_SHADER",
                  "GEOMETRY_SHADER", "EARLY_FRAGMENT_TESTS", "FRAGMENT_SHADER",
                  "LATE_FRAGMENT_TESTS", "COLOR_ATTACHMENT_OUTPUT", "BOTTOM_OF_PIPE"]
COMPUTE_ORDER = ["TOP_OF_PIPE", "DRAW_INDIRECT", "COMPUTE_SHADER", "BOTTOM_OF_PIPE"]
TRANSFER_ORDER = ["TOP_OF_PIPE", "TRANSFER", "BOTTOM_OF_PIPE"]
# sync2 transfer sub-stages: each is its own stage; TRANSFER (= ALL_TRANSFER) holds all of them
TRANSFER_SUBSTAGES = ("COPY", "BLIT", "CLEAR", "RESOLVE")
# VK_EXT_mesh_shader: the mesh primitive pipeline (vk.xml <syncpipeline name="graphics mesh">)
MESH_ORDER = ["TOP_OF_PIPE", "DRAW_INDIRECT", "TASK_SHADER", "MESH_SHADER", "EARLY_FRAGMENT_TESTS",
              "FRAGMENT_SHADER", "LATE_FRAGMENT_TESTS", "COLOR_ATTACHMENT_OUTPUT", "BOTTOM_OF_PIPE"]
# VK_KHR_ray_tracing_pipeline ("ray tracing") and VK_KHR_acceleration_structure
# ("acceleration structure build" / "... copy"): stand-alone pipelines
RAY_TRACING_ORDER = ["TOP_OF_PIPE", "DRAW_INDIRECT", "RAY_TRACING_SHADER", "BOTTOM_OF_PIPE"]
AS_BUILD, AS_COPY = "ACCELERATION_STRUCTURE_BUILD", "ACCELERATION_STRUCTURE_COPY"
AS_BUILD_ORDER = ["TOP_OF_PIPE", AS_BUILD, "BOTTOM_OF_PIPE"]
AS_COPY_ORDER = ["TOP_OF_PIPE", AS_COPY, "BOTTOM_OF_PIPE"]
# ACCELERATION_STRUCTURE_BUILD "specifies the execution of acceleration structure
# commands or acceleration structure copy commands" (VK_KHR_ray_tracing_maintenance1
# adds ACCELERATION_STRUCTURE_COPY for the copies alone): a mask naming BUILD also
# holds the copy commands' accesses, which the model records at stage COPY
STAGE_INCLUDES = {AS_BUILD: (AS_COPY,), "TRANSFER": TRANSFER_SUBSTAGES}
SHADER_STAGES = {"VERTEX_SHADER", "TESSELLATION_CONTROL_SHADER",
                 "TESSELLATION_EVALUATION_SHADER", "GEOMETRY_SHADER", "FRAGMENT_SHADER",
                 "COMPUTE_SHADER", "TASK_SHADER", "MESH_SHADER", "RAY_TRACING_SHADER"}
GRAPHICS_STAGES = frozenset(GRAPHICS_ORDER[1:-1]) | frozenset(MESH_ORDER[1:-1])
LT_STAGE, LT_FLAG = "LAYOUT_TRANSITION", "LAYOUT_TRANSITION_WRITE"
NON_GRAPHICS = {"COMPUTE_SHADER", "TRANSFER", "HOST", LT_STAGE} | set(TRANSFER_SUBSTAGES)
_FRAG_TESTS = frozenset({"EARLY_FRAGMENT_TESTS", "LATE_FRAGMENT_TESTS"})
# access-mask groups (sync2 definitions: UNIFORM_READ and INPUT_ATTACHMENT_READ
# are not part of SHADER_READ)
ACCESS_GROUPS = {
    "SHADER_READ": {"SHADER_SAMPLED_READ", "SHADER_STORAGE_READ", "SHADER_BINDING_TABLE_READ"},
    "SHADER_WRITE": {"SHADER_STORAGE_WRITE"},
    "MEMORY_READ": None,     # every read
    "MEMORY_WRITE": None,    # every write
}
_EARLIER, _LATER = {}, {}
for _order in (GRAPHICS_ORDER, COMPUTE_ORDER, TRANSFER_ORDER, MESH_ORDER, RAY_TRACING_ORDER,
               AS_BUILD_ORDER, AS_COPY_ORDER):
    _inner = _order[1:-1]
    for _i, _s in enumerate(_inner):
        _EARLIER.setdefault(_s, set()).update(_inner[:_i])
        _LATER.setdefault(_s, set()).update(_inner[_i + 1:])
_ALL_NONE = (True, frozenset())
_EMPTY = (False, frozenset())
# synchronization.adoc "Framebuffer Region Dependencies": the
# framebuffer-space pipeline stages. FB_LOCATED adds the pseudo-stage of a layout
# transition (split transitions, covenant/elide.py), which the model places at the
# framebuffer position of the attachment it transitions.
FB_STAGES = frozenset({"FRAGMENT_SHADER", "EARLY_FRAGMENT_TESTS", "LATE_FRAGMENT_TESTS", "COLOR_ATTACHMENT_OUTPUT"})
FB_LOCATED = FB_STAGES | {LT_STAGE}
# every stage name the model knows (for "every stage except the framebuffer-space
# ones"); the checker adds the stages of the frame it checks
_KNOWN = set(_EARLIER) | set(_LATER) | set(TRANSFER_SUBSTAGES) | {"TRANSFER", "HOST", AS_BUILD, AS_COPY} | \
    set(SHADER_STAGES) | set(GRAPHICS_STAGES)


def _located(stage):
    """May an access at `stage` be a framebuffer-space operation? (An unknown stage may.)"""
    return stage is None or stage in FB_LOCATED


def _fbpart(sc):
    """The framebuffer-located stages of an execution scope."""
    every, st = sc
    return (False, (FB_STAGES if every else frozenset()) | (st & FB_LOCATED))


def _nonfb(sc):
    """The other stages of an execution scope (explicit: an access of unknown stage is
    in it only if it is in no scope at all, i.e. never)."""
    every, st = sc
    return (False, (frozenset(_KNOWN - FB_LOCATED) if every else frozenset()) | (st - FB_LOCATED))


def _fz(mask):
    return mask if mask is None or isinstance(mask, frozenset) else frozenset(mask)


@lru_cache(maxsize=None)
def _es(mask, src):
    """Execution scope of a stage mask: (holds every stage, explicit stage set).
    ALL_COMMANDS, BOTTOM_OF_PIPE (src) and TOP_OF_PIPE (dst) hold every stage;
    TOP_OF_PIPE (src) and BOTTOM_OF_PIPE (dst) none; a stage brings the
    logically earlier (src) / later (dst) stages of its pipelines. The pseudo
    stage LAYOUT_TRANSITION is only ever in a scope that names it."""
    if mask is None:
        return _ALL_NONE
    every, out = False, set()
    for s in mask:
        if s == "ALL_COMMANDS" or s == ("BOTTOM_OF_PIPE" if src else "TOP_OF_PIPE"):
            every = True
        elif s in ("TOP_OF_PIPE", "BOTTOM_OF_PIPE"):
            continue
        elif s == "ALL_GRAPHICS":
            out |= GRAPHICS_STAGES
        else:
            out.add(s)
            out |= (_EARLIER if src else _LATER).get(s, set())
            out.update(STAGE_INCLUDES.get(s, ()))
    return (every, frozenset(out))


@lru_cache(maxsize=None)
def _as(mask):
    """Access scope of a stage mask: only the stages named (ALL_COMMANDS and
    ALL_GRAPHICS expand; TOP/BOTTOM_OF_PIPE carry no accesses)."""
    if mask is None:
        return _ALL_NONE
    every, out = False, set()
    for s in mask:
        if s == "ALL_COMMANDS":
            every = True
        elif s in ("TOP_OF_PIPE", "BOTTOM_OF_PIPE"):
            continue
        elif s == "ALL_GRAPHICS":
            out |= GRAPHICS_STAGES
        else:
            out.add(s)
            out.update(STAGE_INCLUDES.get(s, ()))
    return (every, frozenset(out))


def _in_scope(stage, sc):
    every, st = sc
    if stage is None:
        return every            # an access of unknown stage: only a scope of every stage
    if stage == LT_STAGE:
        return LT_STAGE in st
    return every or stage in st


def _in_access_scope(stage, sc):
    if _in_scope(stage, sc):
        return True
    # depth/stencil attachment accesses may happen in either test stage; a
    # mask naming either one holds them (modelling decision M1, as syncval)
    return stage in _FRAG_TESTS and not sc[1].isdisjoint(_FRAG_TESTS)


def _meet(a, b):
    """Do two execution scopes share a stage?"""
    if not a[1].isdisjoint(b[1]):
        return True
    if a[0] and b[0]:
        return True
    if a[0]:
        return bool(b[1] - {LT_STAGE})
    if b[0]:
        return bool(a[1] - {LT_STAGE})
    return not a[1].isdisjoint(b[1])


def _union(a, b):
    return (a[0] or b[0], a[1] | b[1])


def _stage_in(stage, mask, src):
    """Is an access at `stage` in the execution scope of `mask`?"""
    return _in_scope(stage, _es(_fz(mask), src))


def _explicit(stage, mask):
    """Is an access at `stage` in the access scope of `mask`?"""
    return _in_access_scope(stage, _as(_fz(mask)))


def _same_call(a, b):
    """True when a and b are different structs of one barrier call."""
    ca, cb = getattr(a, "call", None), getattr(b, "call", None)
    return ca is not None and ca == cb and getattr(a, "struct", None) != getattr(b, "struct", None)


def _access_in(flag, kind, mask):
    if mask is None:
        return True
    if ("MEMORY_READ" in mask and kind == "read") or ("MEMORY_WRITE" in mask and kind == "write"):
        return True
    if flag is None:
        return False
    if flag in mask:
        return True
    for g in mask:
        members = ACCESS_GROUPS.get(g)
        if members and flag in members and (kind == "write") == g.endswith("WRITE"):
            return True
    return False


def _stages_meet(a, b):
    """Does an execution dependency chain from dst scope a into src scope b exist?"""
    return _meet(_es(_fz(a), False), _es(_fz(b), True))


# ---------------------------------------------------------------- regions
def _disjoint_sorted(r):
    return all(r[k][1] <= r[k + 1][0] for k in range(len(r) - 1))


def _overlap(r1, r2):
    """Intersection of two regions (None = whole resource); () when empty."""
    if r1 is None:
        return r2
    if r2 is None:
        return r1
    if len(r1) * len(r2) > 64 and _disjoint_sorted(r1) and _disjoint_sorted(r2):
        # long interval lists (host writes recorded as byte runs): linear merge
        out, i, j = [], 0, 0
        while i < len(r1) and j < len(r2):
            (a, b), (c, d) = r1[i], r2[j]
            lo, hi = (a if a > c else c), (b if b < d else d)
            if lo < hi:
                out.append((lo, hi))
            if b < d:
                i += 1
            else:
                j += 1
        return tuple(out)
    out = []
    for a, b in r1:
        for c, d in r2:
            lo, hi = max(a, c), min(b, d)
            if lo < hi:
                out.append((lo, hi))
    return tuple(sorted(out))


def _covers(outer, inner):
    """Does region `outer` hold every element of `inner`?"""
    if outer is None:
        return True
    if inner is None:
        return False
    for lo, hi in inner:
        x = lo
        while x < hi:
            for a, b in outer:
                if a <= x < b:
                    x = b
                    break
            else:
                return False
    return True


def _res_scope(b, res, region):
    """Is (res, region) inside the memory-barrier scope of barrier b?"""
    return b.res is None or (b.res == res and _covers(b.region, region))


# ---------------------------------------------------------------- image layouts
# Layout matching is per aspect (Vulkan "image layout matching rules"):
# DEPTH_STENCIL_ATTACHMENT_OPTIMAL is DEPTH_ATTACHMENT_OPTIMAL for the depth
# aspect, ATTACHMENT_OPTIMAL / READ_ONLY_OPTIMAL (sync2) stand for the
# aspect-specific attachment / read-only layouts.
_LAYOUT_ASPECTS = {
    "depth_stencil_attachment": {"depth": "att", "stencil": "att"},
    "depth_attachment": {"depth": "att"},
    "stencil_attachment": {"stencil": "att"},
    "depth_stencil_read": {"depth": "ro", "stencil": "ro"},
    "depth_read_only": {"depth": "ro"},
    "stencil_read_only": {"stencil": "ro"},
    "depth_attachment_stencil_read_only": {"depth": "att", "stencil": "ro"},
    "depth_read_only_stencil_attachment": {"depth": "ro", "stencil": "att"},
    "color_attachment": {"*": "att"},
    "attachment": {"*": "att"},
    "read_only": {"*": "ro|shader_read"},
}


def layouts_match(a, b):
    """Is an image in layout b usable where layout a is required?"""
    if a == b or a is None or b is None:
        return True
    A, B = _LAYOUT_ASPECTS.get(a, {"*": a}), _LAYOUT_ASPECTS.get(b, {"*": b})
    for k in (set(A) | set(B)) - {"*"} or {"*"}:
        x, y = A.get(k, A.get("*")), B.get(k, B.get("*"))
        if x is not None and y is not None and not set(x.split("|")) & set(y.split("|")):
            return False
    return True


def _subtract(r1, r2):
    """Region r1 minus r2 (None = whole; whole minus a part stays whole-ish:
    returned as None only when nothing is removed)."""
    if r2 is None:
        return ()
    if r1 is None:
        r1 = ((-(1 << 62), 1 << 62),)
    out = []
    for lo, hi in r1:
        cur = [(lo, hi)]
        for a, b in r2:
            nxt = []
            for x, y in cur:
                if b <= x or y <= a:
                    nxt.append((x, y))
                    continue
                if x < a:
                    nxt.append((x, a))
                if b < y:
                    nxt.append((b, y))
            cur = nxt
        out.extend(cur)
    return tuple(out)


class _Layouts:
    """Current layout of each image subresource region: per resource a list
    of (region, layout), newest last."""

    def __init__(self, initial, extents=None):
        self.m = {r: [(None, l)] for r, l in initial.items()}
        self.ext = extents or {}        # res -> its finite element range (whole-image layout entries)

    def get(self, res, region=None):
        """Layouts the elements of region are in (empty: unknown)."""
        if region is None and res in self.ext:
            region = self.ext[res]      # a finite range, not an unbounded interval
        out, rem = [], region
        for reg, l in reversed(self.m.get(res, ())):
            if _overlap(rem, reg) == ():
                continue
            if l not in out:
                out.append(l)
            rem = _subtract(rem, reg)
            if rem == ():
                break
        return out

    def set(self, res, region, layout):
        if region is None:
            self.m[res] = [(None, layout)]
            return
        self.m[res] = [e for e in self.m.get(res, ()) if not _covers(region, e[0])] + [(region, layout)]


# ---------------------------------------------------------------- chain states
class _Chain:
    """The dependencies an access (or a semaphore wait) reaches through the
    barriers recorded after it on one queue, in execution mode (the access is
    in the first execution scope of the chain's first barrier) and in memory
    mode (also in its first access scope). Advanced incrementally over the
    queue's barrier list; structs of one call never chain into each other.

    Framebuffer-local barriers (Barrier.fb_local): the x_* /
    m_* fields hold what is reached globally (every operation of those stages);
    the lx_* / lm_* fields what is reached only framebuffer-locally: the
    framebuffer-space stages of a local barrier's second scope, reached by a path
    of local links from a framebuffer-located source (timing_ref classes G/N vs L).
    A local reach orders an access only if it carries the source's Use.fb token
    (self.fb). A local reach followed by an ordinary link is global again."""
    __slots__ = ("pos", "cur_call", "x_other", "x_cur", "x_total", "m_other", "m_cur", "m_ends",
                 "x_last", "m_last", "snap", "fb", "noloc",
                 "lx_other", "lx_cur", "lx_total", "lm_other", "lm_cur", "lm_ends", "lx_last", "lm_last")

    def __init__(self, seed=None, fb=None):
        self.pos = 0
        self.cur_call = None
        self.x_other = self.m_other = self.x_total = seed or _EMPTY
        self.x_cur, self.m_cur = {}, {}
        self.m_ends = {}                   # res (None = global) -> {(region, dst stages, dst access, part)}
        self.x_last = self.m_last = -1     # seq of the last barrier reached (globally)
        self.snap = {}                     # event -> (x, m, lx, lm) reached at its latest set marker
        self.fb = fb                       # the source access's framebuffer token
        self.noloc = False                 # diagnostics only: treat every barrier as global
        self.lx_other = self.lm_other = self.lx_total = _EMPTY
        self.lx_cur, self.lm_cur = {}, {}
        self.lm_ends = {}                  # res -> {(region, dst stages, dst access)}, framebuffer-local
        self.lx_last = self.lm_last = -1   # seq of the last barrier reached framebuffer-locally

    def _reached(self, other, cur):
        """other plus the sets reached in the current call, without the pseudo
        stage LAYOUT_TRANSITION: it links only the two halves of one split
        transition (one call, one struct), never two dependencies."""
        for v in cur.values():
            other = _union(other, (v[0], v[1] - {LT_STAGE}) if LT_STAGE in v[1] else v)
        return other

    def advance(self, bars, upto, src):
        """Process bars[self.pos:upto] ((seq, Barrier) pairs) for the source
        access `src` (an _Acc), or for a semaphore wait (src None)."""
        while self.pos < upto:
            bs, b = bars[self.pos]
            self.pos += 1
            if b.event_mark is not None:
                # vkCmdSetEvent*: what this chain has reached so far is what an
                # event wait on it can chain from (the access itself is before it)
                kind, ev = b.event_mark
                if kind == "set":
                    self.snap[ev] = (self._reached(self.x_other, self.x_cur), self._reached(self.m_other, self.m_cur),
                                     self._reached(self.lx_other, self.lx_cur),
                                     self._reached(self.lm_other, self.lm_cur))
                else:
                    self.snap.pop(ev, None)
                continue
            call = b.call
            if call is None or call != self.cur_call:
                self.x_other = self._reached(self.x_other, self.x_cur)
                self.m_other = self._reached(self.m_other, self.m_cur)
                self.lx_other = self._reached(self.lx_other, self.lx_cur)
                self.lm_other = self._reached(self.lm_other, self.lm_cur)
                self.x_cur, self.m_cur, self.lx_cur, self.lm_cur = {}, {}, {}, {}
                self.cur_call = call if call is not None else object()
            es_src = _es(b.src_stages, True)
            k = b.struct
            xc, mc, lxc, lmc = self.x_cur.get(k), self.m_cur.get(k), self.lx_cur.get(k), self.lm_cur.get(k)
            x_from = _union(self.x_other, xc) if xc else self.x_other
            m_from = _union(self.m_other, mc) if mc else self.m_other
            lx_from = _union(self.lx_other, lxc) if lxc else self.lx_other
            lm_from = _union(self.lm_other, lmc) if lmc else self.lm_other
            dx = dm = False
            ev_ok = True
            if b.event is not None:
                # an event wait: only what was reached at the event's set marker
                sn = self.snap.get(b.event)
                ev_ok = sn is not None
                x_from, m_from, lx_from, lm_from = sn if ev_ok else (_EMPTY, _EMPTY, _EMPTY, _EMPTY)
            located = False
            if src is not None and src.lt is not None and bs == src.seq:
                dx = dm = True                       # a transition: only its own barrier
                located = True
            elif src is not None and ev_ok and not _same_call(src, b):
                if src.lt is not None:
                    dx = dm = bs == src.seq          # a transition: only its own barrier
                    located = True
                elif _in_scope(src.stage, es_src):
                    dx = True
                    dm = src.kind == "write" and _in_access_scope(src.stage, _as(b.src_stages)) and \
                        _access_in(src.flag, "write", b.src_access) and _res_scope(b, src.res, src.region)
                    located = _located(src.stage)
            local = b.fb_local and not self.noloc
            if local:
                gx = (dx and not located) or _meet(x_from, es_src)
                lx = (dx and located) or _meet(lx_from, es_src)
                gm = (dm and not located) or _meet(m_from, es_src)
                lm = (dm and located) or _meet(lm_from, es_src)
            else:
                gx = dx or _meet(x_from, es_src) or _meet(lx_from, es_src)
                gm = dm or _meet(m_from, es_src) or _meet(lm_from, es_src)
                lx = lm = False
            if not (gx or lx):
                continue
            # fb_local "none" (VIEW_LOCAL with a non-zero view offset): framebuffer-space
            # pairs are at different views, so no position is ordered (only class N)
            nopos = b.fb_local == "none"
            es_dst = _es(b.dst_stages, False)
            if gx:
                self.x_cur[k] = _union(self.x_cur.get(k) or _EMPTY, es_dst)
                self.x_total = _union(self.x_total, es_dst)
                self.x_last = bs
            if lx:
                nf, fp = _nonfb(es_dst), _fbpart(es_dst)
                self.x_cur[k] = _union(self.x_cur.get(k) or _EMPTY, nf)      # class N: global
                self.x_total = _union(self.x_total, nf)
                if not nopos:
                    self.lx_cur[k] = _union(self.lx_cur.get(k) or _EMPTY, fp)    # class L
                    self.lx_total = _union(self.lx_total, fp)
                    self.lx_last = bs
            if gm:
                self.m_cur[k] = _union(self.m_cur.get(k) or _EMPTY, es_dst)
                self.m_ends.setdefault(b.res, set()).add((b.region, b.dst_stages, b.dst_access, None))
                self.m_last = bs
            if lm:
                self.m_cur[k] = _union(self.m_cur.get(k) or _EMPTY, _nonfb(es_dst))
                self.m_ends.setdefault(b.res, set()).add((b.region, b.dst_stages, b.dst_access, "N"))
                if not nopos:
                    self.lm_cur[k] = _union(self.lm_cur.get(k) or _EMPTY, _fbpart(es_dst))
                    self.lm_ends.setdefault(b.res, set()).add((b.region, b.dst_stages, b.dst_access))
                    self.lm_last = bs

    def reaches(self, u, memory, overlap):
        """Is access u in the second (access) scope of a dependency reached?"""
        loc = self.fb is not None and u.fb is not None and u.fb == self.fb
        if not memory:
            return _in_scope(u.stage, self.x_total) or (loc and _in_scope(u.stage, self.lx_total))
        for res in (None, u.res):
            for region, ds, da, part in self.m_ends.get(res, ()):
                es = _es(ds, False)
                if part == "N":
                    es = _nonfb(es)
                if (res is None or _covers(region, overlap)) and \
                        _in_scope(u.stage, es) and _in_access_scope(u.stage, _as(ds)) and \
                        _access_in(u.flag, u.kind, da):
                    return True
            if not loc:
                continue
            for region, ds, da in self.lm_ends.get(res, ()):
                if (res is None or _covers(region, overlap)) and \
                        _in_scope(u.stage, _fbpart(_es(ds, False))) and _in_access_scope(u.stage, _as(ds)) and \
                        _access_in(u.flag, u.kind, da):
                    return True
        return False

    def reached_any(self, memory):
        """Every stage set reached, globally or locally (a signal's first scope
        chains from either: a local link followed by the signal is global)."""
        if memory:
            r = _union(self._reached(self.m_other, self.m_cur), self._reached(self.lm_other, self.lm_cur))
        else:
            r = _union(self.x_total, self.lx_total)
        return r


class _Acc:
    """One access event: a Use, or the layout transition of a Barrier."""
    __slots__ = ("queue", "seq", "kind", "stage", "flag", "scope", "res", "region", "op", "where",
                 "cmd", "lt", "call", "struct", "chain", "instance", "pass_op", "atomic", "hid", "fb")

    def __init__(self, queue, seq, op, lt=False):
        self.queue, self.seq, self.op = queue, seq, op
        self.res, self.region, self.where = op.res, op.region, op.where
        self.call, self.struct = op.call, op.struct
        self.chain = self.hid = None        # hid: entries a transition dropped (R-LT)
        if lt:
            self.lt, self.kind, self.stage, self.flag = op, "write", LT_STAGE, LT_FLAG
            self.scope = self.cmd = self.instance = self.pass_op = self.atomic = None
            self.fb = op.fb                 # the attachment position it transitions
        else:
            self.lt, self.kind, self.stage, self.flag = None, op.access, op.stage, op.flag
            self.scope, self.cmd = op.scope, op.cmd
            self.instance, self.pass_op = op.instance, op.pass_op
            self.atomic = getattr(op, "atomic", None)
            self.fb = getattr(op, "fb", None)

    @property
    def access(self):
        return self.kind

    @property
    def key(self):
        return (self.queue, self.seq)


class _Wait:
    __slots__ = ("seq", "sig", "es", "full", "chain", "sigx")

    def __init__(self, seq, sig, es, nbars, sigx=None):
        self.seq, self.sig, self.es = seq, sig, es
        # sigx = {queue m: ((t, flt), ...)}: the ops of m with seq <= t that the
        # signal this wait consumed holds in its first synchronization scope, along the
        # whole relay path. flt is None (every such op) or the (queue, seq, barrier
        # count, src scope, where) of the stage-masked signal of m at the first hop
        # (then only the ops in its stage scope, _in_signal_scope). Masks of later hops
        # are applied when the relay is built (_Checker._sig_relay), so every hop's mask
        # filters the relay, not only the last hop's.
        self.sigx = sigx if sigx is not None else {m: ((t, None),) for m, t in sig.items()}
        self.full = es[0]
        # no "submission" flag: a wait on a host-signalled semaphore is not taken as a
        # queue submission (every later op of the queue in its second scope). That holds for the loader's submit@N waits, which have no stage mask
        # and so are full waits anyway, but not for a device wait on a value the host
        # signalled with vkSignalSemaphore at a restricted pWaitDstStageMask, whose second
        # scope is limited by that mask like any other semaphore wait
        # (cmdbuffers.adoc, VkSemaphoreSubmitInfo / pWaitDstStageMask).
        self.chain = _Chain(es)
        self.chain.pos = nbars


class _ResState:
    """Interval map of one resource: segment i holds the elements in
    [cuts[i-1], cuts[i]); each segment is [writes, reads, shadow]: writes a
    list of _Acc, reads a dict (queue, stage, scope) -> [_Acc, ...] (latest
    first), shadow a list of (entry, the access that dominated it, scope
    generation, command generation): entries still needed by accesses exempt
    from their dominator (same command or same render-pass scope). Exhaustive
    mode only: a fourth item maps id(write) -> the discard transitions that
    killed it there (R1')."""
    __slots__ = ("cuts", "segs")

    def __init__(self):
        self.cuts = []
        self.segs = [[[], {}, [], {}]]

    def _split(self, x):
        i = bisect_left(self.cuts, x)
        if i < len(self.cuts) and self.cuts[i] == x:
            return
        w, r, sh, k = self.segs[i]
        self.cuts.insert(i, x)
        self.segs.insert(i + 1, [list(w), {a: list(v) for a, v in r.items()}, list(sh),
                                 {a: list(v) for a, v in k.items()}])

    def segments(self, region):
        """Indices of the segments that make up region (splitting as needed)."""
        if region is None:
            return range(len(self.segs))
        for lo, hi in region:
            self._split(lo)
            self._split(hi)
        out = set()
        for lo, hi in region:
            out.update(range(bisect_right(self.cuts, lo), bisect_right(self.cuts, hi - 1) + 1))
        return sorted(out)


# ---------------------------------------------------------------- frame ops
def covers_chain(barriers, first, second, first_op=None):
    """Is access `first` = (stage, flag, kind) ordered before `second` by the
    barriers between them (all on one queue, in submission order)? Kept for
    callers outside the checker; the barriers' resources are not checked."""
    s1, f1, k1 = first
    s2, f2, k2 = second
    a = _Acc(None, 0, Use(None, k1, None, "", s1, f1, call=getattr(first_op, "call", None),
                          struct=getattr(first_op, "struct", None)))
    bars = [(i + 1, Barrier(None, None, None, "", b.src_stages, b.src_access, b.dst_stages,
                            b.dst_access, b.call, b.struct)) for i, b in enumerate(barriers)]
    ch = _Chain()
    ch.advance(bars, len(bars), a)
    return ch.reaches(_Acc(None, 0, Use(None, k2, None, "", s2, f2)), k1 == "write", None)


def covers(b, first, second):
    return covers_chain([b], first, second)


@dataclass
class Signal:
    sem: str
    where: str = ""
    # first synchronization scope of a sync2 semaphore signal (VkSemaphoreSubmitInfo::stageMask);
    # None = every stage (vkQueueSubmit, fences, presents, idle waits)
    stages: frozenset = None


@dataclass
class Wait:
    sem: str
    where: str = ""
    stages: frozenset = None   # dst stage mask of a semaphore wait (None = all)


@dataclass
class Acquire:
    swapchain: str
    infinite_timeout: bool = True
    where: str = ""


@dataclass
class Present:
    swapchain: str
    where: str = ""


@dataclass
class Loop:
    body: list
    where: str = ""


@dataclass
class Frame:
    queues: dict = field(default_factory=dict)        # name -> [ops]
    initial_layouts: dict = field(default_factory=dict)
    # element range of each whole image resource, so that a layout map entry for the whole
    # resource (region None) is covered by per-layer entries that together cover every element
    extents: dict = field(default_factory=dict)
    swapchains: dict = field(default_factory=dict)    # name -> (image_count, min_image_count)
    # (semaphore, image, present, re-signal): a semaphore a present waited on,
    # signaled again before that image was re-acquired
    semaphore_reuse: list = field(default_factory=list)
    findings: list = field(default_factory=list)      # (category, code, message, where, notes)
    # what the trace holds but the model does not cover (host_trace: never
    # dropped silently): what -> {"count", "first", "effect", "detail"};
    # effect "memory" (accesses/barriers missing: a warning), "untracked",
    # "approximated" (meta only)
    unsupported: dict = field(default_factory=dict)
    # host-write capture mode the trace was loaded with (host_trace; None: no loader)
    host_capture: str = None

    def q(self, name):
        return self.queues.setdefault(name, [])


def _expand(ops, report, layouts_probe):
    """Replace loops by two copies of their body (enough to expose hazards
    between the end of one iteration and the start of the next) and check
    the fixed-point rule."""
    out = []
    for op in ops:
        if isinstance(op, Loop):
            start = dict(layouts_probe)
            for b in op.body:
                if isinstance(b, Barrier) and b.new is not None:
                    layouts_probe[b.res] = b.new
            for r, l in layouts_probe.items():
                if start.get(r) != l:
                    report.add("timing", "timing.loop-not-fixed-point",
                               f"one iteration leaves '{r}' in layout '{l}' but the loop "
                               f"starts with it in '{start.get(r)}', so the second "
                               f"iteration's barrier is wrong.", op.where,
                               ["make each iteration end in the layout it starts in"])
            layouts_probe.clear()
            layouts_probe.update(start)
            out.extend(_expand(op.body, report, layouts_probe))
            out.extend(_expand(op.body, report, dict(layouts_probe)))
        else:
            if isinstance(op, Barrier) and op.new is not None:
                layouts_probe[op.res] = op.new
            out.append(op)
    return out


def rcgp_would_reject(frame):
    """True if any loop contains a barrier (RCGP's foreach rule, App. C.4)."""
    def walk(ops):
        for op in ops:
            if isinstance(op, Loop):
                if any(isinstance(b, Barrier) for b in op.body) or walk(op.body):
                    return True
        return False
    return any(walk(ops) for ops in frame.queues.values())


def check_swapchains(frame, ops, report):
    """Vulkan lets an app hold at most image_count - min_image_count + 1
    images at once; one more acquire with an infinite timeout may never
    return (the freeze seen in RCGP's samples on a real GPU)."""
    held = {}
    for op in ops:
        if isinstance(op, Acquire):
            n, m = frame.swapchains.get(op.swapchain, (None, None))
            held[op.swapchain] = held.get(op.swapchain, 0) + 1
            if n is not None and held[op.swapchain] > n - m + 1:
                sev = "can block forever" if op.infinite_timeout else "will time out"
                report.add("timing", "timing.swapchain-overacquire",
                           f"this acquire would hold {held[op.swapchain]} images of "
                           f"'{op.swapchain}' at once, but only {n - m + 1} may be held "
                           f"({n} images, minimum {m}); it {sev}.", op.where,
                           ["present (or wait for) an earlier image before acquiring "
                            "another, or limit frames in flight"])
        elif isinstance(op, Present):
            held[op.swapchain] = max(0, held.get(op.swapchain, 0) - 1)


# host operations that need an object's earlier GPU use to have completed (host_trace models
# the use as a read of a pseudo-resource and the host operation as a write)
IN_USE_FLAGS = {
    "HOST_CB_RESET": ("timing.command-buffer-in-use", "reset or re-recorded",
                      "VUID-vkBeginCommandBuffer-commandBuffer-00049"),
    "HOST_FENCE_RESET": ("timing.fence-in-use", "reset", "VUID-vkResetFences-pFences-01123"),
    "HOST_ACQUIRE_SEM": ("timing.acquire-semaphore-in-use", "passed to vkAcquireNextImageKHR",
                         "VUID-vkAcquireNextImageKHR-semaphore-01779"),
}


def _verb(a):
    return "written" if a == "write" else "read"


def _masks(m):
    return ", ".join(sorted(m or [])) or "NONE"


class IndexGuard:
    """The schedule-independence premise of one draw's decoded descriptor indices.

    The loader decoded the indices from the host-written bytes of the draw's vertex /
    instance buffer as of the draw's vkQueueSubmit (trace seq ``snap``) and charged the
    elements outside them as ``Use(cond=guard)``. The guard *fails*, and those Uses are
    then checked like any other access, if

    * ``static`` is set by the loader (bytes not host-determined: device writes or aliased
      images on that memory, incomplete capture, unknown bytes, ...), or
    * the guard read (``Use(guard=guard)``) is in a reported hazard (a write to the
      vertex/instance bytes unordered with the draw), or
    * a host write recorded after ``snap`` is live on those bytes when the guard read is
      checked (it may be ordered before the draw, e.g. through a host-signalled timeline
      semaphore, so the draw may read bytes other than the decoded ones).

    Whether it fails is decided per timing.check run (barrier edits by report_groups /
    elide can change it): check() re-runs with the failed guards widened until no new
    guard fails."""
    __slots__ = ("snap", "static", "where")

    def __init__(self, snap, where=""):
        self.snap, self.static, self.where = snap, None, where


class _GSet:
    """The index guards an epoch read entry stands for (its own and those of the reads it
    superseded), as a union-find set so that a long run of superseding reads (thousands of draws
    reading one vertex buffer) stays linear: `gs` and `snap` (the least snapshot) live at the root,
    `nf` counts the members already marked failed."""
    __slots__ = ("up", "gs", "snap", "nf")

    def __init__(self, g):
        self.up, self.gs, self.snap, self.nf = None, [g], g.snap, 0

    def root(self):
        r = self
        while r.up is not None:
            r = r.up
        while self.up is not None and self.up is not r:     # path compression
            self.up, self = r, self.up
        return r

    def merge(self, other):
        a, b = self.root(), other.root()
        if a is b:
            return a
        if len(a.gs) < len(b.gs):
            a, b = b, a
        b.up = a
        a.gs.extend(b.gs)
        a.snap = min(a.snap, b.snap)
        b.gs = None
        return a

    def fail(self, into):
        r = self.root()
        if r.nf < len(r.gs):
            into.update(r.gs[r.nf:])
            r.nf = len(r.gs)


# ---------------------------------------------------------------- the checker
class _Checker:
    def __init__(self, frame, report, hazards, exhaustive, discard_relaxed, discard_kill=False, slots=False):
        self.report, self.hazards = report, hazards
        self.slots = slots
        self.exhaustive, self.discard_relaxed = exhaustive, discard_relaxed
        self.discard_kill = discard_kill and discard_relaxed
        self.names = list(frame.queues)
        self.qs = {n: _expand(frame.queues[n], report, dict(frame.initial_layouts)) for n in self.names}
        for ops in self.qs.values():        # stage names for _nonfb
            for op in ops:
                if isinstance(op, Use) and op.stage and op.stage not in _KNOWN and op.stage != LT_STAGE:
                    _KNOWN.add(op.stage)
        self.vc = {n: {m: 0 for m in self.names} for n in self.names}
        self.bars = {n: [] for n in self.names}           # queue -> [(seq, Barrier)]
        self.bar_seqs = {n: [] for n in self.names}       # queue -> [seq] (for bisect)
        self.waits = {n: [] for n in self.names}          # queue -> [_Wait]
        self.fullmax = {n: {} for n in self.names}        # queue -> {m: latest m-seq seen by a full wait}
        self.relay_full = {n: {} for n in self.names}     # merged sigx of n's full waits
        self.relay_any = {n: {} for n in self.names}      # ... of all of n's waits
        self.state = {}                                   # res -> _ResState
        self.scope_gen = {n: 0 for n in self.names}       # bumped when a queue leaves a render-pass scope
        self.cmd_gen = {n: 0 for n in self.names}         # bumped when a queue leaves a command
        self.cur_scope = {n: None for n in self.names}
        self.cur_cmd = {n: None for n in self.names}
        self.inst_gen = {n: 0 for n in self.names}        # bumped when a queue leaves a subpass instance
        self.cur_inst = {n: None for n in self.names}
        self.layouts = _Layouts(frame.initial_layouts, frame.extents)
        self.pairs_checked = 0
        self.ev_set = {n: {} for n in self.names}         # queue -> {event: seq of its latest set}
        self.ev_first = {}                                # (queue, seq) of an event-wait barrier -> set seq
        # index guards widened in this run (given), and those found failing / seen in it
        self.failed_guards, self.guard_failed, self.guards_seen = set(), set(), set()
        self.premise_host = 0        # pairs accepted under premise A-HOST only
        self.premise_pairs = None    # tests: [(device write where, host write where)]
        self.gsets = {}     # epoch entry (_Acc) -> _GSet of the guards it stands for

    # -- ordering ---------------------------------------------------------
    def _chain_of(self, e):
        if e.chain is None:
            e.chain = _Chain(fb=e.fb)
            # from the first barrier at or after the access (a transition's own barrier)
            e.chain.pos = bisect_left(self.bar_seqs[e.queue], e.seq)
        return e.chain

    def _wait_reaches(self, e, u, memory, overlap):
        """Is e (on queue m) ordered before u (on queue n) through a semaphore
        wait on n whose signaller had seen e, by the wait's stage mask or a
        barrier chain from it?"""
        n, m, s = u.queue, e.queue, e.seq
        # a wait whose stage mask holds every stage (ALL_COMMANDS, or
        # TOP_OF_PIPE in a second scope) has "all operations performed by all
        # commands" of its batch and later in its second synchronization scope
        # (synchronization.adoc:441-442/699-700 ALL_COMMANDS, :521-524/740-743
        # TOP_OF_PIPE; cmdbuffers.adoc:2387-2393 vkQueueSubmit, :2210-2214
        # VkSemaphoreSubmitInfo), so it orders layout transitions too, like a
        # queue submission.
        if self.fullmax[n].get(m, 0) >= s:
            return True
        bars = self.bars[n]
        self.relay_broken = True
        for w in reversed(self.waits[n]):
            ok = False
            for t, flt in w.sigx.get(m, ()):
                if t < s:
                    continue
                self.relay_broken = False
                if flt is not None and not self._in_signal_scope(e, flt):
                    self.flt_rejected = flt
                    continue
                ok = True
                break
            if not ok:
                continue
            if w.full:
                return True                                 # transitions too
            if u.lt is not None:
                w.chain.advance(bars, len(bars), None)      # through u's own barrier
                if w.chain.x_last == u.seq:
                    return True
                continue
            if _in_scope(u.stage, w.es):
                return True
            w.chain.advance(bars, len(bars), None)
            if w.chain.reaches(u, memory, overlap):
                return True
        return False

    # -- semaphore relays ------------------------------------------------------
    _RELAY_KEEP = 4         # stage-filtered entries kept per queue (fewer = fewer orderings: sound)

    @classmethod
    def _merge_relay(cls, acc, sx, skip=None):
        """acc[m] |= sx[m] for m != skip, pruned: one unfiltered entry (the largest t),
        and only filtered entries newer than it (at most _RELAY_KEEP, the newest).
        Dropping an entry only removes orderings, never adds one."""
        for m, ents in sx.items():
            if m == skip:
                continue
            cur = acc.get(m, ())
            if not cur:
                acc[m] = tuple(ents)
                continue
            allx = cur + tuple(ents)
            top = max((t for t, f in allx if f is None), default=0)
            filt = sorted({(t, f) for t, f in allx if f is not None and t > top},
                          key=lambda x: (-x[0], x[1][1]))[:cls._RELAY_KEEP]
            acc[m] = (((top, None),) if top else ()) + tuple(filt)

    def _sig_relay(self, n, seq, ses, where):
        """What a signal on queue n at seq holds in its first synchronization scope,
        per queue: n's own earlier ops (all of them, or only those in the stage mask's
        scope), and what n's earlier semaphore waits received, but only through a wait
        whose second scope (or a barrier chain from it on n) meets the signal's first
        scope: synchronization.adoc, semaphore signal operation ("the first
        synchronization scope includes ... limited to operations on the pipeline stages
        determined by the source stage mask") and dependency chains
        (synchronization-dependencies-chains). A host wait or a queue submission
        holds every later operation."""
        own_flt = None if ses[0] else (n, seq, len(self.bars[n]), ses, where)
        out = {n: ((seq, own_flt),)}
        if ses[0]:
            self._merge_relay(out, self.relay_any[n])
            return out
        bars = self.bars[n]
        if ses[1] - {LT_STAGE}:
            # a wait of every stage meets any non-empty scope (an empty one, e.g. a
            # signal with TOP_OF_PIPE alone, holds nothing and relays nothing)
            self._merge_relay(out, self.relay_full[n])
        for w in self.waits[n]:
            if w.full:
                continue
            w.chain.advance(bars, len(bars), None)
            if _meet(w.chain.x_total, ses):
                self._merge_relay(out, w.sigx)
        return out

    def _in_signal_scope(self, e, flt):
        """Is e (on the signalling queue, before the signal) in the first
        synchronization scope of a signal with a stage mask? Directly (its stage
        is in the mask's execution scope; the signal's first access scope then
        holds its accesses), or through a barrier chain on that queue whose last
        dst scope meets the mask (memory mode for a write: the chain must make it
        available). A layout transition is ordered only through its own
        barrier's dst scope, unless the mask holds every stage."""
        q, _, nbars, es_sig, _ = flt
        if es_sig[0]:
            return True
        if e.lt is None and _in_scope(e.stage, es_sig):
            return True
        ch = _Chain(fb=e.fb)
        ch.pos = bisect_left(self.bar_seqs[q], e.seq)
        ch.advance(self.bars[q], nbars, e)
        # the write must be made available by the chain's first link (memory
        # mode); later links, up to the signal, need only chain (synchronization.adoc,
        # availability in dependency chains); the last barrier before the signal need
        # not name e's resource (requiring it would be stricter than the spec).
        # A framebuffer-local reach chains into the signal too (the signal is an
        # ordinary dependency, so the chain is global from there)
        return _meet(ch.reached_any(e.kind == "write"), es_sig)

    def _ordered(self, e, u, overlap):
        """None when e is ordered before u, else the kind of failure:
        'same' (one queue), 'wait' (the queues are synchronized, but not up to
        that stage), 'race' (the queues are not synchronized)."""
        self.pairs_checked += 1
        self.flt_rejected = None
        self.relay_broken = False
        m, n = e.queue, u.queue
        memory = e.kind == "write" and not (self.discard_relaxed and u.lt is not None and u.lt.old is None)
        if m == n == "host":
            return None
        if m == n:
            ch = self._chain_of(e)
            bars = self.bars[n]
            ch.advance(bars, len(bars), e)
            if u.lt is not None:
                if (ch.m_last if memory else ch.x_last) == u.seq:
                    return None
                b = u.lt                # captured by the transition's own barrier, on the overlap
                # reached only framebuffer-locally: the same attachment position
                same_pos = e.fb is not None and e.fb == u.fb
                if same_pos and (ch.lm_last if memory else ch.lx_last) == u.seq:
                    return None
                if b.event is not None and e.seq >= self.ev_first.get((n, u.seq), 0):
                    pass                # an event wait: e is not before the event's set
                elif not _same_call(e, b) and e.lt is None and _in_scope(e.stage, _es(b.src_stages, True)) and \
                        (not memory or (_in_access_scope(e.stage, _as(b.src_stages)) and
                                        _access_in(e.flag, "write", b.src_access) and
                                        _res_scope(b, e.res, overlap))) and \
                        (not b.fb_local or not _located(e.stage) or (same_pos and b.fb_local != "none")):
                    return None
            elif ch.reaches(u, memory, overlap):
                return None
            if self._wait_reaches(e, u, memory, overlap):
                return None
            return "same"
        if self.vc[n][m] < e.seq:
            return "race"
        if self._wait_reaches(e, u, memory, overlap):
            return None
        return "wait"

    # -- premise A-HOST: the device-to-host domain operation ---------------------
    class _HostAcc:
        __slots__ = ("stage", "flag", "kind", "res", "fb")

        def __init__(self, res, flag):
            self.stage, self.flag, self.kind, self.res, self.fb = "HOST", flag, flag[5:].lower(), res, None

    def _host_domain(self, e, u):
        """A device write e is ordered before the host write u by a host wait. The
        wait gives an execution dependency only: a fence's access scope holds device
        accesses only (synchronization.adoc, "Signaling a fence and waiting on the host
        does not guarantee ..."), and e is made available to the host domain only by a
        memory dependency whose destination access mask holds HOST_READ or HOST_WRITE
        (synchronization.adoc, memory domain operations). True when e's memory chain on
        its own queue reaches such a barrier at or before the last position of that queue
        that a host wait holds through signals of every stage (Full[host][m]: the barrier
        then chains into that wait). Domain operations on another queue, or reached only
        through stage-masked signals, are not credited (the pair is then reported:
        conservative)."""
        m = e.queue
        bound = self.fullmax["host"].get(m, 0)
        bars = self.bars[m]
        ch = _Chain(fb=e.fb)
        ch.pos = bisect_left(self.bar_seqs[m], e.seq)
        upto = bisect_left(self.bar_seqs[m], bound + 1)
        if upto <= ch.pos:
            return False
        ch.advance(bars, upto, e)
        ov = _overlap(e.region, u.region)
        return any(ch.reaches(self._HostAcc(e.res, f), True, ov) for f in ("HOST_READ", "HOST_WRITE"))

    def _report_host_domain(self, e, u):
        self.premise_host += 1
        if self.premise_pairs is not None:
            self.premise_pairs.append((e.where, u.where))
        st_e = LT_STAGE if e.lt is not None else (e.stage or "any stage")
        fl_e = "layout transition" if e.lt is not None else (e.flag or "any access")
        self.report.add(
            "timing", "timing.premise-host-domain",
            f"premise A-HOST: the host write to '{u.res}' is ordered after the device write "
            f"({st_e}, {fl_e}, {e.where}) only by a host wait. The wait is an execution "
            "dependency; no barrier with HOST in dstStageMask and HOST_READ or HOST_WRITE in "
            "dstAccessMask makes the device write available to the host domain, so the pair is "
            "accepted under premise A-HOST only (no guarantee).",
            u.where,
            ["add a barrier with dstStageMask HOST and dstAccessMask HOST_READ or HOST_WRITE "
             "after the device write, before the signal the host waits on"],
            severity="warning")

    # -- diagnostics: which scope test failed ---------------------------------
    # Every same-queue report names the test that fails, in the spec's terms:
    # the first synchronization scope (execution: srcStageMask), the first access
    # scope (availability: srcStageMask x srcAccessMask, and the barrier's resource),
    # the second synchronization scope (dstStageMask) and the second access scope
    # (visibility: dstStageMask x dstAccessMask). Naming only the stage mask would make
    # a missing srcAccessMask under srcStageMask ALL_COMMANDS read as "ALL_COMMANDS
    # does not include RAY_TRACING_SHADER".
    _SPEC_LT = ("synchronization.adoc, Image Layout Transitions: 'applications must ensure that all "
                "memory writes have been made available before a layout transition is executed'")

    def _explain_lt_after(self, e, u):
        """Why the transition u (same queue) is not ordered after e: (message, fix)."""
        b = u.lt
        memory = e.kind == "write" and not (self.discard_relaxed and b.old is None)
        st_e = LT_STAGE if e.lt is not None else (e.stage or "any stage")
        fl_e = "layout transition" if e.lt is not None else (e.flag or "any access")
        ch = self._chain_of(e)
        ch.advance(self.bars[u.queue], len(self.bars[u.queue]), e)
        direct_x = not _same_call(e, b) and e.lt is None and _in_scope(e.stage, _es(b.src_stages, True))
        exec_ok = direct_x or ch.x_last == u.seq
        head = (f"the layout transition of '{u.res}' (to '{b.new}'"
                f"{', from UNDEFINED' if b.old is None else ''}) is not ordered after the earlier "
                f"{e.kind} ({st_e}, {fl_e})")
        if not exec_ok or not memory:
            return (head + f": the barrier's first synchronization scope (srcStageMask "
                    f"{_masks(b.src_stages)}) does not include {st_e}, and no barrier chain reaches the "
                    "transition, so the transition can race it.",
                    f"add {st_e if e.lt is None else 'the earlier barrier dst stage'} to srcStageMask"
                    + (f" and {fl_e} to srcAccessMask" if memory and e.lt is None else ""))
        # executed after e, but e is never made available
        bad = []
        if not _in_access_scope(e.stage, _as(b.src_stages)):
            bad.append(f"its stage {st_e} is not in the access scope of srcStageMask {_masks(b.src_stages)}"
                       + (" (TOP/BOTTOM_OF_PIPE carry no accesses)" if b.src_stages and
                          set(b.src_stages) <= {"TOP_OF_PIPE", "BOTTOM_OF_PIPE"} else ""))
        if not _access_in(e.flag, "write", b.src_access):
            bad.append(f"srcAccessMask ({_masks(b.src_access)}) does not include {fl_e}")
        if not _res_scope(b, e.res, _overlap(e.region, u.region)):
            bad.append(f"the barrier's subresource range does not hold all of the written region")
        why = "; ".join(bad) or "no memory dependency chain makes it available"
        und = (" This applies to a transition from UNDEFINED too: the old contents may be discarded, "
               "but the earlier write must still be made available, or it can land after the "
               "transition.") if b.old is None else ""
        head = (f"the earlier write ({st_e}, {fl_e}) is not made available before the layout transition "
                f"of '{u.res}' (to '{b.new}'{', from UNDEFINED' if b.old is None else ''})")
        return (head + ": the transition is ordered after the write by an execution dependency (the first "
                f"synchronization scope {_masks(b.src_stages)} holds {st_e}), but the barrier's first "
                f"access scope does not hold the write: {why}. "
                f"A layout transition needs every earlier write made available ({self._SPEC_LT}).{und}",
                f"add {fl_e} to srcAccessMask (with {st_e} in srcStageMask)")

    def _explain_lt_before(self, e, u):
        """Why access u is not ordered after the transition e (same queue)."""
        lt = e.lt
        memory = True                   # a transition is a write: RAW / WAW need visibility
        what = f"{'write' if u.kind == 'write' else 'read'} ({u.stage or 'any stage'}, {u.flag or 'any access'})"
        head = f"the layout transition of '{u.res}' (to '{lt.new}') does not reach this {what}"
        x_ok = _in_scope(u.stage, _es(lt.dst_stages, False))
        ch = self._chain_of(e)
        ch.advance(self.bars[u.queue], len(self.bars[u.queue]), e)
        if not (x_ok or ch.reaches(u, False, None)):
            return (head + f": its own second synchronization scope (dstStageMask {_masks(lt.dst_stages)}) "
                    f"does not include {u.stage or 'any stage'}, and no barrier chained from it does, so "
                    "the access can race the transition.",
                    f"add {u.stage or 'the stage'} to the transition's dstStageMask and "
                    f"{u.flag or 'the access'} to its dstAccessMask")
        bad = []
        if not _in_access_scope(u.stage, _as(lt.dst_stages)):
            bad.append(f"the stage {u.stage} is not in the access scope of dstStageMask {_masks(lt.dst_stages)}")
        if not _access_in(u.flag, u.kind, lt.dst_access):
            bad.append(f"dstAccessMask ({_masks(lt.dst_access)}) does not include {u.flag or 'the access'}")
        why = "; ".join(bad) or "no barrier chained from it has the access in its second access scope"
        head = (f"the writes of the layout transition of '{u.res}' (to '{lt.new}') are not made visible "
                f"to this {what}")
        return (head + ": the access is ordered after the transition by an execution dependency, but it is "
                f"not in the transition's second access scope ({why}).",
                f"add {u.flag or 'the access'} to the transition's dstAccessMask")

    def _explain_between(self, e, u, between):
        """Why the barrier(s) between e and u (same queue, u no transition) do not order them."""
        memory = e.kind == "write"
        st_e, fl_e = e.stage or "any stage", e.flag or "any access"
        st_u, fl_u = u.stage or "any stage", u.flag or "any access"
        ov = _overlap(e.region, u.region)
        ch = self._chain_of(e)
        ch.advance(self.bars[u.queue], len(self.bars[u.queue]), e)
        kind_u = _verb(u.kind) if u.kind == "write" else "read"
        head = (f"the barrier on '{u.res}' does not order the earlier {e.kind} ({st_e}, {fl_e}) "
                f"before this {kind_u} ({st_u}, {fl_u})")
        b = between[-1]
        many = len(between) > 1
        where = f"any of the {len(between)} barriers between them" if many else "the barrier"
        x_reached = ch.x_last >= 0
        if not x_reached:
            return (head + f": {st_e} is not in the first synchronization scope of {where} "
                    f"(srcStageMask {_masks(b.src_stages)}{' of the last one' if many else ''}).",
                    f"add {st_e} to srcStageMask" + (f" and {fl_e} to srcAccessMask" if memory else ""))
        if not ch.reaches(u, False, ov):
            if u.stage is not None and _in_scope(u.stage, _es(b.dst_stages, False)) and \
                    not _in_scope(e.stage, _es(b.src_stages, True)):
                # the last barrier's second scope holds u, so the chain breaks at its first scope:
                # it neither holds e nor meets what the earlier barrier(s) reached
                every, st = ch.x_total
                got = "every stage" if every else (", ".join(sorted(st)[:6]) + (", ..." if len(st) > 6 else "")
                                                     or "no stage")
                return (head + f": the earlier access is in the first synchronization scope of an earlier "
                        f"barrier, whose second scope reaches {got}. The last barrier on '{u.res}' before this "
                        f"access has {st_u} in its dstStageMask ({_masks(b.dst_stages)}), but its first synchronization "
                        f"scope (srcStageMask {_masks(b.src_stages)}) holds neither {st_e} nor a stage the "
                        "earlier barrier reached, so the dependency chain is broken there.",
                        f"add {st_e} to the srcStageMask of the last barrier on '{u.res}' before this access"
                        + (f" and {fl_e} to its srcAccessMask" if memory else ""))
            return (head + f": the earlier access is in the first synchronization scope, but {st_u} is not "
                    f"in the second synchronization scope of {'the chain it starts' if many else 'the barrier'} "
                    f"(dstStageMask {_masks(b.dst_stages)}{' of the last barrier' if many else ''}).",
                    f"add {st_u} to dstStageMask" + (f" and {fl_u} to dstAccessMask" if memory else ""))
        if not memory:
            return (head + ".", "")         # (not reached: WAR needs only execution)
        if ch.m_last < 0:
            bad = []
            if not _in_access_scope(e.stage, _as(b.src_stages)):
                bad.append(f"{st_e} is not in the access scope of srcStageMask {_masks(b.src_stages)}"
                           + (" (TOP/BOTTOM_OF_PIPE carry no accesses)" if b.src_stages and
                              set(b.src_stages) <= {"TOP_OF_PIPE", "BOTTOM_OF_PIPE"} else ""))
            if not _access_in(e.flag, "write", b.src_access):
                bad.append(f"srcAccessMask ({_masks(b.src_access)}) does not include {fl_e}")
            if not _res_scope(b, e.res, e.region):
                bad.append("the barrier's resource or range does not hold the written region")
            return (head + ": they are ordered by an execution dependency only; the write is not made "
                    f"available, because it is not in the first access scope of {where} "
                    f"({'; '.join(bad) or 'no barrier has it'}).",
                    f"add {fl_e} to srcAccessMask")
        bad = []
        if not _in_access_scope(u.stage, _as(b.dst_stages)):
            bad.append(f"{st_u} is not in the access scope of dstStageMask {_masks(b.dst_stages)}")
        if not _access_in(u.flag, u.kind, b.dst_access):
            bad.append(f"dstAccessMask ({_masks(b.dst_access)}) does not include {fl_u}")
        if not _res_scope(b, u.res, ov):
            bad.append("the barrier's resource or range does not hold the accessed region")
        return (head + ": the write is made available, but not visible to this access, because it is not "
                f"in the second access scope of {where} ({'; '.join(bad) or 'of the chain end'}).",
                f"add {fl_u} to dstAccessMask")

    def _local_only(self, e, u):
        """The last framebuffer-local barrier between e and u (one queue) if
        e would be ordered before u with every barrier taken as framebuffer-global,
        else None (diagnostics only)."""
        if e.queue != u.queue:
            return None
        bars = self.bars[u.queue]
        loc = [b for bs, b in bars if bs >= e.seq and b.fb_local]
        if not loc:
            return None
        overlap = _overlap(e.region, u.region)
        memory = e.kind == "write" and not (self.discard_relaxed and u.lt is not None and u.lt.old is None)
        ch = _Chain(fb=e.fb)
        ch.noloc = True
        ch.pos = bisect_left(self.bar_seqs[e.queue], e.seq)
        ch.advance(bars, len(bars), e)
        if u.lt is not None:
            b = u.lt
            ok = (ch.m_last if memory else ch.x_last) == u.seq or (
                not _same_call(e, b) and e.lt is None and b.event is None and
                _in_scope(e.stage, _es(b.src_stages, True)) and
                (not memory or (_in_access_scope(e.stage, _as(b.src_stages)) and
                                _access_in(e.flag, "write", b.src_access) and _res_scope(b, e.res, overlap))))
        else:
            ok = ch.reaches(u, memory, overlap)
        return loc[-1] if ok else None

    # -- reporting ----------------------------------------------------------
    def _report(self, e, u, why):
        if self.gsets:
            for x in (e, u):        # a hazard on a guard read: its decoded indices are not
                gx = self.gsets.get(x)              # schedule-independent
                if gx is not None:
                    gx.fail(self.guard_failed)
        m, n, acc, w = e.queue, u.queue, e.kind, e.where
        st_e = LT_STAGE if e.lt is not None else (e.stage or "any stage")
        fl_e = "layout transition" if e.lt is not None else (e.flag or "any access")
        loc = self._local_only(e, u) if why == "same" and "PRESENT_FIRST_ACQUIRE" not in (e.flag, u.flag) else None
        if loc is not None:
            # ordered only by a framebuffer-local dependency
            what_u = "layout transition" if u.lt is not None else \
                f"{'write' if u.kind == 'write' else 'read'} ({u.stage or 'any stage'}, {u.flag or 'any access'})"
            what_e = "layout transition" if e.lt is not None else f"{e.kind} ({st_e}, {fl_e})"
            if e.fb is None or u.fb is None:
                why_l = ("at least one of them is not an attachment access at its own framebuffer position "
                         "(a storage, sampled or uniform access, or a transition outside a render pass, can "
                         "touch memory that a fragment at another position accessed)")
            elif loc.fb_local == "none":
                why_l = "the dependency's view offset relates different views"
            else:
                why_l = "they access the attachment through different framebuffer mappings"
            msg = (f"the {what_u} of '{u.res}' is ordered after the earlier {what_e} only by a "
                   f"framebuffer-local dependency (dependencyFlags BY_REGION or VIEW_LOCAL). Between "
                   "framebuffer-space stages such a dependency orders operations only at the same framebuffer "
                   f"position (synchronization.adoc, Framebuffer Region Dependencies), and {why_l}.")
            notes = [f"dependency: {loc.where}", f"earlier access: {w}",
                     "drop VK_DEPENDENCY_BY_REGION_BIT / VK_DEPENDENCY_VIEW_LOCAL_BIT from that dependency, or "
                     "add a framebuffer-global dependency between the two accesses"]
            self.report.add("timing", "timing.barrier-scope", msg, u.where, notes)
            if self.hazards is not None:
                self.hazards.append((e.key, u.key, "timing.barrier-scope"))
            return
        if why == "same" and e.lt is not None and u.lt is not None and e.lt.auto and u.lt.auto:
            # two automatic render-pass transitions, no chain between them
            msg = (f"the render pass's automatic layout transition of '{u.res}' (to '{u.lt.new}') is not ordered "
                   f"after an earlier automatic render-pass transition of it (to '{e.lt.new}') by any dependency "
                   "chain. The submission-order guarantee for layout transitions (synchronization.adoc, "
                   "synchronization-image-barrier-layout-transition-order) covers transitions 'performed via image "
                   "memory barriers' relative to other transitions, including render passes' ones; it does not "
                   "state that two render-pass transitions are ordered, so the model does not assume it.")
            notes = [f"earlier access: {w}", "add a subpass dependency to/from VK_SUBPASS_EXTERNAL (or a barrier) "
                     "whose scopes chain the two render passes' attachment accesses"]
            self.report.add("timing", "timing.barrier-scope", msg, u.where, notes)
            if self.hazards is not None:
                self.hazards.append((e.key, u.key, "timing.barrier-scope"))
            return
        if "PRESENT_FIRST_ACQUIRE" in (e.flag, u.flag):
            # first use of a never-presented swapchain image
            img = u.res
            other = w if e.flag == "PRESENT_FIRST_ACQUIRE" else u.where
            mine = u.where if e.flag == "PRESENT_FIRST_ACQUIRE" else w
            what = 'layout transition' if (u.lt is not None or e.lt is not None) else 'write'
            if u.flag == "PRESENT_FIRST_ACQUIRE":
                # the modification comes first in submission order: the image is used before
                # vkAcquireNextImageKHR returned it at all
                msg = (f"'{img}' is modified ({what}) before vkAcquireNextImageKHR returned it. The image "
                       "was not presented before, so no presentation-engine read is pending in practice, "
                       "but the spec allows use of a presentable image only between its acquire and its present.")
                notes = [f"acquire: {other}", f"modified by: {mine}",
                         "wsi.adoc: 'Use of a presentable image must occur only after the image is returned by "
                         "vkAcquireNextImageKHR, and before it is released by vkQueuePresentKHR. This includes "
                         "transitioning the image layout and rendering commands.'",
                         "move the transition after the acquire, ordered after its semaphore wait"]
            else:
                msg = (f"'{img}' is modified ({what}) "
                       "without being ordered after the semaphore or fence of the vkAcquireNextImageKHR that "
                       "returned it. The image was not presented before, so no presentation-engine read is "
                       "pending in practice, but the spec requires the wait for every acquired image.")
                notes = [f"acquire: {other}", f"modified by: {mine}",
                         "wsi.adoc: 'After acquiring a presentable image and before modifying it, the application "
                         "must use a synchronization primitive to ensure that the presentation engine has finished "
                         "reading from the image'", "include the acquire semaphore wait's stage (usually "
                         "COLOR_ATTACHMENT_OUTPUT) in the transition's srcStageMask"]
            self.report.add("timing", "timing.acquire-first-use", msg, mine, notes, severity="warning")
            if self.hazards is not None:
                self.hazards.append((e.key, u.key, "timing.acquire-first-use"))
            return
        if u.lt is not None:
            op = u.lt
            if why == "same":
                code = "timing.barrier-scope"
                msg, fix = self._explain_lt_after(e, u)
                notes = [f"earlier access: {w}", fix]
                if op.event is not None:
                    notes.append(f"the barrier waits on event '{op.event}': only what precedes the event's "
                                 "latest vkCmdSetEvent* is in its first scope")
            elif why == "wait":
                code = "timing.wait-stage-scope"
                msg = (f"the layout transition of '{u.res}' is not ordered after the earlier "
                       f"{_verb(acc)} on '{m}': the semaphore wait that orders the queues only "
                       f"blocks later stages, and this barrier's first scope "
                       f"({_masks(op.src_stages)}) does not chain to it.")
                notes = [f"earlier access: {w}", "include the wait's stage in the "
                         "barrier's srcStageMask (for an acquired swapchain image: "
                         "COLOR_ATTACHMENT_OUTPUT)"]
            else:
                code = "timing.cross-queue-race"
                msg = (f"'{u.res}' is {_verb(acc)} on '{m}' and written by a layout transition on "
                       f"'{n}' with nothing ordering the two.")
                notes = [f"other access: {w}", "add a semaphore (GPU queues) or wait on the "
                         "frame's fence (host) between them"]
        elif why == "same" and e.lt is not None:
            lt = e.lt
            code = "timing.barrier-scope"
            msg, fix = self._explain_lt_before(e, u)
            notes = [f"barrier: {lt.where}", f"earlier access: {w}", fix]
        elif u.flag in IN_USE_FLAGS or e.flag in IN_USE_FLAGS:
            # the host operation may be either access of the pair (whichever the
            # exploration reached later)
            code, verb, vuid = IN_USE_FLAGS[u.flag if u.flag in IN_USE_FLAGS else e.flag]
            if u.flag not in IN_USE_FLAGS:
                m, w = n, u.where
            obj = u.res.split(":", 1)[1] if ":" in u.res else u.res
            msg = (f"'{obj}' is {verb} on the host while work on '{m}' that uses it may still be "
                   "pending: nothing (a fence or semaphore wait) proves that work has completed.")
            notes = [f"pending use: {w}", f"wait for that work first ({vuid})"]
        elif why == "same":
            between = [b for bs, b in self.bars[n] if bs > e.seq and b.res in (u.res, None)]
            if between:
                b = between[-1]
                code = "timing.barrier-scope"
                msg, fix = self._explain_between(e, u, between)
                notes = [f"barrier: {b.where}", f"earlier access: {w}", fix]
                if b.event is not None:
                    notes.append(f"the barrier waits on event '{b.event}': only what precedes the event's "
                                 "latest vkCmdSetEvent* is in its first scope")
            else:
                code = "timing.missing-barrier"
                msg = (f"'{u.res}' is {_verb(acc)} and then {_verb(u.kind)} on queue '{n}' with no "
                       "barrier in between.")
                notes = [f"earlier access: {w}"]
        elif why == "wait" and getattr(self, "flt_rejected", None) is not None:
            flt = self.flt_rejected
            code = "timing.signal-stage-scope"
            msg = (f"'{u.res}' is {_verb(u.kind)} on '{n}' after being {_verb(acc)} on '{m}' "
                   f"({st_e}), but the semaphore signal that orders the two queues has stage mask "
                   f"{_masks(flt[3][1]) if not flt[3][0] else 'ALL_COMMANDS'}, whose first scope does not "
                   "hold that access (nor a barrier chained from it).")
            notes = [f"other access: {w}", f"signal: {flt[4]}",
                     "signal with ALL_COMMANDS, or with a stage mask that includes the access's stage "
                     "(or the dst stage of a barrier after it)"]
        elif why == "wait" and getattr(self, "relay_broken", False):
            code = "timing.signal-stage-scope"
            msg = (f"'{u.res}' is {_verb(u.kind)} on '{n}' after being {_verb(acc)} on '{m}' ({st_e}). "
                   f"The queues are connected only through a chain of semaphores, and a signal along that "
                   "chain has a stage mask whose first synchronization scope does not hold the stages its "
                   "own queue's earlier wait (or a barrier chained from it) unblocks, so the chain is "
                   "broken at that hop.")
            notes = [f"other access: {w}", "signal the intermediate semaphore with ALL_COMMANDS, or with a "
                     "stage mask that meets the stage mask of the wait it relays"]
        elif why == "wait":
            code = "timing.wait-stage-scope"
            msg = (f"'{u.res}' is {_verb(u.kind)} at {u.stage} on '{n}' after being {_verb(acc)} on "
                   f"'{m}', but the semaphore wait that orders the two queues does not block that stage.")
            notes = [f"other access: {w}", "add this stage to the wait's "
                     "stage mask, or chain a barrier from the wait's stage"]
        elif u.flag == "HOST_FREE":
            code = "timing.free-while-in-use"
            msg = (f"'{u.res[4:] if u.res.startswith('set:') else u.res}' is freed on the host while "
                   f"work on '{m}' that uses it may still be executing.")
            notes = [f"used by: {w}", "wait for the fence of every submission "
                     "that uses it first, or defer the free by frames in flight"]
        else:
            code = "timing.cross-queue-race"
            msg = (f"'{u.res}' is {_verb(acc)} on '{m}' and {_verb(u.kind)} on '{n}' with nothing "
                   "ordering the two.")
            notes = [f"other access: {w}", "add a semaphore (GPU queues) or wait on the "
                     "frame's fence (host) between them"]
        self.report.add("timing", code, msg, u.where, notes)
        if self.hazards is not None:
            self.hazards.append((e.key, u.key, code))

    # -- accesses -------------------------------------------------------------
    def _exempt(self, e, u):
        if e.cmd is not None and e.cmd == u.cmd:
            return True             # accesses of one command
        if e.queue == u.queue and u.scope is not None and e.scope == u.scope:
            return True             # attachment accesses of one subpass: rasterization order
        if e.queue == u.queue and u.instance is not None and e.instance is u.instance and \
                (e.pass_op is not None or u.pass_op is not None):
            return True             # a load / store operation and the commands of its subpass
        return atomic_exempt(e, u, self.slots)

    @staticmethod
    def _supersedes(u, e):
        """u (later) stands for e in every later check: same queue, stage,
        access flag, kind, scope and framebuffer position token; neither is a
        transition or in a call."""
        return (u.queue == e.queue and u.stage == e.stage and u.flag == e.flag and u.kind == e.kind
                and u.lt is None and e.lt is None and u.scope == e.scope and u.call is None
                and e.call is None and (u.cmd is None or u.cmd != e.cmd)
                and u.instance is e.instance and u.pass_op == e.pass_op and u.atomic == e.atomic
                and u.fb == e.fb)

    def _alive(self, rec):
        """Can a later access still be exempt from the dominator of a shadow entry?"""
        _, d, sg, cg, ig = rec
        return (d.scope is not None and self.scope_gen[d.queue] == sg) or \
            (d.cmd is not None and self.cmd_gen[d.queue] == cg) or \
            (d.instance is not None and self.inst_gen[d.queue] == ig)

    def _drop(self, shadow, e, u):
        """e is dominated by u: forget it, unless a later access may be exempt from u."""
        if u.scope is not None or u.cmd is not None or u.instance is not None:
            shadow.append((e, u, self.scope_gen[u.queue], self.cmd_gen[u.queue], self.inst_gen[u.queue]))

    def access(self, u):
        st = self.state.get(u.res)
        if st is None:
            st = self.state[u.res] = _ResState()
        segs = st.segments(u.region)
        done = {}
        for i in segs:
            writes, reads, shadow, kills = st.segs[i]
            cands = list(writes)
            if u.kind == "write":
                for lst in reads.values():
                    cands.extend(lst)
            if shadow:
                shadow[:] = [rec for rec in shadow if self._alive(rec)]
                cands.extend(rec[0] for rec in shadow if self._exempt(rec[1], u) and
                             (u.kind == "write" or rec[0].kind == "write"))
            if u.lt is None and u.op.guard is not None:
                # a host write recorded after the draw's submit (or any device write)
                # still live on the decoded bytes: the draw may read other bytes than decoded
                gs = self.gsets.get(u)
                if gs is None:
                    gs = self.gsets[u] = _GSet(u.op.guard)
                snap = gs.root().snap
                for e in writes:
                    hs = None if e.lt is not None else getattr(e.op, "hseq", None)
                    if hs is None or hs > snap:
                        gs.fail(self.guard_failed)
                        break
            k = 0
            while k < len(cands):
                e = cands[k]
                k += 1
                if id(e) in done:
                    continue
                if self.discard_kill and kills and id(e) in kills and any(
                        self._ordered(d, u, _overlap(d.region, u.region)) is None for d in kills[id(e)]):
                    continue            # R1': e was discarded by a transition ordered before u
                if self._exempt(e, u):
                    done[id(e)] = "exempt"
                    continue
                why = self._ordered(e, u, _overlap(e.region, u.region))
                if why is not None and e.lt is not None and u.lt is not None and e.queue == u.queue and \
                        not (e.lt.auto and u.lt.auto):
                    why = "lt-order"    # R-LT: transitions on one queue are ordered among themselves
                                        # (unless both are automatic render-pass transitions);
                                        # e stays live: nothing else is ordered through this, so
                                        # what e stood for (the entries e dropped) is checked too
                    if e.hid:
                        cands.extend(x for x in e.hid if id(x) not in done and
                                     _overlap(x.region, u.region) != ())
                elif why is not None:
                    self._report(e, u, why)
                elif u.queue == "host" and e.queue != "host" and e.kind == "write" and \
                        u.kind == "write" and u.flag == "HOST_WRITE" and not self._host_domain(e, u):
                    self._report_host_domain(e, u)
                    why = "premise"     # like a hazard: u does not stand in for e, so a later
                                        # host write is checked against e too
                elif self.discard_relaxed and e.kind == "write" and u.lt is not None and u.lt.old is None:
                    why = "relaxed"     # ordered by execution only (R1); u discards e's contents
                                        # (R1'), so e races only with what is not ordered after u,
                                        # which then races with u itself
                done[id(e)] = why
        for i in segs:
            writes, reads, shadow, kills = st.segs[i]
            if self.exhaustive:
                for e in writes:
                    if self.discard_kill and done.get(id(e)) == "relaxed":
                        kills.setdefault(id(e), []).append(u)
                if u.kind == "write":
                    writes.append(u)
                else:
                    reads.setdefault((u.queue, u.stage, id(u.scope), id(u.instance), u.pass_op, u.atomic, u.fb),
                                     []).append(u)
                continue
            # an atomic access never stands in for what it is ordered after: a later
            # access with a compatible atomic mode is exempt from it (atomic_exempt) but not
            # from those entries, which must still be checked against it. Dropping
            # them here would miss e.g. fill -> barrier -> compute atomic -> fragment atomic
            dom = u.atomic is None
            if u.kind == "write":
                keep = []
                for e in writes:
                    if (done.get(id(e)) is None and dom) or self._supersedes(u, e) or \
                            (self.discard_kill and done.get(id(e)) == "relaxed"):
                        self._drop(shadow, e, u)
                        if u.lt is not None:
                            u.hid = (u.hid or []) + [e]
                    else:
                        keep.append(e)
                keep.append(u)
                writes[:] = keep
                for k in list(reads):
                    keep = []
                    for e in reads[k]:
                        if done.get(id(e)) is None and dom:
                            self._drop(shadow, e, u)
                            if u.lt is not None:
                                u.hid = (u.hid or []) + [e]
                        else:
                            keep.append(e)
                    if keep:
                        reads[k] = keep
                    else:
                        del reads[k]
            else:
                # the framebuffer position token is part of the key: a read stands in for an
                # earlier one only if every chain from it orders what a chain from the earlier
                # read orders, and a framebuffer-local chain orders only accesses at its own
                # position (Lemma fast, stand-ins under framebuffer-local dependencies)
                k = (u.queue, u.stage, id(u.scope) if u.scope is not None else None,
                     id(u.instance) if u.instance is not None else None, u.pass_op, u.atomic, u.fb)
                lst = reads.get(k)
                if not lst or u.cmd is None:
                    reads[k] = [u]      # same queue, stage and scope: u stands for the earlier read
                    other = None
                else:
                    # keep the latest read of another command too: u's own
                    # command may write next, and is exempt from u
                    other = next((e for e in lst if e.cmd != u.cmd), None)
                    reads[k] = [u, other] if other is not None else [u]
                if lst and self.gsets:
                    # u stands for the reads it replaces, so a later hazard on u also
                    # fails the index guards those reads carried
                    for e in lst:
                        ge = self.gsets.get(e) if e is not other and e is not u else None
                        if ge is not None:
                            gu = self.gsets.get(u)
                            self.gsets[u] = ge if gu is None else gu.merge(ge)

    # -- main loop --------------------------------------------------------------
    def run(self):
        names, qs, vc, report = self.names, self.qs, self.vc, self.report
        pos = {n: 0 for n in names}
        signaled, sigx = {}, {}
        progressed = True
        while progressed:
            progressed = False
            for n in names:
                while pos[n] < len(qs[n]):
                    op = qs[n][pos[n]]
                    if isinstance(op, Wait):
                        if op.sem not in signaled:
                            break                       # blocked until signaled
                        sig = signaled[op.sem]
                        es = _ALL_NONE if (op.stages is None or n == "host") else _es(_fz(op.stages), False)
                        if es[0] or es[1] - {LT_STAGE}:
                            # (a wait whose stage mask is empty orders nothing)
                            for m, t in sig.items():
                                if t > vc[n][m]:
                                    vc[n][m] = t
                            sx = sigx[op.sem]
                            w = _Wait(vc[n][n] + 1, dict(sig), es, len(self.bars[n]), sigx=sx)
                            self.waits[n].append(w)
                            self._merge_relay(self.relay_any[n], sx)
                            if w.full:
                                self._merge_relay(self.relay_full[n], sx)
                            if w.full:
                                fm = self.fullmax[n]
                                for m, ents in sx.items():
                                    for t, flt in ents:
                                        if flt is None and t > fm.get(m, 0):
                                            fm[m] = t    # only what every hop's stage mask holds
                    pos[n] += 1
                    vc[n][n] += 1
                    seq = vc[n][n]
                    progressed = True
                    cmd = op.cmd if isinstance(op, Use) else None
                    if cmd is None or cmd != self.cur_cmd[n]:
                        self.cmd_gen[n] += 1
                        self.cur_cmd[n] = cmd
                    # a shadow entry (_drop) lives while a later access can still be in its
                    # dominator's scope / instance, i.e. until another one starts on the queue.
                    # a Wait or Signal does not end it. Real traces never resume a scope
                    # after one (the loader makes a fresh scope object per recorded render pass
                    # instance), but the reference semantics exempts same-scope pairs whatever
                    # lies between them; ending it there would lose the entry (criterion E, random
                    # frame seed 916 of test_long: transition -> depth write -> Signal -> depth
                    # read in the same scope)
                    sc = op.scope if isinstance(op, Use) else None
                    if sc is not None and sc != self.cur_scope[n]:
                        self.scope_gen[n] += 1
                        self.cur_scope[n] = sc
                    ins = op.instance if isinstance(op, Use) else None
                    if ins is not None and ins is not self.cur_inst[n]:
                        self.inst_gen[n] += 1
                        self.cur_inst[n] = ins
                    if isinstance(op, Signal):
                        signaled[op.sem] = dict(vc[n])
                        sst = getattr(op, "stages", None)
                        ses = _es(_fz(sst), True) if sst is not None else _ALL_NONE
                        sigx[op.sem] = self._sig_relay(n, seq, ses, op.where)
                    elif isinstance(op, Barrier):
                        self.bars[n].append((seq, op))
                        self.bar_seqs[n].append(seq)
                        if op.event_mark is not None:
                            if op.event_mark[0] == "set":
                                self.ev_set[n][op.event_mark[1]] = seq
                            else:
                                self.ev_set[n].pop(op.event_mark[1], None)
                        if op.event is not None:
                            self.ev_first[(n, seq)] = self.ev_set[n].get(op.event, 0)
                        cur = self.layouts.get(op.res, op.region) if op.res is not None else []
                        if op.res is not None and op.new is not None:
                            self.access(_Acc(n, seq, op, lt=True))
                        bad = [c for c in cur if not layouts_match(op.old, c)] if op.old is not None else []
                        if bad:
                            report.add("timing", "timing.stale-barrier",
                                       f"barrier moves '{op.res}' from '{op.old}' but it is "
                                       f"actually in '{bad[0]}'.", op.where)
                        if op.new is not None and op.res is not None:
                            self.layouts.set(op.res, op.region, op.new)
                    elif isinstance(op, Use):
                        cg = op.cond
                        if cg is not None and cg.static is None and cg not in self.failed_guards:
                            continue        # element the draw's decoded indices exclude
                        if op.guard is not None:
                            self.guards_seen.add(op.guard)
                        bad = [c for c in self.layouts.get(op.res, op.region)
                               if not layouts_match(op.layout, c)] if op.layout else []
                        if bad:
                            report.add("timing", "timing.wrong-layout",
                                       f"'{op.res}' is used as '{op.layout}' while it is in "
                                       f"layout '{bad[0]}'.", op.where,
                                       ["add a layout transition before this use"])
                        self.access(_Acc(n, seq, op))
        for n in names:
            if pos[n] < len(qs[n]):
                op = qs[n][pos[n]]
                report.add("timing", "timing.deadlock",
                           f"queue '{n}' waits on '{op.sem}', which is never signaled.",
                           op.where)


def _eligibility(report, unsup, c, cfg):
    """Does this check carry the guarantee (Corollary cor:sched)? Checking
    completeness (every queue ran to its end), coverage (no unmodelled command or
    state that hides memory accesses), the verified configuration, and no pair
    accepted under premise A-HOST alone. Reporting completeness (the grouping budget
    of report_groups) is separate: it never removes a raw diagnostic."""
    codes = {d.code for d in report.items}
    gaps = sorted(k for k, u in unsup.items() if u.get("effect") in ("memory", "untracked"))
    out = {"verified_configuration": cfg["verified_configuration"],
           "checking_complete": "timing.deadlock" not in codes,
           "model_gaps": gaps,
           "host_domain_premise_pairs": c.premise_host}
    out["guarantee_eligible"] = bool(out["verified_configuration"] and out["checking_complete"]
                                     and not gaps and not c.premise_host)
    return out


def check(frame, report=None, hazards=None, exhaustive=False, discard_relaxed=None, stats=None, window=True,
          discard_kill=False, premise=None, slots=None):
    """Check a frame and return the Report.

    window: also decide whether the trace is long enough (Theorem 3,
    covenant/timing_window.py); the result is report.meta["timing.window"]
    and stats["window"], never a diagnostic.

    hazards: optional list; receives (earlier (queue, seq), later (queue, seq),
    code) for every hazard (seq = 1-based position after loop expansion).
    exhaustive: keep every access instead of the epochs (pairwise checking).
    discard_relaxed / discard_kill: relaxations R1 / R1' (module docstring).
    discard_relaxed=None (the default) means spec-strict unless the opt-in heuristic
    COVENANT_HEURISTIC_DISCARD=1 is set (covenant/modes.py).
    slots: accesses through atomically allocated slots do not conflict (opt-in
    heuristic; None: COVENANT_HEURISTIC_SLOTS). The configuration is resolved once
    (modes.resolve) and the same values drive the check and report.meta.
    premise: optional list; receives (device write where, host write where) for every
    pair accepted under premise A-HOST only (warning timing.premise-host-domain).

    report.meta["covenant.eligibility"] says whether the result carries the guarantee
    (every race of the recorded execution is reported): the verified configuration,
    checking complete (no queue left blocked), no model gap that hides memory
    accesses, and no pair accepted under premise A-HOST only."""
    from . import modes as _modes
    cfg = _modes.resolve(discard_relaxed, slots)
    if frame.host_capture is not None and frame.host_capture != cfg["host_capture"]:
        # the capture mode the trace was loaded with is the one that holds
        cfg["host_capture"] = frame.host_capture
        cfg["verified_configuration"] = cfg["verified_configuration"] and frame.host_capture == "conservative"
    discard_relaxed = cfg["discard"] != "strict"
    slots = cfg["atomic_slots"] != "off"
    if discard_kill and discard_relaxed:
        cfg["discard"] = "heuristic-R1'"
    report = report if report is not None else Report()
    if hasattr(report, "meta"):
        report.meta["covenant.modes"] = dict(cfg)
    for cat, code, msg, where, notes in frame.findings:
        report.add(cat, code, msg, where, notes)
    for sem, img, pwhere, swhere in frame.semaphore_reuse:
        report.add("timing", "timing.present-semaphore-reuse",
                   f"'{sem}' is signaled again while the presentation engine may still be "
                   f"waiting on it for '{img}': nothing proves that present has consumed it "
                   "until the same image is acquired again.", swhere,
                   [f"the present that waits on it: {pwhere}",
                    "keep one render-finished semaphore per swapchain image "
                    "(VUID-vkQueueSubmit-pSignalSemaphores-00067)"])
    for ops in frame.queues.values():
        check_swapchains(frame, _expand(ops, Report(), dict(frame.initial_layouts)), report)
    unsup = getattr(frame, "unsupported", None) or {}
    if unsup and hasattr(report, "meta"):
        report.meta["model.unsupported"] = {k: dict(v) for k, v in unsup.items()}
    for what, u in sorted(unsup.items()):
        if u.get("effect") != "memory":
            continue
        if u.get("kind") == "command":
            # a soundness caveat for the frame, never a silent skip
            report.add("timing", "timing.unmodelled-command",
                       f"'{what}' is not modelled ({u.get('count', 1)} occurrence(s)): its memory accesses "
                       "and synchronization are unknown, so the timing verdict does not cover it.",
                       u.get("first", ""), [u["detail"]] if u.get("detail") else [], severity="warning")
        else:
            report.add("timing", "timing.unmodelled",
                       f"the model does not cover {what} ({u.get('count', 1)} occurrence(s)): "
                       "hazards involving it are not checked.", u.get("first", ""),
                       [u["detail"]] if u.get("detail") else [], severity="warning")
    # decoded descriptor indices are used only while their IndexGuard holds. A run
    # that finds a guard failing is repeated with that guard's elements charged (cond
    # Uses checked); widening only adds reads of images, never accesses of the guarded
    # vertex bytes, so a guard's outcome does not depend on which guards are widened
    # and this ends after at most one extra run in practice (bounded by the guards)
    failed = set()
    while True:
        rep = Report()
        # a hazard sink with a budget (report_groups) hands out fresh sinks of its own kind
        hz = (hazards.fresh() if hasattr(hazards, "fresh") else []) if hazards is not None else None
        c = _Checker(frame, rep, hz, exhaustive, discard_relaxed, discard_kill, slots)
        c.failed_guards = set(failed)
        c.premise_pairs = [] if premise is not None else None
        c.run()
        new = {g for g in c.guard_failed if g.static is None} - failed
        if not new:
            break
        failed |= new
    for d in rep.items:
        report.add(d.checker, d.code, d.message, d.where, d.notes, severity=d.severity)
    if hz:
        hazards.extend(hz)
    if premise is not None:
        premise.extend(c.premise_pairs)
    if hasattr(report, "meta"):
        report.meta["covenant.eligibility"] = _eligibility(report, unsup, c, cfg)
    if c.guards_seen and hasattr(report, "meta"):
        report.meta["timing.desc_refine.failed"] = frozenset(failed)    # guards widened in this run
        report.meta["timing.desc_refine"] = {
            "draw_guards": len(c.guards_seen),
            "widened_static": sum(1 for g in c.guards_seen if g.static is not None),
            "widened_dynamic": len(failed)}
    if stats is not None:
        stats["pairs_checked"] = c.pairs_checked
    if window:
        # Theorem 3: is the trace long enough to stand for every longer run?
        from .timing_window import window as _window
        info = _window(frame)
        if hasattr(report, "meta"):
            report.meta["timing.window"] = info
        if stats is not None:
            stats["window"] = info.summary()
    return report
