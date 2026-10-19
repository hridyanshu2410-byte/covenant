"""Executable reference semantics for COVENANT's timing model.

This is a deliberately slow, direct transcription of the Vulkan
Synchronization chapter (and the Memory Model appendix) over the op
vocabulary of ``covenant.timing`` (Use, Barrier, Signal, Wait on queues).
It is the *specification* the fast checker ``timing.check`` is tested
against (tests/test_timing_ref.py): Theorem 1 says every race this module
finds is reported by the checker (Haz_spec ⊆ Haz_C).

Nothing here is shared with timing.py: stage expansion, access scopes,
chaining and the race definition are re-derived from the spec text.

Events
------
* every ``Use`` on every queue is an access event (resource, region, kind,
  stage, access flag, render-pass scope, command id);
* every ``Barrier`` with ``res`` set and ``new`` set is a layout transition:
  a *write* event "LT" of that subresource located at the barrier;
* loops are unrolled twice (as in timing._expand); ops after a wait that is
  never satisfied never execute and are not events.

Dependencies (nodes of the chain graph)
---------------------------------------
* BAR: a Barrier op at (queue n, position p).
    first sync scope   = events on n before p whose stage is in ES_src(src mask)
    second sync scope  = events on n after p whose stage is in ES_dst(dst mask)
    first access scope = those with stage in AS(src mask), flag in src access,
                         on the barrier's resource/region (global if res None)
    second access scope likewise with the dst masks.
  Structs of one call (same ``call``, different ``struct``) sit at one
  position: they neither chain into each other nor capture each other's
  layout transitions.
* SIG: a Signal op. First scope = every earlier event on its queue (all
  stages, all accesses: a semaphore/fence signal with ALL_COMMANDS; this
  includes layout transitions, which are part of earlier commands). A
  Signal with ``stages`` (sync2 VkSemaphoreSubmitInfo::stageMask) has first
  scope ES_src(stages): an access is in it by its stage, a dependency chains
  into it when its second scope meets ES_src(stages), and a layout transition
  only through its own barrier (unless the mask holds every stage). Ignoring
  ``stages`` here would make relays through a third queue or the host too
  permissive.
* WAIT: a Wait op paired with the signals of its name. Second scope = events
  after it on its queue whose stage is in ES_dst(wait stages) (None = all);
  a wait whose ES_dst holds every stage (ALL_COMMANDS, TOP_OF_PIPE) also holds
  later layout transitions (ALL_COMMANDS "specifies all operations
  performed by all commands", synchronization.adoc:441-442);
  second access scope = all accesses in that sync scope. A queue submission
  (a wait whose signal is on the host) has every later command of the queue in
  its second scope, layout transitions included ("all commands submitted"). If a name is
  signaled more than once, the wait is reached only through *every* signal
  (meet; valid Vulkan never does this).
* host: program order. Every host event is in the first scope of every later
  host Signal and in the second scope of every earlier host Wait; host
  accesses are ordered among themselves by program order. A fence/host wait
  is a full memory dependency (assumption A_host, shared with syncval).

Stage expansion (execution scopes only)
  ES_src(M): each stage plus the logically earlier stages of its pipelines;
             TOP_OF_PIPE = {}, BOTTOM_OF_PIPE = ALL_COMMANDS = all.
  ES_dst(M): each stage plus the logically later ones;
             BOTTOM_OF_PIPE = {}, TOP_OF_PIPE = ALL_COMMANDS = all.
  AS(M):     only stages named (ALL_COMMANDS and ALL_GRAPHICS expand;
             TOP/BOTTOM carry no accesses). Depth/stencil attachment accesses
             (EARLY/LATE_FRAGMENT_TESTS) are in an access scope naming
             either test stage (modelling decision M1: the model records one
             test stage for an access the hardware may do in either).
  The pseudo-stage LAYOUT_TRANSITION (elide's split transitions) is in a
  scope only when named explicitly, and links only the two halves of one split
  transition (same call and struct): two dependencies never chain through it
  otherwise ("an application must not rely on ordering of image layout
  transitions to influence ordering of other commands").
  An event with unknown stage (None) is in a scope only if the scope holds
  every stage; an unknown access flag only if the mask has MEMORY_READ /
  MEMORY_WRITE (or is None).

Relations
  a =>x b  (execution): a chain D1..Dn of dependencies, a in D1's first
           sync scope, b in Dn's second, consecutive ones chaining
           (D_i's second scope meets D_{i+1}'s first scope).
  a =>m b  (memory): the same with a in D1's first *access* scope and b in
           Dn's second *access* scope (intermediate links need only chain).
  A layout transition LT of barrier B: a => LT iff a reaches B (as above with
  Dn = B); LT => b iff a chain starts at B (or at a later Signal on B's queue)
  and ends with b in Dn's second (access) scope.

Race (Haz_spec)
  Two events a, b conflict when they touch overlapping elements of the same
  resource, at least one writes, they are not accesses of one command, and
  they are not both attachment accesses of one render-pass scope on one
  queue (rasterization order), and they are not an attachment load/store
  operation and an access of a command in the same subpass instance (a load
  happens-before / a store happens-after every command of its subpass that
  accesses the attachment), and they are not two shader accesses on one
  queue whose binding modes are compatible (both device-scope atomics, or
  reads, on every block member both touch: the Vulkan memory model has no data
  race between atomics; spirv.modes_compatible; atomically allocated slots are
  ordinary accesses here, as in the verified configuration),
  and they are not two layout transitions on one queue at least one of
  which is performed via an image memory barrier (R-LT,
  synchronization.adoc:262-269: "Layout transitions that are performed via
  image memory barriers execute in their entirety in submission order,
  relative to other image layout transitions submitted to the same queue,
  including those performed by render passes"; two automatic render-pass
  transitions (Ev.auto) are not covered and need a dependency; this orders
  only the two transitions, not other commands, so it adds no edge to the
  dependency relations below).
  Framebuffer-local dependencies (synchronization.adoc:1773-1856): a barrier
  with ``fb_local`` (BY_REGION, or VIEW_LOCAL with viewOffset 0) orders a
  pair of framebuffer-space stages only between accesses with the same
  framebuffer position token ``fb``; a stage pair involving a non-framebuffer
  stage stays global. ``fb_local == "none"`` (VIEW_LOCAL with a non-zero
  viewOffset) orders no framebuffer-space pair at all (conservative). A chain
  stays local only while every link is local.
  The pair races unless ord(a, b) or ord(b, a),
  where ord(x, y) = x =>m y if x writes, x =>x y if x only reads (WAR).
  ``discard_relaxed`` (default False, the specification: every earlier write
  must be made available before any layout transition, synchronization.adoc
  "Image Layout Transitions"): a write before a transition from UNDEFINED
  (old=None) needs only =>x (contents are discarded); the checker's opt-in
  heuristic R1 (COVENANT_HEURISTIC_DISCARD=1, covenant/modes.py), which is not
  the specification and is outside the theorem. Its closure R1' (``discard_kill``, default False:
  it hides a real WAW, see covenant/timing.py): such a discard d kills the
  write w on the elements d covers, so (w, b) does not race when w =>x d and
  d is ordered before b (d =>m b; d is itself a write that b must follow).
  This is syncval's view (the transition replaces w as the last write) and the
  pattern of an UNDEFINED transition with srcAccessMask 0.

Regions: ``region`` on Use/Barrier is None (whole resource) or a tuple of
half-open integer intervals (bytes of a buffer; aspect-slot * layers + layer
of one mip level of an image). Races are decided per element: the overlap is
split into atoms at every barrier region boundary on that resource.
"""
from collections import defaultdict

from . import timing as _t     # only for the op dataclasses

GRAPHICS = ["TOP_OF_PIPE", "DRAW_INDIRECT", "VERTEX_INPUT", "VERTEX_SHADER",
            "TESSELLATION_CONTROL_SHADER", "TESSELLATION_EVALUATION_SHADER", "GEOMETRY_SHADER",
            "EARLY_FRAGMENT_TESTS", "FRAGMENT_SHADER", "LATE_FRAGMENT_TESTS",
            "COLOR_ATTACHMENT_OUTPUT", "BOTTOM_OF_PIPE"]
COMPUTE = ["TOP_OF_PIPE", "DRAW_INDIRECT", "COMPUTE_SHADER", "BOTTOM_OF_PIPE"]
TRANSFER = ["TOP_OF_PIPE", "TRANSFER", "BOTTOM_OF_PIPE"]
# vk.xml <syncpipeline>: "graphics mesh", "ray tracing", "acceleration structure build/copy"
MESH = ["TOP_OF_PIPE", "DRAW_INDIRECT", "TASK_SHADER", "MESH_SHADER", "EARLY_FRAGMENT_TESTS",
        "FRAGMENT_SHADER", "LATE_FRAGMENT_TESTS", "COLOR_ATTACHMENT_OUTPUT", "BOTTOM_OF_PIPE"]
RAY_TRACING = ["TOP_OF_PIPE", "DRAW_INDIRECT", "RAY_TRACING_SHADER", "BOTTOM_OF_PIPE"]
AS_BUILD = ["TOP_OF_PIPE", "ACCELERATION_STRUCTURE_BUILD", "BOTTOM_OF_PIPE"]
AS_COPY = ["TOP_OF_PIPE", "ACCELERATION_STRUCTURE_COPY", "BOTTOM_OF_PIPE"]
PIPELINES = [GRAPHICS, COMPUTE, TRANSFER, MESH, RAY_TRACING, AS_BUILD, AS_COPY]
# Synchronization chapter: ACCELERATION_STRUCTURE_BUILD "specifies the execution of
# acceleration structure commands or acceleration structure copy commands"; the
# model records copy-command accesses at ACCELERATION_STRUCTURE_COPY, so a mask
# naming BUILD holds that stage as well (in every scope)
INCLUDES = {"ACCELERATION_STRUCTURE_BUILD": ("ACCELERATION_STRUCTURE_COPY",),
            # sync2: ALL_TRANSFER (= TRANSFER) "is equivalent to specifying all of" COPY, BLIT,
            # RESOLVE and CLEAR; the model records transfer commands at those stages
            "TRANSFER": ("COPY", "BLIT", "CLEAR", "RESOLVE")}
LT = "LAYOUT_TRANSITION"
FT = {"EARLY_FRAGMENT_TESTS", "LATE_FRAGMENT_TESTS"}
# synchronization.adoc "Framebuffer Region Dependencies": the
# framebuffer-space stages. FBX adds the pseudo-stage of a layout transition, which the
# model places at the framebuffer position of the attachment it transitions
FB = frozenset({"FRAGMENT_SHADER", "EARLY_FRAGMENT_TESTS", "LATE_FRAGMENT_TESTS", "COLOR_ATTACHMENT_OUTPUT"})
FBX = FB | {LT}
GRAPHICS_ONLY = (set(GRAPHICS) | set(MESH)) - {"TOP_OF_PIPE", "BOTTOM_OF_PIPE"}
READ_GROUPS = {"SHADER_READ": {"SHADER_SAMPLED_READ", "SHADER_STORAGE_READ", "SHADER_BINDING_TABLE_READ",
                               "SHADER_READ"}}
WRITE_GROUPS = {"SHADER_WRITE": {"SHADER_STORAGE_WRITE", "SHADER_WRITE"}}


class Ev:
    __slots__ = ("key", "queue", "pos", "op", "res", "region", "kind", "stage", "flag", "scope",
                 "call", "struct", "lt_of", "cmd", "discard", "instance", "pass_op", "atomic", "fb", "auto")

    def __init__(self, queue, pos, op, lt_of=None):
        self.queue, self.pos, self.op, self.lt_of = queue, pos, op, lt_of
        self.key = (queue, pos + 1)          # the checker's (queue, seq)
        self.region = getattr(op, "region", None)
        self.cmd = getattr(op, "cmd", None)
        self.instance = None if lt_of is not None else getattr(op, "instance", None)
        self.pass_op = None if lt_of is not None else getattr(op, "pass_op", None)
        self.atomic = None if lt_of is not None else getattr(op, "atomic", None)
        self.call, self.struct = getattr(op, "call", None), getattr(op, "struct", None)
        self.res = op.res
        self.fb = getattr(op, "fb", None)          # a transition: its barrier's attachment token
        self.auto = bool(getattr(op, "auto", False)) if lt_of is not None else False
        if lt_of is not None:
            self.kind, self.stage, self.flag, self.scope = "write", None, None, None
            self.discard = op.old is None
        else:
            self.kind, self.stage, self.flag, self.scope = op.access, op.stage, op.flag, op.scope
            self.discard = False

    def __repr__(self):
        return f"Ev({self.queue}#{self.pos} {'LT' if self.lt_of else self.kind} {self.res})"


# ---------------------------------------------------------------- scopes
class Scopes:
    def __init__(self, universe):
        self.U = frozenset(universe) - {LT}

    def _pipelines(self, s):
        ps = [p for p in PIPELINES if s in p]
        return ps or [["TOP_OF_PIPE", s, "BOTTOM_OF_PIPE"]]

    def es(self, mask, src):
        """Execution scope stage set of a mask (None = all stages)."""
        if mask is None:
            return self.U
        out = set()
        for s in mask:
            if s == "ALL_COMMANDS":
                out |= self.U
            elif s == "ALL_GRAPHICS":
                out |= GRAPHICS_ONLY
            elif s == LT:
                out.add(LT)
            elif s == ("BOTTOM_OF_PIPE" if src else "TOP_OF_PIPE"):
                out |= self.U
            elif s == ("TOP_OF_PIPE" if src else "BOTTOM_OF_PIPE"):
                continue
            else:
                out.add(s)
                out.update(INCLUDES.get(s, ()))
                for p in self._pipelines(s):
                    i = p.index(s)
                    out.update(p[1:i] if src else p[i + 1:-1])
        return frozenset(out)

    def as_(self, mask):
        """Access scope stage set (no logical expansion)."""
        if mask is None:
            return self.U
        out = set()
        for s in mask:
            if s == "ALL_COMMANDS":
                out |= self.U
            elif s == "ALL_GRAPHICS":
                out |= GRAPHICS_ONLY
            elif s in ("TOP_OF_PIPE", "BOTTOM_OF_PIPE"):
                continue
            else:
                out.add(s)
                out.update(INCLUDES.get(s, ()))
        return frozenset(out)

    def stage_in(self, stage, stages):
        if stage is None:
            return self.U <= stages
        return stage in stages

    def stage_in_access(self, stage, stages):
        if stage is None:
            return self.U <= stages
        if stage in stages:
            return True
        return stage in FT and bool(FT & stages)         # modelling decision M1

    @staticmethod
    def flag_in(flag, kind, mask):
        if mask is None:
            return True
        if ("MEMORY_READ" in mask and kind == "read") or ("MEMORY_WRITE" in mask and kind == "write"):
            return True
        if flag is None:
            return False
        if flag in mask:
            return True
        groups = READ_GROUPS if kind == "read" else WRITE_GROUPS
        return any(flag in members for g, members in groups.items() if g in mask)


def _overlap(r1, r2):
    """Intersection of two regions (None = whole) -> region, or () if empty."""
    if r1 is None:
        return r2
    if r2 is None:
        return r1
    out = []
    for a, b in r1:
        for c, d in r2:
            lo, hi = max(a, c), min(b, d)
            if lo < hi:
                out.append((lo, hi))
    return tuple(sorted(out))


def _contains(region, x):
    return region is None or any(a <= x < b for a, b in region)


# ---------------------------------------------------------------- the model
class Reference:
    def __init__(self, frame, discard_relaxed=False, discard_kill=False):
        self.discard_relaxed = discard_relaxed
        self.discard_kill = discard_kill and discard_relaxed
        self.queues = {n: self._unroll(ops) for n, ops in frame.queues.items()}
        self._executed()
        stages = set()
        for ops in self.queues.values():
            for op in ops:
                if isinstance(op, _t.Use) and op.stage:
                    stages.add(op.stage)
                if isinstance(op, _t.Barrier):
                    for m in (op.src_stages, op.dst_stages):
                        stages |= set(m or ())
                if isinstance(op, (_t.Wait, _t.Signal)) and getattr(op, "stages", None):
                    stages |= set(op.stages)
        stages |= set().union(*PIPELINES) | {"HOST"}
        stages -= {"ALL_COMMANDS", "ALL_GRAPHICS"}
        self.S = Scopes(stages)
        self._events()
        self._deps()

    # -- frame ------------------------------------------------------------
    @staticmethod
    def _unroll(ops):
        out = []
        for op in ops:
            if isinstance(op, _t.Loop):
                body = Reference._unroll(op.body)
                out.extend(body)
                out.extend(body)
            else:
                out.append(op)
        return out

    def _executed(self):
        """Prefix of every queue that can execute (waits need a signal that executes)."""
        self.limit = {n: 0 for n in self.queues}
        sig_names = set()
        changed = True
        while changed:
            changed = False
            for n, ops in self.queues.items():
                i = self.limit[n]
                while i < len(ops):
                    op = ops[i]
                    if isinstance(op, _t.Wait) and op.sem not in sig_names:
                        break
                    if isinstance(op, _t.Signal):
                        sig_names.add(op.sem)
                    i += 1
                if i != self.limit[n]:
                    self.limit[n] = i
                    changed = True

    def _events(self):
        self.events = []
        for n, ops in self.queues.items():
            for p, op in enumerate(ops[:self.limit[n]]):
                if isinstance(op, _t.Use):
                    self.events.append(Ev(n, p, op))
                elif isinstance(op, _t.Barrier) and op.res is not None and op.new is not None:
                    self.events.append(Ev(n, p, op, lt_of=op))

    def _deps(self):
        """Dependency nodes and the chaining relation, closed transitively
        (bitsets over node indices)."""
        S = self.S
        nodes = []          # (kind, queue, pos, op, extra)
        self.bar_at = {}
        # first-scope position of each barrier: its own position, or for an event
        # wait the position of the event's latest set before it (-1: none / reset)
        self.first = {}
        sigs = defaultdict(list)
        for n, ops in self.queues.items():
            last_set = {}
            for p, op in enumerate(ops[:self.limit[n]]):
                if isinstance(op, _t.Barrier):
                    mark = getattr(op, "event_mark", None)
                    if mark is not None:
                        if mark[0] == "set":
                            last_set[mark[1]] = p
                        else:
                            last_set.pop(mark[1], None)
                    ev = getattr(op, "event", None)
                    self.first[(n, p)] = p if ev is None else last_set.get(ev, -1)
                    self.bar_at[(n, p)] = len(nodes)
                    nodes.append(("BAR", n, p, op, (S.es(op.src_stages, True), S.es(op.dst_stages, False),
                                                    S.as_(op.src_stages), S.as_(op.dst_stages))))
                elif isinstance(op, _t.Signal):
                    sigs[op.sem].append(len(nodes))
                    # a sync2 signal's stage mask limits its first synchronization
                    # scope (None = every stage); (stage set, holds every stage)
                    st = getattr(op, "stages", None)
                    ses = S.U if st is None else S.es(st, True)
                    nodes.append(("SIG", n, p, op, (ses, S.U <= ses)))
        for n, ops in self.queues.items():
            for p, op in enumerate(ops[:self.limit[n]]):
                if isinstance(op, _t.Wait):
                    es = S.U if (op.stages is None or n == "host") else S.es(op.stages, False)
                    ss = tuple(sigs.get(op.sem, ()))
                    # no special case for a semaphore the host signalled. A queue
                    # submission (the loader's submit@N wait) has no stage mask, so es is every
                    # stage already; a device wait on a host vkSignalSemaphore value keeps its
                    # pWaitDstStageMask (treating every host-signalled wait as a submission
                    # would be too permissive).
                    nodes.append(("WAIT", n, p, op, (es, ss, False)))
        self.nodes = nodes
        by_queue = defaultdict(list)
        for i, nd in enumerate(nodes):
            by_queue[nd[1]].append(i)
        for q in by_queue.values():
            q.sort(key=lambda i: nodes[i][2])
        self.by_queue = by_queue
        # chain states. A dependency node is reached in a class:
        #   G  the path so far orders the source before every operation of the node's
        #      second scope;
        #   N  (a framebuffer-local barrier reached by a framebuffer-located source or
        #      path) only its operations in non-framebuffer-space stages, globally;
        #   L  (the same) its framebuffer-space operations, but only at the source's
        #      own framebuffer position: an end in class L needs the same Use.fb token.
        # A pair of a first-scope and a second-scope operation of a framebuffer-local
        # dependency is framebuffer-local exactly when both are in framebuffer-space
        # stages ("any dependency between framebuffer-space pipeline stages is
        # framebuffer-local"); otherwise it is an ordinary dependency. A chain with at
        # least one ordinary link is ordinary; one made only of local links is local.
        # successors: D -> D' when D's second scope meets D''s first scope
        succ = defaultdict(set)
        waits_of_sig = defaultdict(list)
        for i, nd in enumerate(nodes):
            if nd[0] == "WAIT":
                for s in nd[4][1]:
                    waits_of_sig[s].append(i)
        for i, nd in enumerate(nodes):
            kind, n, p = nd[0], nd[1], nd[2]
            if kind == "SIG":
                succ[(i, "G")].update((w, "G") for w in waits_of_sig[i])   # all-signals rule in _closure
                continue
            for cls in self.classes(i):
                second = self.second(i, cls)
                st = (i, cls)
                for j in by_queue[n]:
                    nd2 = nodes[j]
                    if nd2[2] <= p:
                        continue
                    if nd2[0] == "BAR" and self.first[(n, nd2[2])] <= p:
                        continue                    # an event wait: only what precedes the set chains in
                    if nd2[0] == "BAR":
                        if n == "host":
                            continue
                        if kind == "BAR" and _same_call(nd[3], nd2[3]):
                            continue
                        common = second & nd2[4][0]
                        if LT in common and not _same_struct(nd[3], nd2[3]):
                            common = common - {LT}
                        if common:
                            if cls == "L" and _local(nd2[3]):
                                succ[st].update(self.entered_locally(j))
                            else:
                                succ[st].add((j, "G"))
                    elif nd2[0] == "SIG":
                        # chains into a signal only through its first scope (not every
                        # signal holds every earlier operation: its stage mask limits it)
                        if n == "host" or (second and nd2[4][1]) or (second & nd2[4][0]):
                            succ[st].add((j, "G"))
        self.succ = succ

    def classes(self, i):
        nd = self.nodes[i]
        if nd[0] == "BAR" and _local(nd[3]):
            return ("G", "N") if _nopos(nd[3]) else ("G", "N", "L")
        return ("G",)

    def entered_locally(self, j):
        """States of local barrier node j entered by a framebuffer-located source or
        a local path: its framebuffer-space pairs are local (L), the others global (N)."""
        return ((j, "N"),) if _nopos(self.nodes[j][3]) else ((j, "L"), (j, "N"))

    def second(self, i, cls):
        """Second stage set of node i reached in class cls."""
        nd = self.nodes[i]
        base = nd[4][1] if nd[0] == "BAR" else nd[4][0]
        if cls == "N":
            return base - FBX
        if cls == "L":
            return base & FBX
        return base

    # -- reachability ------------------------------------------------------
    def _closure(self, start):
        """States (node, class) reachable from the start set; a WAIT node of a name
        signaled k > 1 times needs all k signals reached (meet)."""
        seen = set()
        stack = list(start)
        multi = [i for i, nd in enumerate(self.nodes) if nd[0] == "WAIT" and len(nd[4][1]) > 1]
        while True:
            while stack:
                st = stack.pop()
                if st in seen:
                    continue
                nd = self.nodes[st[0]]
                if nd[0] == "WAIT" and len(nd[4][1]) > 1 and not all((s, "G") in seen for s in nd[4][1]):
                    continue
                seen.add(st)
                stack.extend(x for x in self.succ.get(st, ()) if x not in seen)
            stack = [(i, "G") for i in multi if (i, "G") not in seen and
                     all((s, "G") in seen for s in self.nodes[i][4][1])]
            if not stack:
                return seen

    def _start(self, e, memory, x):
        """States whose first (access) scope holds event e (for element x)."""
        S, out = self.S, []
        if e.lt_of is not None:
            j = self.bar_at[(e.queue, e.pos)]
            out.extend(self.entered_locally(j) if _local(e.lt_of) else ((j, "G"),))
        for j in self.by_queue[e.queue]:
            nd = self.nodes[j]
            if nd[2] <= e.pos:
                continue
            if nd[0] == "SIG":
                # e is in the first scope of a stage-masked signal only by its stage; a
                # layout transition only through its own barrier (unless the mask is full)
                if e.queue == "host" or nd[4][1] or (e.lt_of is None and S.stage_in(e.stage, nd[4][0])):
                    out.append((j, "G"))
            elif nd[0] == "BAR" and e.queue != "host" and e.lt_of is None:
                b = nd[3]
                if self.first[(e.queue, nd[2])] <= e.pos:
                    continue                    # an event wait: e is not before the set
                if _same_call(e, b):
                    continue
                if not S.stage_in(e.stage, nd[4][0]):
                    continue
                if memory:
                    if not (S.stage_in_access(e.stage, nd[4][2]) and S.flag_in(e.flag, e.kind, b.src_access)):
                        continue
                    if b.res is not None and (b.res != e.res or not _contains(getattr(b, "region", None), x)):
                        continue
                if _local(b) and _located(e.stage):
                    out.extend(self.entered_locally(j))
                else:
                    out.append((j, "G"))
        return out

    def _ends(self, e, memory, x, fb):
        """States whose second (access) scope holds event e, for a source with token fb."""
        S, out = self.S, set()
        for j in self.by_queue[e.queue]:
            nd = self.nodes[j]
            if nd[2] >= e.pos:
                continue
            if nd[0] == "WAIT":
                # a wait of every stage holds every operation of every later
                # command, layout transitions included (ALL_COMMANDS "specifies all operations
                # performed by all commands"), like a queue submission
                if e.queue == "host" or S.U <= nd[4][0] or \
                        (e.lt_of is None and S.stage_in(e.stage, nd[4][0])):
                    out.add((j, "G"))
            elif nd[0] == "BAR" and e.queue != "host":
                b = nd[3]
                if e.lt_of is not None:
                    continue                    # a transition is only in its own barrier's scope
                if memory:
                    if not (S.stage_in_access(e.stage, nd[4][3]) and S.flag_in(e.flag, e.kind, b.dst_access)):
                        continue
                    if b.res is not None and (b.res != e.res or not _contains(getattr(b, "region", None), x)):
                        continue
                for cls in self.classes(j):
                    if cls == "L" and (fb is None or e.fb != fb):
                        continue                # framebuffer-local: only at the same position
                    if S.stage_in(e.stage, self.second(j, cls)):
                        out.add((j, cls))
        if e.lt_of is not None:
            j = self.bar_at[(e.queue, e.pos)]
            out.add((j, "G"))
            if fb is not None and e.fb == fb and "L" in self.classes(j):
                out.add((j, "L"))
        return out

    def ordered(self, a, b, memory, x=None):
        """a =>x b (memory=False) or a =>m b (memory=True) for element x."""
        if a.queue == b.queue == "host":
            return a.pos < b.pos
        if a.queue == b.queue and b.pos <= a.pos:
            return False
        key = (a.key, a.lt_of is not None, memory, x if memory else None)
        cl = self._reach_cache.get(key)
        if cl is None:
            cl = self._reach_cache[key] = self._closure(self._start(a, memory, x))
        key = (b.key, b.lt_of is not None, memory, x if memory else None, a.fb)
        ends = self._ends_cache.get(key)
        if ends is None:
            ends = self._ends_cache[key] = self._ends(b, memory, x, a.fb)
        return not cl.isdisjoint(ends)

    _reach_cache = None

    # -- races ---------------------------------------------------------------
    def conflicting(self, a, b):
        if a.res != b.res or (a.kind == "read" and b.kind == "read"):
            return ()
        if a.cmd is not None and a.cmd == b.cmd:
            return ()
        if a.queue == b.queue and a.scope is not None and a.scope is b.scope:
            return ()
        if a.queue == b.queue and a.instance is not None and a.instance is b.instance and \
                (a.pass_op is not None or b.pass_op is not None):
            return ()           # load/store operation vs the commands of its subpass
        if _t.atomic_exempt(a, b):
            return ()           # atomics (Vulkan memory model); slots are not exempt here
        if a.queue == b.queue and a.lt_of is not None and b.lt_of is not None and not (a.auto and b.auto):
            # R-LT: "Layout transitions that are performed via image memory barriers execute in
            # their entirety in submission order, relative to other image layout transitions
            # submitted to the same queue, including those performed by render passes".
            # only a pair with at least one image-memory-barrier
            # transition; two automatic render-pass transitions are not exempt
            return ()
        if a.key == b.key:
            return ()
        ov = _overlap(a.region, b.region)
        return ov

    def _atoms(self, res, ov):
        """Representative elements of the overlap, split at barrier region bounds."""
        if ov is None:
            pts = sorted(self._bounds.get(res, ()))
            if not pts:
                return [None]
            reps = [pts[0] - 1] + pts      # below the first bound, and each [p, next)
            return reps
        out = []
        for lo, hi in ov:
            cuts = sorted({lo} | {p for p in self._bounds.get(res, ()) if lo < p < hi})
            out.extend(cuts)
        return out

    def _ord(self, x, y, el):
        if x.kind == "write":
            if self.discard_relaxed and y.lt_of is not None and y.discard:
                return self.ordered(x, y, False)
            return self.ordered(x, y, True, el)
        return self.ordered(x, y, False)

    def _killed(self, x, y, el):
        """R1': write x is discarded by a transition d from UNDEFINED that
        x is ordered before (=>x) and that is ordered before y."""
        if not self.discard_kill or x.kind != "write":
            return False
        for d in self._discards.get(x.res, ()):
            if d is x or d is y:
                continue
            if el is None and d.region is not None:
                continue
            if el is not None and not _contains(d.region, el):
                continue
            if self.ordered(x, d, False) and self._ord(d, y, el):
                return True
        return False

    def races(self):
        """[(a, b)] of racing event pairs (a before b in (queue, position) order
        when on one queue).

        IndexGuard: a Use with ``cond`` (a descriptor-array element outside a
        draw's decoded indices) is an event only when its guard fails: the loader set
        ``static``, the guard read (``guard``) races, or a host write recorded after the
        guard's snapshot (``hseq > snap``) is ordered before the guard read."""
        failed = set()
        while True:
            out = self._races(failed)
            new = set()
            for a, b in out:
                for x in (a, b):
                    g = getattr(x.op, "guard", None) if x.lt_of is None else None
                    if g is not None:
                        new.add(g)
            for g_ev in self.events:
                g = getattr(g_ev.op, "guard", None) if g_ev.lt_of is None else None
                if g is None or g in new:
                    continue
                for w in self.events:
                    hs = getattr(w.op, "hseq", None) if w.lt_of is None else None
                    if w.res == g_ev.res and w.kind == "write" and hs is not None and hs > g.snap \
                            and _overlap(w.region, g_ev.region) != () and self.ordered(w, g_ev, False):
                        new.add(g)
                        break
            new = {g for g in new if g.static is None} - failed
            if not new:
                return out
            failed |= new

    def _races(self, failed):
        self._reach_cache, self._ends_cache = {}, {}
        self._bounds = defaultdict(set)
        for nd in self.nodes:
            if nd[0] == "BAR" and nd[3].res is not None and getattr(nd[3], "region", None):
                for lo, hi in nd[3].region:
                    self._bounds[nd[3].res].update((lo, hi))
        by_res = defaultdict(list)
        self._discards = defaultdict(list)
        for e in self.events:
            cg = getattr(e.op, "cond", None) if e.lt_of is None else None
            if cg is not None and cg.static is None and cg not in failed:
                continue            # not dynamically used while the guard holds
            by_res[e.res].append(e)
            if e.lt_of is not None and e.discard:
                self._discards[e.res].append(e)
        out = []
        for res, evs in by_res.items():
            for i, a in enumerate(evs):
                for b in evs[i + 1:]:
                    ov = self.conflicting(a, b)
                    if ov == ():
                        continue
                    if a.queue == b.queue and b.pos < a.pos:
                        a2, b2 = b, a
                    else:
                        a2, b2 = a, b
                    for el in self._atoms(res, ov):
                        if not (self._ord(a2, b2, el) or self._ord(b2, a2, el) or
                                self._killed(a2, b2, el) or self._killed(b2, a2, el)):
                            out.append((a2, b2))
                            break
        return out


def _local(b):
    """Is barrier b framebuffer-local (dependencyFlags BY_REGION or VIEW_LOCAL)?"""
    return bool(getattr(b, "fb_local", False))


def _nopos(b):
    """VIEW_LOCAL with a non-zero view offset: no framebuffer-space pair is at one position."""
    return getattr(b, "fb_local", False) == "none"


def _located(stage):
    """May an access at this stage be a framebuffer-space operation (then a
    framebuffer-local dependency orders it only locally)? Unknown stages may."""
    return stage is None or stage in FBX


def _same_call(a, b):
    ca, cb = getattr(a, "call", None), getattr(b, "call", None)
    return ca is not None and ca == cb and getattr(a, "struct", None) != getattr(b, "struct", None)


def _same_struct(a, b):
    """One struct of one barrier call (the halves of a split transition)."""
    ca, cb = getattr(a, "call", None), getattr(b, "call", None)
    return ca is not None and ca == cb and getattr(a, "struct", None) == getattr(b, "struct", None)


def races(frame, discard_relaxed=False):
    """Haz_spec(frame): list of racing event pairs (Ev, Ev)."""
    return Reference(frame, discard_relaxed).races()


def race_keys(frame, discard_relaxed=False):
    """{frozenset({(queue, seq), (queue, seq)})} of racing pairs; seq is the
    checker's 1-based position after loop expansion."""
    return {frozenset((a.key, b.key)) for a, b in races(frame, discard_relaxed)}


# ---------------------------------------------------------------- queue family ownership
def ownership_ref(ops, universe):
    """Reference for host_trace.Ownership (resources.adoc, "Queue Family
    Ownership Transfer", per buffer range / image subresource), element by
    element. ops: ("write"|"read", res, region, family) / ("release", res,
    region, src, dst) / ("acquire", res, region, src, dst); region None = all
    of range(universe). Returns per op: for a read the set of elements whose
    last write was on another family (and not acquired since); for an acquire
    whether releases src->dst covered all of it; else None."""
    owner, pend, out = {}, {}, []

    def elems(region):
        if region is None:
            return set(range(universe))
        return {x for lo, hi in region for x in range(lo, hi)}

    for op in ops:
        kind, res = op[0], op[1]
        xs = elems(op[2])
        if kind == "write":
            for x in xs:
                owner[(res, x)] = op[3]
            out.append(None)
        elif kind == "read":
            bad = {x for x in xs if owner.get((res, x), op[3]) != op[3]}
            for x in bad:
                owner[(res, x)] = op[3]
            out.append(bad)
        elif kind == "release":
            for x in xs:
                pend[(res, x)] = (op[3], op[4])
            out.append(None)
        else:
            out.append(all(pend.get((res, x)) == (op[3], op[4]) for x in xs))
            for x in xs:
                pend.pop((res, x), None)
                owner[(res, x)] = op[4]
    return out
