"""Seeded random frames for the timing reference-semantics property tests
(tests/test_timing_ref.py) and helpers that read a checker's hazard pairs.

Every op gets a unique ``where`` ("<queue>#<index>"), so a checker's
diagnostics name both events of a hazard: the op at ``where`` and the op in
the note "earlier access: ..." / "other access: ..." / "used by: ...".
"""
import random
import re

from covenant.diagnostics import Report
from covenant.timing import Barrier, Frame, Signal, Use, Wait

STAGES = ["TOP_OF_PIPE", "DRAW_INDIRECT", "VERTEX_INPUT", "VERTEX_SHADER", "GEOMETRY_SHADER",
          "EARLY_FRAGMENT_TESTS", "FRAGMENT_SHADER", "LATE_FRAGMENT_TESTS", "COLOR_ATTACHMENT_OUTPUT",
          "COMPUTE_SHADER", "TRANSFER", "BOTTOM_OF_PIPE", "ALL_COMMANDS", "ALL_GRAPHICS"]
ACCESS_STAGES = ["VERTEX_INPUT", "VERTEX_SHADER", "FRAGMENT_SHADER", "EARLY_FRAGMENT_TESTS",
                 "LATE_FRAGMENT_TESTS", "COLOR_ATTACHMENT_OUTPUT", "COMPUTE_SHADER", "TRANSFER",
                 "DRAW_INDIRECT"]
FLAGS = {
    "VERTEX_INPUT": (["VERTEX_ATTRIBUTE_READ", "INDEX_READ"], []),
    "DRAW_INDIRECT": (["INDIRECT_COMMAND_READ"], []),
    "VERTEX_SHADER": (["UNIFORM_READ", "SHADER_SAMPLED_READ", "SHADER_STORAGE_READ"], ["SHADER_STORAGE_WRITE"]),
    "FRAGMENT_SHADER": (["UNIFORM_READ", "SHADER_SAMPLED_READ", "SHADER_STORAGE_READ", "INPUT_ATTACHMENT_READ"],
                        ["SHADER_STORAGE_WRITE"]),
    "COMPUTE_SHADER": (["UNIFORM_READ", "SHADER_SAMPLED_READ", "SHADER_STORAGE_READ"], ["SHADER_STORAGE_WRITE"]),
    "EARLY_FRAGMENT_TESTS": (["DEPTH_STENCIL_ATTACHMENT_READ"], ["DEPTH_STENCIL_ATTACHMENT_WRITE"]),
    "LATE_FRAGMENT_TESTS": (["DEPTH_STENCIL_ATTACHMENT_READ"], ["DEPTH_STENCIL_ATTACHMENT_WRITE"]),
    "COLOR_ATTACHMENT_OUTPUT": (["COLOR_ATTACHMENT_READ"], ["COLOR_ATTACHMENT_WRITE"]),
    "TRANSFER": (["TRANSFER_READ"], ["TRANSFER_WRITE"]),
}
ACCESS_MASK_BITS = ["UNIFORM_READ", "SHADER_READ", "SHADER_WRITE", "SHADER_SAMPLED_READ", "SHADER_STORAGE_READ",
                    "SHADER_STORAGE_WRITE", "COLOR_ATTACHMENT_READ", "COLOR_ATTACHMENT_WRITE",
                    "DEPTH_STENCIL_ATTACHMENT_READ", "DEPTH_STENCIL_ATTACHMENT_WRITE", "TRANSFER_READ",
                    "TRANSFER_WRITE", "MEMORY_READ", "MEMORY_WRITE", "INPUT_ATTACHMENT_READ",
                    "VERTEX_ATTRIBUTE_READ", "INDIRECT_COMMAND_READ", "INDEX_READ"]
FB = {"FRAGMENT_SHADER", "EARLY_FRAGMENT_TESTS", "LATE_FRAGMENT_TESTS", "COLOR_ATTACHMENT_OUTPUT"}
LAYOUTS = ["general", "color_attachment", "shader_read", "transfer_dst", "transfer_src"]


def _subset(rng, items, p):
    return frozenset(x for x in items if rng.random() < p)


def _region(rng, opts):
    if not opts.get("regions") or rng.random() < 0.5:
        return None
    lo = rng.randrange(0, 12)
    return ((lo, lo + rng.randrange(1, 5)),)


def random_frame(seed, n_ops=40, queues=("host", "q0", "q1"), **opts):
    """A random frame over a few resources: accesses, barriers (some in
    multi-struct calls, some layout transitions), semaphores between device
    queues, host submits/fence waits and host writes."""
    rng = random.Random(seed)
    devs = [q for q in queues if q != "host"]
    res = ["r0", "r1", "r2"][:opts.get("n_res", 3)]
    Q = {q: [] for q in queues}
    nsig = [0]
    scope = {q: None for q in devs}
    call_id = [0]

    def where(q):
        return f"{q}#{len(Q[q])}"

    pending_sig = []
    for _ in range(n_ops):
        q = rng.choice(devs) if rng.random() < 0.85 or "host" not in Q else "host"
        r = rng.random()
        if q == "host":
            if r < 0.4:
                Q[q].append(Use(rng.choice(res), "write", None, where(q), "HOST", "HOST_WRITE",
                                region=_region(rng, opts)))
            elif r < 0.7 and devs:
                d = rng.choice(devs)
                if opts.get("host_sig_waits") and rng.random() < 0.4:
                    # a host vkSignalSemaphore value waited by a device queue at a
                    # restricted pWaitDstStageMask (opt-in, so older seeds keep their frames)
                    name = f"t{len(Q[q])}=1"
                    Q[q].append(Signal(name, where(q)))
                    Q[d].append(Wait(name, where(d), _subset(rng, STAGES, 0.2) or None))
                    continue
                name = f"submit@{len(Q[q])}"
                Q[q].append(Signal(name, where(q)))
                Q[d].append(Wait(name, where(d)))
            elif pending_sig:
                name = pending_sig.pop(rng.randrange(len(pending_sig)))
                Q[q].append(Wait(name, where(q)))
            continue
        if r < 0.5:
            st = rng.choice(ACCESS_STAGES + [None] * opts.get("none_stage", 0))
            reads, writes = FLAGS.get(st, (["MEMORY_READ"], ["MEMORY_WRITE"]))
            acc = "write" if (writes and rng.random() < 0.45) else "read"
            flag = rng.choice(writes if acc == "write" else reads)
            if opts.get("none_flag") and rng.random() < 0.1:
                flag = None
            sc = None
            if st in ("COLOR_ATTACHMENT_OUTPUT", "EARLY_FRAGMENT_TESTS", "LATE_FRAGMENT_TESTS") and rng.random() < 0.5:
                if scope[q] is None or rng.random() < 0.3:
                    scope[q] = object()
                sc = scope[q]
            Q[q].append(Use(rng.choice(res), acc, None, where(q), st, flag, sc, region=_region(rng, opts)))
            if opts.get("fb") and st in FB and rng.random() < 0.6:
                # an access at its own framebuffer position (one of two positions)
                Q[q][-1].fb = rng.choice(["p0", "p1"])
            if opts.get("passes") and scope[q] is not None and rng.random() < 0.7:
                # recorded inside the current subpass; some are its load/store operations
                Q[q][-1].instance = scope[q]
                if rng.random() < 0.25:
                    Q[q][-1].pass_op = rng.choice(["load", "store"])
            if opts.get("atomics") and st in ("VERTEX_SHADER", "FRAGMENT_SHADER", "COMPUTE_SHADER") and \
                    rng.random() < 0.6:
                # storage accesses that are atomics / slots / reads on block members 0..1
                Q[q][-1].atomic = frozenset((rng.choice(["atomic", "atomic", "slot", "read"]),
                                             rng.choice([0, 1, None])) for _ in range(rng.randrange(1, 3)))
            if opts.get("cmd") and rng.random() < 0.2:
                # a second access of the same command
                u = Q[q][-1]
                u.cmd = u.where
                acc2 = "write" if (writes and rng.random() < 0.45) else "read"
                Q[q].append(Use(rng.choice(res), acc2, None, where(q), st,
                                rng.choice(writes if acc2 == "write" else reads), sc, cmd=u.cmd,
                                region=_region(rng, opts)))
        elif r < 0.85:
            call_id[0] += 1
            nstruct = rng.choice([1, 1, 1, 2, 3])
            for k in range(nstruct):
                rres = rng.choice(res + [None])
                old = new = None
                if rres is not None and opts.get("transitions", True) and rng.random() < 0.25:
                    new = rng.choice(LAYOUTS)
                    old = rng.choice([None] + LAYOUTS)
                Q[q].append(Barrier(rres, old, new, where(q),
                                    _subset(rng, STAGES, 0.18), _subset(rng, ACCESS_MASK_BITS, 0.15),
                                    _subset(rng, STAGES, 0.18), _subset(rng, ACCESS_MASK_BITS, 0.15),
                                    call=f"c{call_id[0]}" if nstruct > 1 or rng.random() < 0.5 else None,
                                    struct=k, region=_region(rng, opts) if rres else None))
                if opts.get("fb") and rng.random() < 0.5:
                    # BY_REGION / VIEW_LOCAL; a transition at one position or none
                    Q[q][-1].fb_local = True if rng.random() < 0.85 else "none"
                    if new is not None:
                        Q[q][-1].fb = rng.choice([None, "p0", "p1"])
                if opts.get("auto_lt") and new is not None:
                    # an automatic render-pass transition (R-LT needs one barrier transition)
                    Q[q][-1].auto = rng.random() < 0.6
            if nstruct == 1 and Q[q][-1].call is None:
                pass
        else:
            nsig[0] += 1
            name = f"s{nsig[0]}"
            # sync2 signal stage masks (opt-in, so older seeds keep their frames)
            sst = None
            if opts.get("sig_masks") and rng.random() < 0.6:
                sst = _subset(rng, STAGES, 0.25)
            Q[q].append(Signal(name, where(q), sst))
            others = [d for d in devs if d != q]
            if others and rng.random() < 0.8:
                d = rng.choice(others)
                stages = None if rng.random() < 0.3 else _subset(rng, STAGES, 0.2)
                if opts.get("no_empty_wait") and stages is not None and not stages:
                    stages = None
                Q[d].append(Wait(name, where(d), stages))
            elif "host" in Q:
                pending_sig.append(name)
    return Frame(queues=Q)


_NOTE = re.compile(r"^(?:earlier access|other access|used by): (\S+)")


class AllReport(Report):
    """A Report that keeps every diagnostic (no de-duplication)."""

    def add(self, *args, **kw):
        from covenant.diagnostics import Diagnostic
        self.items.append(Diagnostic(*args, **kw))


def where_keys(frame):
    """where -> (queue, seq) for frames whose ops have unique wheres."""
    out = {}
    for q, ops in frame.queues.items():
        for i, op in enumerate(ops):
            if hasattr(op, "where"):
                out[op.where] = (q, i + 1)
    return out


def checker_pairs_from_report(frame, report):
    """{frozenset({key_a, key_b}): code} from a checker's diagnostics."""
    wk = where_keys(frame)
    out = {}
    for d in report.items:
        if d.code in ("timing.deadlock", "timing.stale-barrier", "timing.wrong-layout"):
            continue
        b = wk.get(d.where)
        a = None
        for n in d.notes:
            m = _NOTE.match(n)
            if m:
                a = wk.get(m.group(1))
        if a and b:
            out[frozenset((a, b))] = d.code
    return out
