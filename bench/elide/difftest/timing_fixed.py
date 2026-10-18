"""A triage variant of covenant.timing.check with two suspected soundness fixes.

This is NOT the checker: it is a copy of the same-queue part of
timing.check (plus elide's transition_reach obligation) used only to test
hypotheses about why the checker and syncval disagree. covenant/ is not
modified. Two independent switches:

FIX_ACCESS_SCOPE ("access scopes are explicit")
    Vulkan: "the first access scope only includes memory accesses performed by
    pipeline stages *explicitly* specified in the source stage mask" (same for
    the second). Logically earlier/later stages, and TOP_OF_PIPE /
    BOTTOM_OF_PIPE, extend only the *execution* scope. timing.py uses
    _stage_in (which includes logically earlier/later stages and TOP/BOTTOM)
    for availability and visibility too. Here availability/visibility need
    the stage explicitly in the mask (ALL_COMMANDS, and ALL_GRAPHICS for
    graphics stages, expand as the spec says).

FIX_SAME_CALL ("barrier structs of one command do not chain")
    elide.build lays the structs of one vkCmdPipelineBarrier* call out as
    consecutive timing.Barrier ops (B1 ; layout transition ; B2 per image
    transition), so timing.covers_chain lets one struct chain into another
    struct of the same call, and lets a struct directly capture another
    struct's layout transition. In Vulkan all structs of one command share
    one position: a struct's first scope holds only earlier commands, and a
    layout transition is ordered only by its own struct. Here a barrier may
    chain from / capture only (a) barriers of other calls or (b) its own
    struct (B1 -> B2).

Ops must be annotated by annotate(frame) (call id / struct id per op).
"""
import re

from covenant import timing
from covenant.diagnostics import Report
from covenant.timing import Barrier, Use, Wait, Signal, _stage_in, _access_in, _stages_meet, _expand

FIX_ACCESS_SCOPE = True
FIX_SAME_CALL = True
LT_STAGE = "LAYOUT_TRANSITION"
_CALL = re.compile(r"^(vkCmdPipelineBarrier2?(?:KHR)?) @seq (\d+)\b.*?submitted @seq (\d+) on ")
NON_GRAPHICS = {"COMPUTE_SHADER", "TRANSFER", "HOST", LT_STAGE}


def _explicit(stage, mask):
    if not FIX_ACCESS_SCOPE:
        return True     # access test then falls back to the execution test alone
    if mask is None or stage is None:
        return True
    if "ALL_COMMANDS" in mask:
        return True
    if "ALL_GRAPHICS" in mask and stage not in NON_GRAPHICS:
        return True
    return stage in mask


def annotate(frame):
    """Give every Barrier / layout-transition Use a _call and _struct id."""
    sid = 0
    for ops in frame.queues.values():
        cur = None
        for op in ops:
            m = _CALL.match(op.where) if isinstance(op, (Barrier, Use)) else None
            call = (int(m.group(2)), int(m.group(3))) if m else None
            if isinstance(op, Barrier):
                if op.where.endswith(" (after layout transition)") and cur is not None:
                    op._call, op._struct = call, cur
                    cur = None
                else:
                    sid += 1
                    op._call, op._struct = call, sid
                    cur = sid if op.dst_stages == frozenset({LT_STAGE}) else None
            elif isinstance(op, Use):
                if op.stage == LT_STAGE and call is not None:
                    op._call, op._struct = call, cur
                else:
                    op._call = op._struct = None
    return frame


def _may_chain(c, b):
    if not FIX_SAME_CALL:
        return True
    cc, bc = getattr(c, "_call", None), getattr(b, "_call", None)
    return cc is None or cc != bc or c._struct == b._struct


def _may_capture(b, first_op):
    if not FIX_SAME_CALL or first_op is None:
        return True
    fc = getattr(first_op, "_call", None)
    return fc is None or fc != getattr(b, "_call", None) or first_op._struct == b._struct


def covers_chain(barriers, first, second, first_op=None):
    s1, f1, k1 = first
    s2, f2, k2 = second
    captured = []
    for b in barriers:
        direct = _may_capture(b, first_op) and _stage_in(s1, b.src_stages, True) and \
            (k1 != "write" or (_explicit(s1, b.src_stages) and _access_in(f1, k1, b.src_access)))
        chained = any(_may_chain(c, b) and _stages_meet(c.dst_stages, b.src_stages) for c in captured)
        if direct or chained:
            captured.append(b)
    for b in captured:
        if not _stage_in(s2, b.dst_stages, False):
            continue
        if k1 == "write" and k2 == "read" and not (_explicit(s2, b.dst_stages) and _access_in(f2, k2, b.dst_access)):
            continue
        return True
    return False


def _verb(a):
    return "written" if a == "write" else "read"


def check(frame, report=None):
    """Same-queue hazards and transition scopes, as timing.check, with the fixes.
    (The workloads use one GPU queue plus host fence waits; cross-queue and
    swapchain rules are taken from timing.check unchanged by running it and
    keeping only its non-same-queue codes.)"""
    report = report or Report()
    base = timing.check(frame, Report())
    for d in base.items:
        if d.code not in ("timing.barrier-scope", "timing.missing-barrier"):
            report.add(d.checker, d.code, d.message, d.where, d.notes)
    for n, ops0 in frame.queues.items():
        ops = _expand(ops0, Report(), dict(frame.initial_layouts))
        history = {}
        barriers = []
        seq = 0
        for op in ops:
            seq += 1
            if isinstance(op, Wait):
                # a new submission / semaphore wait: everything earlier on this
                # queue is ordered (host fence wait, as timing.py treats it)
                history = {}
                barriers = []
                continue
            if isinstance(op, Barrier):
                barriers.append((seq, op))
                if op.res is not None and op.new is not None:
                    _check_transition(op, seq, history, barriers, report)
                continue
            if not isinstance(op, Use):
                continue
            for (s, acc, w, sig, first_op) in history.get(op.res, []):
                if acc == "read" and op.access == "read":
                    continue
                if op.scope is not None and sig[3] == op.scope:
                    continue
                between = [b for bs, b in barriers if bs > s and b.res in (op.res, None)]
                if covers_chain(between, sig[:3], (op.stage, op.flag, op.access), first_op):
                    continue
                if between:
                    report.add("timing", "timing.barrier-scope",
                               f"the barrier on '{op.res}' does not cover this access: the earlier {_verb(acc)} "
                               f"({sig[0] or 'any stage'}, {sig[1] or 'any access'}) and this "
                               f"{_verb(op.access) if op.access == 'write' else 'read'} "
                               f"({op.stage or 'any stage'}, {op.flag or 'any access'}) are not both in its "
                               "stage/access masks.", op.where, [f"barrier: {between[-1].where}", f"earlier access: {w}"])
                else:
                    report.add("timing", "timing.missing-barrier",
                               f"'{op.res}' is {_verb(acc)} and then {_verb(op.access)} on queue '{n}' with no "
                               "barrier in between.", op.where, [f"earlier access: {w}"])
            history.setdefault(op.res, []).append((seq, op.access, op.where, (op.stage, op.flag, op.access, op.scope), op))
    return report


def _check_transition(op, seq, history, barriers, report):
    for (s, acc, w, sig, first_op) in history.get(op.res, []):
        between = [b for bs, b in barriers if s < bs <= seq and b.res in (op.res, None)]
        k1 = "read" if (acc == "read" or op.old is None) else "write"
        captured = []
        for b in between:
            direct = _may_capture(b, first_op) and _stage_in(sig[0], b.src_stages, True) and \
                (k1 != "write" or (_explicit(sig[0], b.src_stages) and _access_in(sig[1], k1, b.src_access)))
            if direct or any(_may_chain(c, b) and _stages_meet(c.dst_stages, b.src_stages) for c in captured):
                captured.append(b)
        if op in captured:
            continue
        report.add("timing", "timing.barrier-scope",
                   f"the layout transition of '{op.res}' (to '{op.new}') is not ordered after the earlier "
                   f"{_verb(acc)} ({sig[0] or 'any stage'}, {sig[1] or 'any access'}).", op.where,
                   [f"earlier access: {w}"])
        return


def transition_reach(f2, rep):
    """elide.Elider.transition_reach with the same two fixes."""
    for qn, ops in f2.queues.items():
        for k, b2 in enumerate(ops):
            if not (isinstance(b2, Barrier) and b2.src_stages == frozenset({LT_STAGE})):
                continue
            res, captured = b2.res, [b2]
            for op in ops[k + 1:]:
                if isinstance(op, Wait):
                    break
                if isinstance(op, Barrier):
                    if op.res not in (None, res):
                        continue
                    reached = any(_may_chain(c, op) and _stages_meet(c.dst_stages, op.src_stages) for c in captured)
                    if op.res == res and op.dst_stages == frozenset({LT_STAGE}):
                        if not reached:
                            rep.add("timing", "elide.transition-unreached",
                                    f"the layout transition of '{res}' is not ordered before the next transition "
                                    "of it by a chain from its own barrier.", op.where, [f"transition: {b2.where}"])
                        break
                    if reached:
                        captured.append(op)
                    continue
                if isinstance(op, Use) and op.res == res and op.stage != LT_STAGE:
                    ok = any(_stage_in(op.stage, c.dst_stages, False) and _explicit(op.stage, c.dst_stages) and
                             _access_in(op.flag, op.access, c.dst_access) for c in captured)
                    if not ok:
                        rep.add("timing", "elide.transition-unreached",
                                f"the layout transition of '{res}' does not reach this {op.access} "
                                f"({op.stage or 'any stage'}, {op.flag or 'any access'}) through its own dst scope "
                                "or a barrier chained from it.", op.where, [f"transition: {b2.where}"])


def hazards(el, plan):
    """{(code, where, message)} of the fixed oracle on el.build(plan)."""
    f2, _ = el.build(plan)
    annotate(f2)
    rep = check(f2, Report())
    transition_reach(f2, rep)
    return {(d.code, d.where, d.message): d for d in rep.items}
