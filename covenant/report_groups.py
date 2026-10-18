"""Repair-based grouping of timing diagnostics.

A real engine produces thousands of timing diagnostics that are a handful of
mistakes repeated per draw, per frame and per frame in flight. This module
groups them in three steps:

1. Site keys. Every hazard (a pair of accesses the checker reported) gets a
   canonical key: the code, the resource (swapchain image index dropped), and
   the two program sites (the ``where`` of each access with every trace
   sequence number, submit number and mapped-memory offset removed). Other
   diagnostics (wrong layout, ownership, ...) are keyed by code, site and
   message. Equal keys form one *site group*.

2. Repairs. For each site group with a same-queue hazard the module
   synthesizes a minimal barrier/mask repair of its representative hazard
   (e, u) and verifies it with the checker, which is elide's oracle used in
   the other direction (covenant/elide.py narrows masks as long as no
   diagnostic appears; here masks are widened until the group's diagnostics
   disappear and still none appears):
     * the barrier the diagnostic names (the last one between e and u, or the
       layout transition's own barrier) gets e's stage (and access, when e
       writes) added to its first scope and u's stage and access to its second;
     * if there is none, a barrier is inserted just before u's command, outside
       any render pass instance: srcStage = e's stage, srcAccess = e's access
       if e writes, dstStage / dstAccess = u's.
   A candidate is accepted when every hazard of the representative pair is gone
   and the checker reports no site key that was not there before. Cross-queue
   and host hazards need a semaphore or a fence wait; they are grouped but no
   repair is synthesized.

3. Merging. A verified repair often fixes several site groups (one barrier
   for all draws of a pass, one dst mask for every later reader). The groups
   that disappear under a repair are merged into it (greedy set cover, largest
   first). The result is the list of *repair groups*: what to change, and which
   diagnostics that removes.

Severity. A group is a technicality when every hazard in it only concerns
contents that are discarded anyway (the later access is a transition from
UNDEFINED, or a load / store operation DONT_CARE), and for queue family
ownership findings on a device with a single queue family (``--single-family``,
by default when the trace uses only one family). Everything else keeps the
diagnostics' severity.

Budget. Grouping runs the checker as an oracle up to
2 + max_repairs times, the base and verification runs in pairwise mode, whose hazard
list grows with the number of racing event pairs (quadratic in the worst case: a
long GSK path-rendering trace exhausted 15 GB). Every oracle run's hazard pairs go to a sink with a
cap (``--max-pairs``) and a resident-memory budget (``--budget-mb``); a run that
exceeds either is abandoned and the result is marked ``status: "incomplete"`` with
the reason, never presented as complete:

* base epoch run over budget: rerun without hazard pairs; the diagnostics are grouped
  by (code, site, message) only, no repairs;
* base pairwise run over budget: no repair is synthesized (none could be verified);
* a verification run over budget: the repair search stops there; the site groups not
  yet repaired stay ungrouped, each with a note.

Every diagnostic of the checker's report is a member of exactly one group in every case
(nothing is dropped from the raw list). The CLI writes an ``incomplete`` stub to the
JSON output before it starts, so a run killed from outside leaves an incomplete result,
not a missing (or clean) one.

CLI:  python3 -m covenant.report_groups trace.jsonl [--spirv-dir DIR] [--json OUT]
          [--no-repair] [--single-family] [--max-repairs N] [--budget-mb MB] [--max-pairs N]
"""
import argparse
import dataclasses
import json
import re
import sys
import time
from collections import OrderedDict, defaultdict

from . import timing
from .diagnostics import Report
from .timing import Barrier, Frame, Use

_SEQ = re.compile(r"(submitted )?@seq \d+|\+0x[0-9a-f]+|before call seq \d+|seq \d+")
_SWAP = re.compile(r"\[\d+\]")
PAIR_CODES = {"timing.barrier-scope", "timing.missing-barrier", "timing.wait-stage-scope",
              "timing.cross-queue-race", "timing.free-while-in-use", "timing.acquire-first-use",
              "timing.signal-stage-scope"}
SAME_QUEUE = {"timing.barrier-scope", "timing.missing-barrier"}
OWNERSHIP = {"timing.ownership-missing", "timing.ownership-unmatched-acquire"}
# verify repairs in pairwise mode by event identity (COVENANT_REPAIR_VERIFY=sites
# selects the weaker rule: epoch mode, no new diagnostic site key)
import os as _os
PAIRWISE_VERIFY = _os.environ.get("COVENANT_REPAIR_VERIFY", "pairs") != "sites"


class BudgetExceeded(Exception):
    """An oracle run went over the grouping budget (hazard pairs or resident memory)."""


def rss_mb():
    """Resident set size of this process in MB (0 if unknown)."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * _os.sysconf("SC_PAGE_SIZE") // (1 << 20)
    except (OSError, ValueError, IndexError):
        return 0


class HazardSink(list):
    """timing.check's hazard list with a budget: raises BudgetExceeded past max_pairs
    pairs or when the process's resident memory passes budget_mb (checked every 4096
    pairs)."""

    def __init__(self, max_pairs, budget_mb):
        super().__init__()
        self.max_pairs, self.budget_mb = max_pairs, budget_mb

    def fresh(self):
        return HazardSink(self.max_pairs, self.budget_mb)

    def _guard(self, n):
        if self.max_pairs and n > self.max_pairs:
            raise BudgetExceeded(f"more than {self.max_pairs} hazard pairs in one oracle run")
        if self.budget_mb and n // 4096 != (n - 1) // 4096 and rss_mb() > self.budget_mb:
            raise BudgetExceeded(f"resident memory over {self.budget_mb} MB in an oracle run")

    def append(self, x):
        self._guard(len(self) + 1)
        list.append(self, x)

    def extend(self, xs):
        xs = list(xs)
        self._guard(len(self) + len(xs))
        list.extend(self, xs)


def site(where):
    """A program site: `where` without trace positions."""
    return re.sub(r"\s+", " ", _SWAP.sub("[*]", _SEQ.sub("", where or ""))).strip()


_HELD = re.compile(r"would hold \d+ images")


def message_key(d):
    """The message part of a non-pair diagnostic's site key. An over-acquire
    message embeds how many images would be held, which grows by one per
    acquire at the same site: that count is not part of the site."""
    m = site(d.message)
    if d.code == "timing.swapchain-overacquire":
        m = _HELD.sub("would hold N images", m)
    return m


_CALL_SUFFIX = re.compile(r"\s*\((render pass[^)]*|subpass[^)]*|global memory barrier|execution dependency)\)\s*$")
_CB = re.compile(r"command_buffer#\d+")


def call_site(where):
    """The barrier call a Barrier op comes from (render pass dependencies and
    transitions of one vkCmd*RenderPass, structs of one vkCmdPipelineBarrier)."""
    return _CALL_SUFFIX.sub("", site(where))


def _short(where, n=110):
    return where if len(where) <= n else where[:n - 3] + "..."


@dataclasses.dataclass
class Hazard:
    code: str
    e: tuple                 # (queue, seq) of the earlier access
    u: tuple                 # (queue, seq) of the later access
    diag: object = None


@dataclasses.dataclass
class SiteGroup:
    key: tuple
    code: str
    resource: str
    first: str               # site of the earlier access ("" for single-site diagnostics)
    second: str              # site of the later access / the diagnostic
    hazards: list
    diagnostics: list
    severity: str = "error"
    why_technical: str = ""
    repair: object = None
    repair_note: str = ""


@dataclasses.dataclass
class Repair:
    kind: str                # "widen" | "insert"
    queue: str
    at: str                  # the barrier widened / the access the barrier goes before
    src_stages: frozenset
    src_access: frozenset
    dst_stages: frozenset
    dst_access: frozenset
    res: str = None
    fixes: list = dataclasses.field(default_factory=list)     # site group keys

    def describe(self):
        f = lambda m: "|".join(sorted(m)) or "NONE"
        if self.kind == "widen":
            return (f"widen the barrier call {_short(call_site(self.at))} on {self.queue}: add srcStage {f(self.src_stages)}, "
                    f"srcAccess {f(self.src_access)}, dstStage {f(self.dst_stages)}, dstAccess {f(self.dst_access)}")
        return (f"insert a pipeline barrier on {self.queue} before {_short(self.at)}: "
                f"{f(self.src_stages)} / {f(self.src_access)} -> {f(self.dst_stages)} / {f(self.dst_access)}")


class Grouper:
    def __init__(self, frame, single_family=False, repair=True, max_repairs=60, log=None,
                 budget_mb=None, max_pairs=None):
        self.frame = frame
        self.single_family = single_family
        self.do_repair = repair
        self.max_repairs = max_repairs
        self.log = log or (lambda *a: None)
        self.checks = 0
        self.budget_mb = budget_mb if budget_mb is not None else _env_int("COVENANT_GROUP_BUDGET_MB", 8000)
        self.max_pairs = max_pairs if max_pairs is not None else _env_int("COVENANT_GROUP_MAX_PAIRS", 2_000_000)
        self.incomplete = []        # why the result is not the complete grouping
        self.pairs_known = True     # the base run's hazard pairs are known

    # -- the oracle ---------------------------------------------------------
    def _check(self, frame, exhaustive=False):
        self.checks += 1
        if self.budget_mb and rss_mb() > self.budget_mb:
            raise BudgetExceeded(f"resident memory over {self.budget_mb} MB before an oracle run")
        hz = HazardSink(self.max_pairs, self.budget_mb)
        r = timing.check(frame, Report(), hazards=hz, exhaustive=exhaustive, window=False)
        return r, list(hz)

    # Repairs are verified in pairwise mode and by event
    # identity, as elision is. An event's identity is its (queue, index) in the
    # original frame; _apply carries it over to widened barriers and gives inserted
    # barriers (never transitions, so never events) none.
    def _pairs(self, frame, hz, ident=None):
        """{int} of unordered event pairs by identity (one int per pair rather than a
        frozenset of tuples: about a fifth of the memory)."""
        qi = getattr(self, "_qindex", None)
        if qi is None:
            qi = self._qindex = {n: i for i, n in enumerate(self.frame.queues)}
        enc = lambda k: -1 if k is None else (qi.get(k[0], len(qi)) << 40) | k[1]
        out = set()
        for e, u, _ in hz:
            ie = (e[0], e[1]) if ident is None else ident.get(id(frame.queues[e[0]][e[1] - 1]))
            iu = (u[0], u[1]) if ident is None else ident.get(id(frame.queues[u[0]][u[1] - 1]))
            a, b = sorted((enc(ie), enc(iu)))
            out.add((a << 64) | (b & ((1 << 64) - 1)))
        return out

    def _op(self, frame, key):
        q, seq = key
        return frame.queues[q][seq - 1]

    # -- step 1: site groups ---------------------------------------------------
    def _pair_key(self, frame, code, e, u):
        oe, ou = self._op(frame, e), self._op(frame, u)
        we = site(oe.where) + (" [layout transition]" if isinstance(oe, Barrier) else "")
        wu = site(ou.where) + (" [layout transition]" if isinstance(ou, Barrier) else "")
        return (code, _SWAP.sub("[*]", str(ou.res)), we, wu)

    def _site_keys(self, frame, report, hz):
        """{key} of a check's result (for the no-new-diagnostic obligation)."""
        keys = {self._pair_key(frame, c, e, u) for e, u, c in hz}
        for d in report.items:
            if d.code not in PAIR_CODES:
                keys.add((d.code, "", "", site(d.where), message_key(d)))
        return keys

    def groups(self):
        t0 = time.time()
        fr = self.frame
        try:
            report, hz = self._check(fr)
        except BudgetExceeded as ex:
            # no hazard pairs; the diagnostics alone, grouped by (code, site, message)
            self.incomplete.append(f"base run: {ex}; diagnostics grouped by code, site and message only, "
                                   "no repairs")
            self.pairs_known, self.do_repair = False, False
            self.checks += 1
            report, hz = timing.check(fr, Report(), window=False), []
        hz = self._strip_held_guards(report, hz)
        fr = self.frame
        self.report, self.hz = report, hz
        by_key = OrderedDict()
        # diagnostics name the later access: match them to hazards by (code, where)
        diag_at = defaultdict(list)
        for d in report.items:
            diag_at[(d.code, d.where)].append(d)
        for e, u, code in hz:
            k = self._pair_key(fr, code, e, u)
            g = by_key.get(k)
            if g is None:
                g = by_key[k] = SiteGroup(k, code, k[1], k[2], k[3], [], [])
            ou = self._op(fr, u)
            ds = diag_at.get((code, ou.where), [])
            g.hazards.append(Hazard(code, e, u, ds[0] if ds else None))
            seen = g.__dict__.setdefault("_seen", set())
            for d in ds:
                if id(d) not in seen:
                    seen.add(id(d))
                    g.diagnostics.append(d)
        for d in report.items:
            if d.code in PAIR_CODES and self.pairs_known:
                continue
            k = (d.code, "", "", site(d.where), message_key(d))
            g = by_key.get(k)
            if g is None:
                g = by_key[k] = SiteGroup(k, d.code, "", "", site(d.where), [], [])
            g.diagnostics.append(d)
        self.site_groups = list(by_key.values())
        self.base_keys = set(by_key)
        if self.do_repair and PAIRWISE_VERIFY:
            try:
                r_pw, hz_pw = self._check(fr, exhaustive=True)
                self.base_keys_pw = self._site_keys(fr, r_pw, hz_pw) | self.base_keys
                self.base_pairs = self._pairs(fr, hz_pw)
                del r_pw, hz_pw
            except BudgetExceeded as ex:
                self.incomplete.append(f"pairwise base run: {ex}; no repair synthesized (none can be verified)")
                self.do_repair = False
                self.repair_skipped = "repair not attempted: pairwise oracle over the grouping budget"
        for g in self.site_groups:
            self._severity(g)
        self.log(f"{len(report.items)} diagnostics, {len(hz)} hazards -> {len(self.site_groups)} site groups "
                 f"({time.time() - t0:.1f} s)")
        return self.site_groups

    def _strip_held_guards(self, report, hz):
        """Drop from the frame the descriptor-array reads that a draw's IndexGuard excluded
        and that stayed excluded in the base run (timing.IndexGuard; skipped by the checker,
        but they are most of the frame's memory in GSK traces: 10 M of 10.7 M accesses in
        a GSK path-rendering trace). Sound for grouping: a repair only adds synchronization, so a guard
        read that races with nothing and sees no later host write in the base frame does so
        in every repaired frame too, and its excluded reads stay excluded. The base run's
        hazard positions are renumbered to the smaller frame."""
        failed = report.meta.get("timing.desc_refine.failed") if hasattr(report, "meta") else None
        if failed is None:
            return hz
        fr = self.frame
        queues, remap, dropped = {}, {}, 0
        for n, ops in fr.queues.items():
            keep, pos = [], []
            for o in ops:
                cg = getattr(o, "cond", None) if isinstance(o, Use) else None
                if cg is not None and cg.static is None and cg not in failed:
                    pos.append(None)
                    dropped += 1
                    continue
                keep.append(o)
                pos.append(len(keep))
            queues[n], remap[n] = keep, pos
        if not dropped:
            return hz
        self.frame = Frame(queues=queues, initial_layouts=dict(fr.initial_layouts), extents=dict(fr.extents),
                           swapchains=dict(fr.swapchains), semaphore_reuse=list(fr.semaphore_reuse),
                           findings=list(fr.findings), unsupported=dict(getattr(fr, "unsupported", {}) or {}))
        self.stripped = dropped
        self.log(f"{dropped} excluded descriptor-array reads dropped for grouping (guards held)")
        return [((e[0], remap[e[0]][e[1] - 1]), (u[0], remap[u[0]][u[1] - 1]), c) for e, u, c in hz]

    # -- severity -----------------------------------------------------------------
    def _discarded(self, h):
        fr = self.frame
        oe, ou = self._op(fr, h.e), self._op(fr, h.u)
        if isinstance(ou, Barrier) and ou.old is None and ou.new is not None and not (
                isinstance(oe, Use) and oe.access == "read"):
            return "the later access is a transition from UNDEFINED, which discards the contents"
        for o in (oe, ou):
            if isinstance(o, Use) and o.pass_op and "DONT_CARE" in o.where:
                return f"the {o.pass_op} operation is DONT_CARE: the contents are undefined anyway"
        return ""

    def _severity(self, g):
        sev = {d.severity for d in g.diagnostics} or {"error"}
        g.severity = "error" if "error" in sev else sorted(sev)[0]
        if g.code == "timing.acquire-first-use":
            g.severity, g.why_technical = "technicality", ("the image was never presented, so no "
                                                           "presentation-engine read is pending (spec rule only)")
            return
        if g.code in OWNERSHIP and self.single_family:
            g.severity, g.why_technical = "technicality", "the device has one queue family"
            return
        if g.hazards:
            whys = [self._discarded(h) for h in g.hazards]
            if all(whys):
                g.severity, g.why_technical = "technicality", whys[0]

    # -- step 2: repairs ----------------------------------------------------------
    def _candidates(self, h):
        fr = self.frame
        e, u = h.e, h.u
        if e[0] != u[0] or h.code not in SAME_QUEUE:
            return []
        q = u[0]
        ops = fr.queues[q]
        oe, ou = self._op(fr, e), self._op(fr, u)
        e_lt, u_lt = isinstance(oe, Barrier), isinstance(ou, Barrier)
        e_write = e_lt or oe.access == "write"
        F = frozenset
        st = lambda o: F({o.stage}) if isinstance(o, Use) and o.stage else F({"ALL_COMMANDS"})
        fl = lambda o, rd: (F({o.flag}) if isinstance(o, Use) and o.flag else
                            F({"MEMORY_READ" if rd else "MEMORY_WRITE"}))
        src_s = F() if e_lt else st(oe)
        src_a = F() if (e_lt or not e_write) else fl(oe, False)
        dst_s = F() if u_lt else st(ou)
        dst_a = F() if (u_lt or not e_write) else fl(ou, ou.access == "read")
        out = []
        # widen: the transition's own barrier, else the last barrier between them on the resource
        if e_lt:
            b = (oe, e[1] - 1)
        elif u_lt:
            b = (ou, u[1] - 1)
        else:
            b = None
            for i in range(u[1] - 2, e[1] - 1, -1):
                o = ops[i]
                if isinstance(o, Barrier) and o.res in (ou.res, None):
                    b = (o, i)
                    break
        if b is not None:
            o, i = b
            out.append(Repair("widen", q, o.where, src_s, src_a, dst_s, dst_a, o.res))
        # insert: before u's command, outside any render pass instance
        if not (e_lt or u_lt) and self._insert_pos(ops, u[1] - 1, e[1] - 1) is not None:
            out.append(Repair("insert", q, ou.where, src_s, src_a, dst_s, dst_a, ou.res))
        return out

    @staticmethod
    def _insert_pos(ops, i, lo=-1):
        """Where a barrier for the access at index i goes: before the first op
        of its command, and before the render pass instance it is in; None if
        that would not be after index lo."""
        p = i
        cmd = getattr(ops[i], "cmd", None)
        while p > 0 and cmd is not None and getattr(ops[p - 1], "cmd", None) is cmd:
            p -= 1
        while p > 0:
            inst = getattr(ops[p - 1], "instance", None)
            if inst is None or not any(getattr(o, "instance", None) is inst for o in ops[p:i + 1]):
                break
            p -= 1
        return p if p > lo else None

    def _apply(self, rep):
        """The frame with the repair applied to every occurrence of its site
        (a recorded command runs once per submission of its command buffer)."""
        fr = self.frame
        queues = {n: list(ops) for n, ops in fr.queues.items()}
        self._ident = {id(o): (n, i + 1) for n, ops in queues.items() for i, o in enumerate(ops)}
        ops = queues[rep.queue]
        at = site(rep.at)
        if rep.kind == "widen":
            at = call_site(rep.at)
            for i, o in enumerate(ops):
                if isinstance(o, Barrier) and call_site(o.where) == at:
                    w = lambda a, b: None if a is None else frozenset(a) | b
                    ops[i] = dataclasses.replace(o, src_stages=w(o.src_stages, rep.src_stages),
                                                 src_access=w(o.src_access, rep.src_access),
                                                 dst_stages=w(o.dst_stages, rep.dst_stages),
                                                 dst_access=w(o.dst_access, rep.dst_access))
                    self._ident[id(ops[i])] = (rep.queue, i + 1)
        else:
            pos = sorted({p for i, o in enumerate(ops)
                          if isinstance(o, Use) and o.res == rep.res and site(o.where) == at
                          for p in [self._insert_pos(ops, i)] if p is not None}, reverse=True)
            for p in pos:
                ops.insert(p, Barrier(None, None, None, f"(repair) before {rep.at}", rep.src_stages,
                                      rep.src_access, rep.dst_stages, rep.dst_access))
        return Frame(queues=queues, initial_layouts=dict(fr.initial_layouts), extents=dict(fr.extents),
                     swapchains=dict(fr.swapchains),
                     semaphore_reuse=list(fr.semaphore_reuse), findings=list(fr.findings))

    def _verify(self, g, rep):
        f2 = self._apply(rep)
        if not PAIRWISE_VERIFY:     # weaker rule: epoch mode, site keys only
            r2, hz2 = self._check(f2)
            keys = self._site_keys(f2, r2, hz2)
            if keys - self.base_keys:
                return None, f"adds {len(keys - self.base_keys)} new diagnostic site(s)"
            if g.key in keys:
                return None, "does not remove the group's hazards"
            return keys, ""
        # pairwise mode, so a pair that is gone is not a race (Theorem 1);
        # no new site key and no new racing event pair by identity (a repair only
        # adds synchronization, so by Lemma (repair monotonicity) it cannot add a
        # race; this checks that the checker agrees on the repaired frame)
        r2, hz2 = self._check(f2, exhaustive=True)
        keys = self._site_keys(f2, r2, hz2)
        if keys - self.base_keys_pw:
            return None, f"adds {len(keys - self.base_keys_pw)} new diagnostic site(s)"
        new_pairs = self._pairs(f2, hz2, self._ident) - self.base_pairs
        if new_pairs:
            self.new_pair_rejects = getattr(self, "new_pair_rejects", 0) + 1
            return None, f"adds {len(new_pairs)} new racing event pair(s)"
        if g.key in keys:
            return None, "does not remove the group's hazards"
        return keys, ""

    def repairs(self):
        self.repair_list = []
        if not self.do_repair:
            why = getattr(self, "repair_skipped", None)
            if why:
                for g in self.site_groups:
                    g.repair_note = why if g.hazards and g.code in SAME_QUEUE else (
                        "needs a semaphore or a fence wait (not synthesized)" if g.hazards else "not a barrier problem")
            return []
        stopped = None
        fixed_by = []
        tried = 0
        for g in sorted(self.site_groups, key=lambda g: -len(g.hazards)):
            if not g.hazards or g.code not in SAME_QUEUE:
                g.repair_note = ("needs a semaphore or a fence wait (not synthesized)" if g.hazards else
                                 "not a barrier problem")
                continue
            if any(g.key in fx for _, fx in fixed_by):
                continue            # already fixed by an earlier repair
            if tried >= self.max_repairs:
                g.repair_note = "repair search budget exhausted"
                continue
            if stopped:
                g.repair_note = stopped
                continue
            notes = []
            for rep in self._candidates(g.hazards[0]):
                tried += 1
                try:
                    keys, why = self._verify(g, rep)
                except BudgetExceeded as ex:
                    stopped = "repair search stopped: verification over the grouping budget"
                    self.incomplete.append(f"verification run {tried}: {ex}; repair search stopped")
                    g.repair_note = stopped
                    break
                if keys is None:
                    notes.append(f"{rep.kind}: {why}")
                    continue
                fx = self.base_keys - keys
                rep.fixes = sorted(fx, key=str)
                fixed_by.append((rep, fx))
                break
            else:
                g.repair_note = "; ".join(notes) or "no candidate repair"
            if stopped and not g.repair_note:
                g.repair_note = stopped
        # merge: largest fix sets first, each site group to the first repair that removes it
        fixed_by.sort(key=lambda x: -len(x[1]))
        assigned = {}
        for rep, fx in fixed_by:
            mine = [k for k in fx if k not in assigned]
            if not mine:
                continue
            for k in mine:
                assigned[k] = rep
            rep.fixes = sorted(mine, key=str)
            self.repair_list.append(rep)
        for g in self.site_groups:
            if g.key in assigned:
                g.repair = assigned[g.key]
        return self.repair_list

    # -- output ---------------------------------------------------------------------
    def result(self):
        sg = self.site_groups
        # the same repair in another command buffer (one per frame in flight) is one repair
        fam = OrderedDict()
        for rep in self.repair_list:
            fam.setdefault(_CB.sub("command_buffer#*", site(rep.describe())), []).append(rep)
        rg = []
        for k, reps in fam.items():
            rep = reps[0]
            members = [g for g in sg if g.repair in reps]
            sev = "error" if any(g.severity == "error" for g in members) else (
                "technicality" if all(g.severity == "technicality" for g in members) else members[0].severity)
            rg.append(OrderedDict(repair=rep.describe(), kind=rep.kind, queue=rep.queue, severity=sev,
                                  occurrences=len(reps),
                                  site_groups=len(members), diagnostics=len({id(d) for g in members for d in g.diagnostics}),
                                  hazards=sum(len(g.hazards) for g in members),
                                  codes=sorted({g.code for g in members}),
                                  resources=sorted({g.resource for g in members})[:12]))
        unrep = [g for g in sg if g.repair is None]
        # unrepaired site groups that differ only by the command buffer (the same
        # recording code run into one secondary / per-frame command buffer each) are
        # one group, as repairs in several command buffers already are
        ufam = OrderedDict()
        for g in unrep:
            k = (g.code, g.resource, _CB.sub("command_buffer#*", g.first), _CB.sub("command_buffer#*", g.second),
                 _CB.sub("command_buffer#*", g.key[4]) if len(g.key) > 4 else "")
            ufam.setdefault(k, []).append(g)
        og = []
        for gs in ufam.values():
            g = gs[0]
            sev = "error" if any(x.severity == "error" for x in gs) else (
                "technicality" if all(x.severity == "technicality" for x in gs) else g.severity)
            og.append(OrderedDict(code=g.code, severity=sev, resource=g.resource,
                                  earlier=_short(g.first), later=_short(g.second),
                                  diagnostics=len({id(d) for x in gs for d in x.diagnostics}),
                                  hazards=sum(len(x.hazards) for x in gs),
                                  occurrences=len(gs),
                                  why_technical=g.why_technical if sev == "technicality" else "",
                                  note=g.repair_note))
        # every diagnostic of the report is in exactly one group, budget or not
        member_ids = {id(d) for g in sg for d in g.diagnostics}
        ungrouped = [d for d in self.report.items if id(d) not in member_ids]
        for d in ungrouped:
            og.append(OrderedDict(code=d.code, severity=d.severity, resource="", earlier="", later=_short(d.where),
                                  diagnostics=1, hazards=0, occurrences=1, why_technical="",
                                  note="not matched to a hazard pair"))
        return OrderedDict(
            status="incomplete" if self.incomplete else "complete",
            incomplete_reasons=list(self.incomplete),
            budget=OrderedDict(budget_mb=self.budget_mb, max_pairs=self.max_pairs, max_repairs=self.max_repairs),
            diagnostics=len(self.report.items), hazards=len(self.hz), site_groups=len(sg),
            repair_groups=len(rg), unrepaired_site_groups=len(unrep),
            reported_groups=len(rg) + len(og),
            technicalities=sum(1 for g in og if g["severity"] == "technicality") +
            sum(1 for r in rg if r["severity"] == "technicality"),
            oracle_runs=self.checks,
            repair_verify="pairwise, event identity" if PAIRWISE_VERIFY else "epoch, site keys",
            excluded_reads_dropped=getattr(self, "stripped", 0),
            new_pair_rejects=getattr(self, "new_pair_rejects", 0),
            repairs=rg,
            other_groups=og)


def _env_int(name, default):
    try:
        return int(_os.environ.get(name, "") or default)
    except ValueError:
        return default


def group(frame, single_family=False, repair=True, max_repairs=60, log=None, budget_mb=None, max_pairs=None):
    g = Grouper(frame, single_family, repair, max_repairs, log, budget_mb, max_pairs)
    g.groups()
    g.repairs()
    return g.result()


def format_result(res):
    lines = []
    if res.get("status") == "incomplete":
        lines.append("REPORTING INCOMPLETE (grouping budget hit; every diagnostic is still listed): "
                     + " | ".join(res.get("incomplete_reasons", [])))
    lines += [f"{res['diagnostics']} diagnostics ({res['hazards']} hazard pairs) -> {res['site_groups']} site groups "
             f"-> {res['reported_groups']} groups ({res['repair_groups']} with a verified repair, "
             f"{res['technicalities']} technicalities)"]
    for i, r in enumerate(res["repairs"], 1):
        occ = f", in {r['occurrences']} command buffers" if r["occurrences"] > 1 else ""
        lines.append(f"  R{i} [{r['severity']}] {r['diagnostics']} diagnostics in {r['site_groups']} site groups{occ} "
                     f"({', '.join(r['codes'])}; {', '.join(r['resources'][:4])})")
        lines.append(f"      fix: {r['repair']}")
    for i, g in enumerate(res["other_groups"], 1):
        occ = f" (in {g['occurrences']} command buffers)" if g.get("occurrences", 1) > 1 else ""
        lines.append(f"  G{i} [{g['severity']}] {g['code']} x{g['diagnostics']} on {g['resource'] or '-'}{occ}: "
                     f"{g['earlier'] + ' -> ' if g['earlier'] else ''}{g['later']}")
        if g["why_technical"]:
            lines.append(f"      technicality: {g['why_technical']}")
        if g["note"]:
            lines.append(f"      {g['note']}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("trace")
    ap.add_argument("--spirv-dir")
    ap.add_argument("--json")
    ap.add_argument("--no-repair", action="store_true")
    ap.add_argument("--single-family", action="store_true", default=None,
                    help="the device has one queue family (default: when the trace uses one)")
    ap.add_argument("--max-repairs", type=int, default=60, help="candidate repairs to verify at most")
    ap.add_argument("--budget-mb", type=int, default=None,
                    help="resident-memory budget of the oracle runs (default 8000, env COVENANT_GROUP_BUDGET_MB)")
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="hazard pairs one oracle run may hold (default 2000000, env COVENANT_GROUP_MAX_PAIRS)")
    a = ap.parse_args(argv)
    if a.json:
        # a run killed from outside leaves an incomplete result, never a missing or clean one
        with open(a.json, "w") as f:
            json.dump({"status": "incomplete", "incomplete_reasons": ["grouping did not finish"]}, f)
    from .host_trace import load_trace
    m = load_trace(a.trace, spirv_dir=a.spirv_dir)
    fam = {r.get("family") for r in m.records if r.get("fn") == "vkGetDeviceQueue"}
    single = a.single_family if a.single_family is not None else len(fam) <= 1
    # grouping needs only the frame; the raw records (hundreds of MB for long traces) go
    m.records = None
    import gc
    gc.collect()
    g = Grouper(m.frame, single, not a.no_repair, a.max_repairs, lambda *x: print(*x, file=sys.stderr),
                a.budget_mb, a.max_pairs)
    del m           # the grouper may replace the frame by a smaller one (_strip_held_guards)
    gc.collect()
    g.groups()
    g.repairs()
    res = g.result()
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1, default=str)
    print(format_result(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
