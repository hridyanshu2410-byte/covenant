"""Proof-licensed barrier elision.

Given a model-driver trace (see model_driver/README.md), decide for every
recorded vkCmdPipelineBarrier / vkCmdPipelineBarrier2 call whether it is

* ``required``  -- kept as recorded,
* ``remove``    -- dropped entirely, or
* ``narrow``    -- kept, but some of its barrier structs are dropped and/or
                   their stage/access masks are replaced by narrower ones,

such that the program still has no hazard under ANY schedule its
synchronization allows. The plan is consumed by the VK_LAYER_COVENANT_elide
Vulkan layer (elide_layer/), which applies it on a real driver.

    python3 -m covenant.elide trace.jsonl [--spirv-dir DIR] [-o plan.json]

What is proved, and how
-----------------------
The oracle is ``covenant.timing.check``: vector-clock happens-before over every
queue, which reports any pair of conflicting accesses (same resource, at least
one write) that the recorded barriers/semaphores/fences do not order, plus
layout errors. A change is accepted only when re-running the oracle on the
changed frame reports no diagnostic that the unchanged frame did not already
report ("no new hazards"). The search is greedy in recording order and every
step is checked on top of all changes accepted before it, so the final plan
is checked as a whole (and re-checked once more at the end): barriers that are
individually redundant but jointly needed are never removed together.

Before handing a frame to the oracle, elide.py strengthens it so that the
oracle's model is at least as strict as Vulkan's for the obligations that
matter here (the strengthened frame is used for BOTH the baseline and every
candidate, so "no new hazards" is relative to the same model):

1. Layout transitions are accesses. A barrier with a layout transition (or an
   ownership transfer that carries one) is split into
   ``B1 (src masks -> LT) ; write(res, LAYOUT_TRANSITION) ; B2 (LT -> dst masks)``,
   so a narrowed src mask must still order every earlier access before the
   transition and a narrowed dst mask every later access after it.
   Transition barriers are never removed (only narrowed).
2. Write-after-write needs the dst access too (syncval and the spec want the
   later write in the dst access scope). timing.check checks this itself;
   the alternative, a read "proxy" of the same stage/flag before each
   write Use (sharing a private scope with its write), is available as
   Elider.waw_proxy = True.
3. Attachment load/store operations are accesses. host_trace models them as
   Uses at vkCmdBeginRendering / vkCmdEndRendering (and at the first / last
   subpass using an attachment of a render pass), in the instance's scope.
   (Frames without them, inject_load_store=True, get them injected around the
   first/last draw of each rendering instance.)
4. Every acquire hands the image over from the presentation engine.
   host_trace models the presentation engine's read of a swapchain image
   only from its first present on, so without this the FIRST acquire of an
   image has no access to order against, and the prover could narrow the
   acquire barrier's srcStageMask from COLOR_ATTACHMENT_OUTPUT (the acquire
   semaphore's wait stage) to TOP_OF_PIPE, breaking the chain with the
   semaphore wait (Khronos synchronization examples, "swapchain image
   acquire and present"; the spec lets the application touch the image only
   after the acquire semaphore/fence signals). build() therefore puts a
   presentation-engine read of the image before EVERY acquire signal
   (Elider.acquire_read = False omits it).

Things the oracle cannot see are never touched ("pinned"):
* any barrier struct with HOST in a stage mask (host reads are not traced),
* queue family ownership transfers (release/acquire pairs),
* masks the trace could not name (raw integers),
* every barrier in a command-buffer recording that contains a command the
  model does not interpret (events, secondary command buffers, query copies,
  a render pass the trace did not record, any ``unimplemented`` command other
  than timestamps, query resets and debug labels/markers), or a rendering
  instance without draws or with resolves. (Legacy render passes are
  interpreted: subpass dependencies, the implicit EXTERNAL dependencies,
  attachment layout transitions and load/store operations are timing ops of
  host_trace, so they do not pin their recording;
  ``Elider.pin_legacy_render_passes = True`` pins it.)
* barriers recorded but never submitted in the trace,
* every barrier in a submission batch that signals a semaphore with a stage
  mask narrower than ALL_COMMANDS (timing.Signal has no stage mask, so the
  oracle cannot see what such a signal leaves unordered).

Narrowing candidates are built from the accesses actually around each
occurrence (stages of earlier accesses / flags of earlier writes for the src
side; stages and flags of later accesses for the dst side), first since the
previous barrier ("local"), then over the whole queue history ("wide"); a
candidate is always a subset of the recorded masks (never adds a stage the app
did not ask for), uses only legacy (sync1) bit names so it is valid for both
entry points, and adds LATE_FRAGMENT_TESTS next to EARLY_FRAGMENT_TESTS and
attachment READ next to attachment WRITE (real hardware does both).

Limits (stated, not hidden): the proof covers the executed path recorded in
the trace (host control flow is observed, not analysed), and inherits the
model's resolution (whole buffers, images per mip level, every bound
descriptor assumed used). Cross-submission ordering through a host fence wait
is treated as a full dependency, as timing.py and syncval do.

Barrier identity (stable across runs)
-------------------------------------
Two keys are written for every recorded barrier call; the layer can match by
either, and in both modes refuses to touch a call whose content signature
differs from the plan's.

* strict: ``(rec, idx)`` -- ``rec`` is the 0-based ordinal of the
  vkBeginCommandBuffer call, counted over the whole process in call order;
  ``idx`` is the 0-based index of the barrier call within that recording
  (vkCmdPipelineBarrier, vkCmdPipelineBarrier2 and ...2KHR all count). This is
  exactly what was proved, and needs the program to record command buffers in
  the same order (single-threaded or deterministic recording).
* chain: ``chain`` -- FNV-1a 64 over the signatures of all barrier calls of the
  recording up to and including this one (the hash state continues from the
  previous barrier's chain value). Independent of recording order and thread,
  and of how many frames run: in a program that re-records the same frame, the
  j-th barrier of every frame has the same chain value. Chain entries are the
  conservative merge of every recorded instance with that key (any instance
  that needs a barrier struct keeps it; narrowed masks are unioned), and the
  merged plan is re-verified on the whole trace before it is written.
* ``sig`` -- FNV-1a 64 of the call's canonical content string (handles
  excluded): ``"<1|2>|<dependencyFlags>"`` then, in order, ``"|M:ss:sa:ds:da"``
  per memory barrier, ``"|B:ss:sa:ds:da:sfam:dfam:offset:size"`` per buffer
  barrier, ``"|I:ss:sa:ds:da:sfam:dfam:old:new:aspect:baseMip:levels:baseLayer:layers"``
  per image barrier, and ``"|E:ss:ds"`` for a sync1 call with no structs; all
  numbers decimal, sync1 stage masks repeated in every struct, families as
  recorded (VK_QUEUE_FAMILY_IGNORED = 4294967295), VK_WHOLE_SIZE = 2^64-1.
"""
import argparse
import copy
import hashlib
import json
import os
import dataclasses
import re
import sys
import time
from collections import OrderedDict

from . import host_trace, timing
from .host_trace import _HOST_NO_MEMORY_EFFECT, _NO_MEMORY_EFFECT
from .diagnostics import Report
from .timing import Barrier, Use, Wait, Frame

FNV_OFFSET = 0xcbf29ce484222325
FNV_PRIME = 0x100000001b3
QUEUE_FAMILY_IGNORED = 0xFFFFFFFF
WHOLE_SIZE = 0xFFFFFFFFFFFFFFFF
LT_STAGE, LT_FLAG = "LAYOUT_TRANSITION", "LAYOUT_TRANSITION_WRITE"
BARRIER_FNS = ("vkCmdPipelineBarrier", "vkCmdPipelineBarrier2", "vkCmdPipelineBarrier2KHR")
# unimplemented commands that touch no resource the barriers order
HARMLESS_UNMODELLED = {"vkCmdWriteTimestamp", "vkCmdWriteTimestamp2", "vkCmdWriteTimestamp2KHR",
                       "vkCmdResetQueryPool", "vkCmdBeginQuery", "vkCmdEndQuery",
                       # debug annotations: no resource access at all
                       "vkCmdDebugMarkerBeginEXT", "vkCmdDebugMarkerEndEXT", "vkCmdDebugMarkerInsertEXT",
                       "vkCmdBeginDebugUtilsLabelEXT", "vkCmdEndDebugUtilsLabelEXT",
                       "vkCmdInsertDebugUtilsLabelEXT"}
# legacy render passes: modelled by host_trace (begin/next/end; the driver records the *2 entry
# points under these names), pinned only with Elider.pin_legacy_render_passes
LEGACY_RP_FNS = {"vkCmdBeginRenderPass", "vkCmdBeginRenderPass2", "vkCmdBeginRenderPass2KHR"}
PIN_FNS = {"vkCmdExecuteCommands", "vkCmdWaitEvents", "vkCmdWaitEvents2", "vkCmdSetEvent",
           "vkCmdSetEvent2", "vkCmdResetEvent", "vkCmdResetEvent2", "vkCmdCopyQueryPoolResults"}
# host_trace access names -> legacy (sync1-valid) names
ACCESS_LEGACY = {"SHADER_SAMPLED_READ": "SHADER_READ", "SHADER_STORAGE_READ": "SHADER_READ",
                 "SHADER_STORAGE_WRITE": "SHADER_WRITE"}
WRITE_IMPLIES_READ = {"COLOR_ATTACHMENT_WRITE": "COLOR_ATTACHMENT_READ",
                      "DEPTH_STENCIL_ATTACHMENT_WRITE": "DEPTH_STENCIL_ATTACHMENT_READ"}
STAGE_PAIRS = {"EARLY_FRAGMENT_TESTS": "LATE_FRAGMENT_TESTS", "LATE_FRAGMENT_TESTS": "EARLY_FRAGMENT_TESTS"}
_WHERE = re.compile(r"^(vkCmdPipelineBarrier2?(?:KHR)?) @seq (\d+)\b.*?submitted @seq (\d+) on ")
_CMD_WHERE = re.compile(r"^(vkCmd\w+) @seq (\d+)\b.*?submitted @seq (\d+) on ")
_ANY_WHERE = re.compile(r"^vkCmd\w+ @seq (\d+)\b.*?submitted @seq (\d+) on ")


def fnv1a(data, h=FNV_OFFSET):
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


# ---------------------------------------------------------------- Vulkan names
class VkNames:
    """Name <-> value tables read from vulkan_core.h (stage/access 2 bits,
    image layouts, aspect and dependency flags)."""

    def __init__(self, header=None):
        header = header or self.find_header()
        self.values = {}
        aliases = {}
        text = open(header).read()
        for m in re.finditer(r"static const Vk\w+FlagBits2(?:KHR)? (VK_\w+) = (0x[0-9A-Fa-f]+)ULL;", text):
            self.values[m.group(1)] = int(m.group(2), 16)
        for m in re.finditer(r"static const Vk\w+FlagBits2(?:KHR)? (VK_\w+) = (VK_\w+);", text):
            aliases[m.group(1)] = m.group(2)
        for m in re.finditer(r"^\s+(VK_[A-Z0-9_]+) = (-?0x[0-9A-Fa-f]+|-?\d+|VK_[A-Z0-9_]+),?\s*$", text, re.M):
            v = m.group(2)
            if v.startswith("VK_"):
                aliases.setdefault(m.group(1), v)
            else:
                self.values.setdefault(m.group(1), int(v, 0))
        for _ in range(4):
            for a, b in aliases.items():
                if a not in self.values and b in self.values:
                    self.values[a] = self.values[b]

    @staticmethod
    def find_header():
        cands = [os.environ.get("VULKAN_HEADER"), "/usr/include/vulkan/vulkan_core.h"]
        if os.environ.get("VULKAN_SDK"):
            cands.append(os.path.join(os.environ["VULKAN_SDK"], "include", "vulkan", "vulkan_core.h"))
        for c in cands:
            if c and os.path.exists(c):
                return c
        raise FileNotFoundError("vulkan_core.h not found; set VULKAN_HEADER")

    def flags(self, names):
        """Trace flag list (names and/or raw ints) -> int, or None if a name is unknown."""
        v = 0
        for n in names or ():
            if isinstance(n, int):
                v |= n
            elif n in self.values:
                v |= self.values[n]
            elif n.endswith("_NONE") or n.endswith("_NONE_KHR"):
                continue
            else:
                return None
        return v

    def enum(self, name):
        if isinstance(name, int) or name is None:
            return name or 0
        return self.values[name]

    def stage_bits(self, short):
        return 0 if not short else self._bits(short, "VK_PIPELINE_STAGE_2_")

    def access_bits(self, short):
        return 0 if not short else self._bits(short, "VK_ACCESS_2_")

    def _bits(self, short, prefix):
        v = 0
        for s in short:
            v |= self.values[f"{prefix}{s}_BIT"]
        return v


# ---------------------------------------------------------------- call model
class Elem:
    """One barrier struct of a call (kind M/B/I, or E for a sync1 call with none)."""

    def __init__(self, kind, i, raw):
        self.kind, self.i, self.raw = kind, i, raw
        self.pin = None          # reason it may not be removed or narrowed
        self.transition = False  # may be narrowed, never removed
        self.res = []            # resource names of its timing ops

    @property
    def label(self):
        return {"M": "memory", "B": "buffer", "I": "image", "E": "exec"}[self.kind] + f"[{self.i}]"

    def masks(self):
        r = self.raw
        return (host_trace.stage_mask(r.get("src_stage")), host_trace.access_mask(r.get("src_access", [])),
                host_trace.stage_mask(r.get("dst_stage")), host_trace.access_mask(r.get("dst_access", [])))


class Call:
    def __init__(self, rec, rec_ord, idx):
        self.rec = rec
        self.seq, self.fn, self.cb = rec["seq"], rec["fn"], rec["cb"]
        self.sync1 = rec["fn"] == "vkCmdPipelineBarrier"
        self.rec_ord, self.idx = rec_ord, idx
        self.elems = []
        for i, b in enumerate(rec.get("memory_barriers") or []):
            self.elems.append(Elem("M", i, b))
        for i, b in enumerate(rec.get("buffer_barriers") or []):
            self.elems.append(Elem("B", i, b))
        for i, b in enumerate(rec.get("image_barriers") or []):
            self.elems.append(Elem("I", i, b))
        if not self.elems and self.sync1:
            self.elems.append(Elem("E", 0, {"src_stage": rec.get("src_stage"), "dst_stage": rec.get("dst_stage"),
                                            "src_access": [], "dst_access": []}))
        self.pin = None
        self.occurrences = 0
        self.submits = set()     # seqs of the vkQueueSubmit calls that executed it
        self.sig = self.chain = None

    @property
    def key(self):
        return (self.rec_ord, self.idx)


def call_signature(call, names):
    r = call.rec
    v = "1" if call.sync1 else "2"
    s = f"{v}|{names.flags(r.get('dependency_flags'))}"

    def fam(x):
        return QUEUE_FAMILY_IGNORED if x is None else x

    def m4(b):
        return (f"{names.flags(b.get('src_stage'))}:{names.flags(b.get('src_access'))}:"
                f"{names.flags(b.get('dst_stage'))}:{names.flags(b.get('dst_access'))}")
    for b in r.get("memory_barriers") or []:
        s += f"|M:{m4(b)}"
    for b in r.get("buffer_barriers") or []:
        size = WHOLE_SIZE if b.get("size") == "whole" else b.get("size")
        s += f"|B:{m4(b)}:{fam(b.get('src_family'))}:{fam(b.get('dst_family'))}:{b.get('offset', 0)}:{size}"
    for b in r.get("image_barriers") or []:
        s += (f"|I:{m4(b)}:{fam(b.get('src_family'))}:{fam(b.get('dst_family'))}:"
              f"{names.enum(b.get('old_layout'))}:{names.enum(b.get('new_layout'))}:{names.flags(b.get('aspect'))}:"
              f"{b.get('base_mip', 0)}:{b.get('level_count', 0)}:{b.get('base_layer', 0)}:{b.get('layer_count', 0)}")
    if call.sync1 and not any(r.get(k) for k in ("memory_barriers", "buffer_barriers", "image_barriers")):
        s += f"|E:{names.flags(r.get('src_stage'))}:{names.flags(r.get('dst_stage'))}"
    return s


# ---------------------------------------------------------------- the analysis
class Elider:
    # legacy render passes are interpreted by host_trace, so they do not pin
    # every barrier of their recording (True = pin them)
    pin_legacy_render_passes = False
    # pin barriers between the accesses of a baseline hazard
    taint_baseline_hazards = True
    # final re-check in exhaustive (pairwise) mode
    pairwise_recheck = True
    # strengthening 4 of the module docstring (a presentation-engine read before every acquire signal)
    acquire_read = True

    def __init__(self, trace, spirv_dir=None, header=None, log=None):
        self.trace = trace
        self.names = VkNames(header)
        self.log = log or (lambda *a: None)
        recs = host_trace.load_records(trace)
        self.loader = host_trace._Loader(recs, spirv_dir if spirv_dir is not None else os.environ.get("COVENANT_SPIRV_DIR"))
        self.model = self.loader.run()
        self.recs = recs
        self.frame = self.model.frame
        # host_trace models attachment load/store operations itself
        self.inject_load_store = False
        # timing.check demands the later write of a WAW pair in the dst
        # access scope itself, so the read proxies of
        # obligation 2 are off; with them, a proxy (a read carrying a write
        # flag) is outside every SHADER_WRITE / MEMORY_WRITE access scope
        self.waw_proxy = False
        self.checks = 0
        self._collect_calls()
        self._map_occurrences()
        self._collect_render_instances()
        self._pin()

    # -- calls, recordings, keys ------------------------------------------
    def _collect_calls(self):
        self.calls = []                 # in trace order
        self.by_seq = {}
        self.recording_of_seq = {}      # cmd seq -> recording ordinal
        self.recordings = {}            # ordinal -> {"cb", "cmds": [records]}
        cur = {}
        n = -1
        # an unmodelled command the trace cannot place in a recording (a stub
        # recorded without its command buffer: older model drivers) or an
        # unmodelled host call: its accesses and synchronization could sit next
        # to any barrier, so every barrier is pinned
        self.global_pin = None
        for r in self.recs:
            fn = r["fn"]
            if r.get("warning") and fn not in HARMLESS_UNMODELLED and self.global_pin is None and (
                    (fn.startswith("vkCmd") and r.get("cb") not in cur) or
                    (not fn.startswith("vkCmd") and not fn.startswith(_HOST_NO_MEMORY_EFFECT)
                     and fn not in _NO_MEMORY_EFFECT)):
                self.global_pin = f"the trace holds {fn} @seq {r['seq']}, which the model does not interpret " \
                                  "and cannot place in a command buffer"
            if fn == "vkBeginCommandBuffer":
                n += 1
                cur[r["cb"]] = {"ord": n, "idx": 0, "chain": FNV_OFFSET}
                self.recordings[n] = {"cb": r["cb"], "cmds": []}
            elif fn.startswith("vkCmd") and "cb" in r and r["cb"] in cur:
                st = cur[r["cb"]]
                self.recordings[st["ord"]]["cmds"].append(r)
                self.recording_of_seq[r["seq"]] = st["ord"]
                if fn in BARRIER_FNS:
                    c = Call(r, st["ord"], st["idx"])
                    sig = call_signature(c, self.names)
                    c.sig_text = sig
                    c.sig = f"{fnv1a(sig.encode()):016x}"
                    st["chain"] = fnv1a(sig.encode(), st["chain"])
                    c.chain = f"{st['chain']:016x}"
                    st["idx"] += 1
                    self.calls.append(c)
                    self.by_seq[r["seq"]] = c
            elif fn in ("vkEndCommandBuffer",):
                pass

    def _expected_ops(self, call):
        L = self.loader
        out = []
        for e in call.elems:
            if e.kind == "I":
                b = e.raw
                for x in L.img_res(b["image"], b.get("base_mip", 0), b.get("level_count")):
                    out.append((e, x))
        for e in call.elems:
            if e.kind == "B":
                out.append((e, L.n(e.raw["buffer"])))
        for e in call.elems:
            if e.kind == "M":
                out.append((e, None))
        for e in call.elems:
            if e.kind == "E":
                out.append((e, None))
        return out

    def _map_occurrences(self):
        """id(Barrier op) -> (call, elem); every op of a barrier call is mapped
        or the call is pinned."""
        self.op_elem = {}
        runs = {}       # (call seq, submit seq, queue) -> [ops]
        for qn, ops in self.frame.queues.items():
            prev = None
            for op in ops:
                if isinstance(op, Barrier):
                    m = _WHERE.match(op.where)
                    if m and int(m.group(2)) in self.by_seq:
                        k = (int(m.group(2)), int(m.group(3)), qn)
                        if k != prev:
                            runs.setdefault(k, []).append([])
                        runs[k][-1].append(op)
                        prev = k
                        continue
                prev = None
        for (seq, sub, _), groups in runs.items():
            call = self.by_seq[seq]
            call.submits.add(sub)
            exp = self._expected_ops(call)
            for e, x in exp:
                if x not in e.res:
                    e.res.append(x)
            for g in groups:
                if not exp or len(g) % len(exp):
                    call.pin = call.pin or "its timing ops could not be matched to its barrier structs"
                    continue
                for k in range(0, len(g), len(exp)):
                    chunk = g[k:k + len(exp)]
                    if any(op.res != x for op, (e, x) in zip(chunk, exp)):
                        call.pin = call.pin or "its timing ops could not be matched to its barrier structs"
                        continue
                    call.occurrences += 1
                    for op, (e, _) in zip(chunk, exp):
                        self.op_elem[id(op)] = (call, e)

    # -- rendering instances (load/store obligations) ---------------------
    def _collect_render_instances(self):
        """Per recording: dynamic rendering instances with their draws."""
        self.instances = []     # dicts: rec, begin, end, draws(seqs), atts
        self.bad_recordings = {}
        L = self.loader
        for ordn, rc in self.recordings.items():
            inst = None
            for c in rc["cmds"]:
                fn = c["fn"]
                if fn in PIN_FNS or (fn in LEGACY_RP_FNS and self.pin_legacy_render_passes):
                    self.bad_recordings.setdefault(ordn, f"it records {fn}, whose synchronization the model does not interpret")
                elif fn in LEGACY_RP_FNS and c.get("render_pass") not in L.render_passes:
                    self.bad_recordings.setdefault(ordn, f"it records {fn} with a render pass the trace did not create")
                if c.get("warning") and fn not in HARMLESS_UNMODELLED:
                    self.bad_recordings.setdefault(ordn, f"it records {fn}, which the model driver does not interpret")
                if fn in ("vkCmdBeginRendering", "vkCmdBeginRenderingKHR"):
                    atts = []
                    for a in c.get("color") or []:
                        if a and a.get("view"):
                            atts.append(("color", a))
                    for role in ("depth", "stencil"):
                        a = c.get(role)
                        if a and a.get("view"):
                            atts.append((role, a))
                    if any(a.get("resolve_view") for _, a in atts):
                        self.bad_recordings.setdefault(ordn, "a rendering instance resolves (not modelled)")
                    inst = {"rec": ordn, "begin": c["seq"], "draws": [], "atts": atts}
                elif fn in ("vkCmdEndRendering", "vkCmdEndRenderingKHR") and inst:
                    if not inst["draws"]:
                        self.bad_recordings.setdefault(
                            ordn, "a rendering instance has no draw, so its load/store operations are not modelled")
                    else:
                        self.instances.append(inst)
                    inst = None
                elif fn.startswith("vkCmdDraw") and inst is not None:
                    inst["draws"].append(c["seq"])
        for inst in self.instances:
            for role, a in inst["atts"]:
                a["_res"] = L.view_res(a["view"])

    def _staged_signal_batches(self):
        """{(submit seq, command buffer)} of batches that signal a semaphore with
        a stage mask narrower than ALL_COMMANDS. timing.Signal carries no stage
        mask, so the oracle treats every signal as ALL_COMMANDS; only
        host_trace's static timing.signal-scope finding looks at it, and that
        finding is computed once from the recorded barriers. Barriers in such a
        batch are therefore pinned."""
        out = set()
        for r in self.recs:
            if r["fn"] not in ("vkQueueSubmit", "vkQueueSubmit2", "vkQueueSubmit2KHR"):
                continue
            for b in r.get("submits") or []:
                for sg in b.get("signal") or []:
                    S = host_trace.stage_mask(sg.get("stage"))
                    if S is not None and "ALL_COMMANDS" not in S:
                        for cb in b.get("command_buffers") or []:
                            out.add((r["seq"], cb))
        return out

    def _pin(self):
        staged = self._staged_signal_batches()
        for c in self.calls:
            if c.pin:
                continue
            ordn = c.rec_ord
            if self.global_pin:
                c.pin = self.global_pin
            elif ordn in self.bad_recordings:
                c.pin = "its command buffer: " + self.bad_recordings[ordn]
            elif c.occurrences == 0:
                c.pin = "it is never submitted in the trace (no evidence)"
            elif any((sub, c.cb) in staged for sub in c.submits):
                c.pin = ("it is submitted in a batch that signals a semaphore with a stage mask narrower than "
                         "ALL_COMMANDS, which the oracle does not model")
            for e in c.elems:
                r = e.raw
                ss, sa, ds, da = e.masks()
                sf, df = r.get("src_family"), r.get("dst_family")
                if any(isinstance(x, int) for k in ("src_stage", "src_access", "dst_stage", "dst_access")
                       for x in (r.get(k) or [])) or None in (ss, ds):
                    e.pin = "its masks contain bits the model cannot name"
                elif sf is not None and df is not None and sf != df:
                    e.pin = "it is a queue family ownership transfer"
                elif "HOST" in (ss or ()) or "HOST" in (ds or ()) or \
                        {"HOST_READ", "HOST_WRITE"} & (set(sa or ()) | set(da or ())):
                    e.pin = "it synchronizes with the host, which the trace does not observe"
                if e.kind == "I" and r.get("old_layout") != r.get("new_layout"):
                    e.transition = True

    # -- building the frame the oracle sees -------------------------------
    def _elem_masks(self, call, e, action):
        """The masks a (possibly narrowed) element has in the frame."""
        if action is None or action == "keep":
            return e.masks()
        return action[1]

    def build(self, plan):
        """plan: {call key: {elem label: "keep" | "remove" | ("narrow", (ss, sa, ds, da))}}
        -> (strengthened Frame, positions {(call key, elem label): [(queue, index)]})."""
        fr = self.frame
        queues, pos = {}, {}
        self._src_elem = {}
        # elision premise: a stable identity for every event of the
        # built frame, the same across plans, so "no new hazard" can compare racing event
        # pairs instead of diagnostic keys (keys repeat when a command buffer is submitted
        # more than once, so a new pair could hide behind an old report with the same key)
        self._ident = {}
        ident = self._ident

        def reg(obj, key):
            ident[id(obj)] = key
            return obj
        # load/store injection points
        first_draw, last_draw = {}, {}
        for inst in self.instances:
            first_draw[inst["draws"][0]] = inst
            last_draw[inst["draws"][-1]] = inst
        for qn, ops in fr.queues.items():
            out = []
            i = 0
            n = len(ops)
            while i < n:
                op = ops[i]
                if isinstance(op, Barrier):
                    ce = self.op_elem.get(id(op))
                    if ce is not None:
                        call, e = ce
                        act = plan.get(call.key, {}).get(e.label)
                        if act == "remove":
                            i += 1
                            continue
                        if isinstance(act, tuple):
                            ss, sa, ds, da = act[1]
                            op = Barrier(op.res, op.old, op.new, op.where, ss, sa, ds, da,
                                         call=op.call, struct=op.struct, region=op.region,
                                         fb_local=op.fb_local, fb=op.fb, auto=op.auto)
                        pos.setdefault((call.key, e.label), []).append((qn, len(out)))
                    src = ops[i]
                    if op.new is not None:
                        # B1 ; transition ; B2 are one struct of one call (timing._same_call)
                        out.append(reg(Barrier(op.res, op.old, op.new, op.where, op.src_stages, op.src_access,
                                               frozenset({LT_STAGE}), frozenset({LT_FLAG}),
                                               call=op.call, struct=op.struct, region=op.region,
                                               event=op.event, fb_local=op.fb_local, fb=op.fb, auto=op.auto),
                                       ("b1", id(src))))  # an event wait bounds B1's first scope
                        if self.op_elem.get(id(op)):
                            self._src_elem[id(out[-1])] = self.op_elem[id(op)]
                        out.append(reg(Use(op.res, "write", None, op.where + " (layout transition)", LT_STAGE,
                                           LT_FLAG, call=op.call, struct=op.struct, region=op.region, fb=op.fb),
                                       ("lt", id(src))))
                        out.append(reg(Barrier(op.res, None, None, op.where + " (after layout transition)",
                                               frozenset({LT_STAGE}), frozenset({LT_FLAG}), op.dst_stages,
                                               op.dst_access, call=op.call, struct=op.struct, region=op.region,
                                               fb_local=op.fb_local),
                                       ("b2", id(src))))
                    else:
                        out.append(reg(op, ("op", id(src))))
                        if self.op_elem.get(id(ops[i])):
                            self._src_elem[id(op)] = self.op_elem[id(ops[i])]
                    i += 1
                    continue
                if isinstance(op, Use):
                    m = _CMD_WHERE.match(op.where)
                    dseq = int(m.group(2)) if m and m.group(1).startswith("vkCmdDraw") else None
                    # load/store operations are Uses of host_trace's frame already
                    # (vkCmdBeginRendering / vkCmdEndRendering); _load_store is kept
                    # for frames built without them
                    if self.inject_load_store and dseq in first_draw and \
                            (i == 0 or not self._same_cmd(ops[i - 1], op)):
                        out.extend(reg(x, ("ls", id(op), True, k)) for k, x in
                                   enumerate(self._load_store(first_draw[dseq], ops, i, load=True)))
                    k0 = len(out)
                    self._emit_use(out, op)
                    for k in range(k0, len(out)):
                        reg(out[k], ("use", id(op), k - k0))
                    if self.inject_load_store and dseq in last_draw and \
                            (i + 1 >= n or not self._same_cmd(ops[i + 1], op)):
                        out.extend(reg(x, ("ls", id(op), False, k)) for k, x in
                                   enumerate(self._load_store(last_draw[dseq], ops, i, load=False)))
                    i += 1
                    continue
                if (self.acquire_read and qn.startswith("present:") and isinstance(op, timing.Signal)
                        and op.where.startswith("vkAcquireNextImageKHR")
                        and not (out and (out[-1].where == op.where
                                          or out[-1].where.endswith(" until " + op.where)))):
                    # strengthening 4: the image is the presentation engine's until this signal
                    out.append(reg(Use(qn[len("present:"):], "read", None,
                                       f"presentation engine releases {qn[len('present:'):]} at {op.where}"),
                                   ("acq-read", id(op))))
                out.append(reg(op, ("op", id(op))))
                i += 1
            queues[qn] = out
        f2 = Frame(queues=queues, initial_layouts=dict(fr.initial_layouts), swapchains=dict(fr.swapchains),
                   semaphore_reuse=list(fr.semaphore_reuse), findings=list(fr.findings))
        return f2, pos

    @staticmethod
    def _same_cmd(a, b):
        if not isinstance(a, Use):
            return False
        ma, mb = _CMD_WHERE.match(a.where), _CMD_WHERE.match(b.where)
        return bool(ma and mb and ma.group(2) == mb.group(2) and ma.group(3) == mb.group(3))

    def _emit_use(self, out, op):
        if op.access != "write" or not self.waw_proxy:
            out.append(op)
            return
        scope = op.scope if op.scope is not None else object()
        out.append(dataclasses.replace(op, access="read", where=op.where + " [the write needs visibility]",
                                       scope=scope))
        out.append(dataclasses.replace(op, scope=scope))

    def _load_store(self, inst, ops, i, load):
        """Uses for the loadOp (load=True) / storeOp of every attachment of a
        rendering instance, in the scope of its draws."""
        m = _CMD_WHERE.match(ops[i].where)
        sub = m.group(3) if m else "?"
        scope = None
        # the draws' attachment uses carry the instance's scope
        j = i
        while j < len(ops) and isinstance(ops[j], Use) and self._same_cmd(ops[j], ops[i]):
            if ops[j].scope is not None:
                scope = ops[j].scope
                break
            j += 1
        j = i
        while scope is None and j >= 0 and isinstance(ops[j], Use) and self._same_cmd(ops[j], ops[i]):
            scope = ops[j].scope
            j -= 1
        scope = scope or object()
        out = []
        fn = "vkCmdBeginRendering" if load else "vkCmdEndRendering"
        for role, a in inst["atts"]:
            op_name = a.get("load" if load else "store") or ""
            color = role == "color"
            if load:
                if op_name.endswith("LOAD_OP_LOAD"):
                    acc = "read"
                elif op_name.endswith("LOAD_OP_NONE_KHR") or op_name.endswith("LOAD_OP_NONE_EXT") or op_name.endswith("LOAD_OP_NONE"):
                    continue
                else:
                    acc = "write"
                stage = "COLOR_ATTACHMENT_OUTPUT" if color else "EARLY_FRAGMENT_TESTS"
            else:
                if "STORE_OP_NONE" in op_name:
                    continue
                acc = "write"
                stage = "COLOR_ATTACHMENT_OUTPUT" if color else "LATE_FRAGMENT_TESTS"
            flag = ("COLOR_ATTACHMENT_" if color else "DEPTH_STENCIL_ATTACHMENT_") + ("READ" if acc == "read" else "WRITE")
            lay = host_trace.short_layout(a.get("layout"))
            where = f"{fn} ({'load' if load else 'store'} op of {role} attachment) for draw @seq {m.group(2) if m else '?'}, submitted @seq {sub}"
            for res in a.get("_res") or []:
                u = Use(res, acc, lay, where, stage, flag, scope)
                if acc == "write":
                    self._emit_use(out, u)
                else:
                    out.append(u)
        return out

    # -- the oracle ----------------------------------------------------------
    def hazards(self, plan, exhaustive=False):
        self.checks += 1
        f2, pos = self.build(plan)
        hz = []
        rep = timing.check(f2, Report(), hazards=hz, exhaustive=exhaustive, window=False)
        self.transition_reach(f2, rep)
        self.last_pairs = self.event_pairs(f2, hz)
        return {self.hkey(d): d for d in rep.items}, f2, pos

    def event_pairs(self, f2, hz):
        """{frozenset({identity, identity})} of the racing pairs the checker found on a
        built frame, by event identity (stable across plans; see build). The checker's
        (queue, position) keys index the queue after loop expansion; built frames hold no
        loops, so they index f2.queues directly."""
        assert not any(isinstance(op, timing.Loop) for ops in f2.queues.values() for op in ops)
        out = set()
        for a, b, _ in hz:
            ia = self._ident[id(f2.queues[a[0]][a[1] - 1])]
            ib = self._ident[id(f2.queues[b[0]][b[1] - 1])]
            out.add(frozenset((ia, ib)))
        return out

    @staticmethod
    def hkey(d):
        """Identity of a diagnostic for "no new hazard": code, location, text,
        and the earlier access it names, so that a change
        giving an already-racing access a NEW racing partner counts as new."""
        earlier = next((n for n in d.notes if str(n).startswith("earlier access")), None)
        return (d.code, d.where, d.message, earlier)

    def _taint(self, f2):
        """Pin every barrier element that lies between the two
        accesses of a hazard the program ALREADY has (the baseline). "No new
        hazard" cannot see a change that makes such a pair worse: the checker
        reports the pair once, with the same text, whether the barrier between
        them lacks one access bit or all of them (seen on Sascha's computenbody:
        dropping SHADER_READ from a barrier whose WAW is already reported).
        Same queue: the barriers between the earlier and the later access.
        Otherwise (another queue, the host): every barrier of the recordings
        that hold the accesses."""
        index = {}
        for qn, ops in f2.queues.items():
            for i, op in enumerate(ops):
                m = _ANY_WHERE.match(getattr(op, "where", "") or "")
                if m:
                    index.setdefault((int(m.group(1)), int(m.group(2))), []).append((qn, i))
        tainted = {}

        def mark(call, e, why):
            if not e.pin:
                tainted.setdefault((call.key, e.label), why)
                e.pin = why
        for d in self.base_items:
            ml = _ANY_WHERE.match(d.where or "")
            note = next((n for n in d.notes if str(n).startswith("earlier access")), "")
            me = _ANY_WHERE.match(note[len("earlier access: "):]) if note else None
            why = (f"a hazard the program already has ({d.code} at {d.where[:80]}) involves the accesses "
                   "this barrier orders; 'no new hazard' cannot show a change keeps that pair no worse")
            L = index.get((int(ml.group(1)), int(ml.group(2)))) if ml else None
            E = index.get((int(me.group(1)), int(me.group(2)))) if me else None
            same = L and E and {q for q, _ in L} & {q for q, _ in E}
            if same:
                q = sorted(same)[0]
                lo = min(i for qq, i in E if qq == q)
                hi = max(i for qq, i in L if qq == q)
                for op in f2.queues[q][min(lo, hi):max(lo, hi) + 1]:
                    ce = self.op_elem.get(id(op)) if isinstance(op, Barrier) else None
                    if ce is None and isinstance(op, Barrier):
                        ce = self._src_elem.get(id(op))
                    if ce:
                        mark(ce[0], ce[1], why)
            else:
                recs = {self.recording_of_seq.get(int(m.group(1))) for m in (ml, me) if m}
                for call in self.calls:
                    if call.rec_ord in recs:
                        for e in call.elems:
                            mark(call, e, why)
        return tainted

    @staticmethod
    def transition_reach(f2, rep):
        """Obligation 1b. A layout transition must reach every later access of
        its image through its OWN dst scope, directly or by chaining (later
        barriers whose src scope meets an already-reached dst scope). The
        oracle would also accept a later ALL_COMMANDS barrier capturing the
        transition directly; syncval and the spec's barrier chaining do not,
        so this check closes that gap. Scans to the next transition of the
        image or the next queue wait (a new submission or semaphore wait)."""
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
                        reached = any(not timing._same_call(c, op) and
                                      timing._stages_meet(c.dst_stages, op.src_stages) for c in captured)
                        if op.res == res and op.dst_stages == frozenset({LT_STAGE}):
                            if not reached:
                                rep.add("timing", "elide.transition-unreached",
                                        f"the layout transition of '{res}' is not ordered before the next "
                                        "transition of it by a chain from its own barrier.", op.where,
                                        [f"transition: {b2.where}"])
                            break
                        if reached:
                            captured.append(op)
                        continue
                    if isinstance(op, Use) and op.res == res and op.stage != LT_STAGE:
                        ok = any(timing._stage_in(op.stage, c.dst_stages, False) and
                                 timing._explicit(op.stage, c.dst_stages) and
                                 timing._access_in(op.flag, op.access, c.dst_access) for c in captured)
                        if not ok:
                            rep.add("timing", "elide.transition-unreached",
                                    f"the layout transition of '{res}' does not reach this "
                                    f"{op.access} ({op.stage or 'any stage'}, {op.flag or 'any access'}) "
                                    "through its own dst scope or a barrier chained from it.", op.where,
                                    [f"transition: {b2.where}"])

    def new_hazards(self, plan):
        h, f2, pos = self.hazards(plan)
        return [d for k, d in h.items() if k not in self.base], f2, pos

    # -- narrowing candidates ---------------------------------------------------
    def candidates(self, call, e, f2, pos, plan):
        """[(label, (ss, sa, ds, da))] narrower than the element's current
        masks, narrowest first."""
        cur = self._elem_masks(call, e, plan.get(call.key, {}).get(e.label))
        oss, osa, ods, oda = e.masks()
        if None in (oss, osa, ods, oda):
            return []
        global_res = e.kind in ("M", "E")
        res = set(e.res)
        out = []
        for mode in ("local", "wide"):
            src_st, src_ac, dst_st, dst_ac = set(), set(), set(), set()
            src_any = dst_any = False
            for qn, k in pos.get((call.key, e.label), []):
                ops = f2.queues[qn]
                # the element's own ops: one Barrier, or B1 / LT / B2 for a transition
                lo = k
                hi = k + 2 if (isinstance(ops[k], Barrier) and ops[k].dst_stages == frozenset({LT_STAGE})) else k
                j = lo - 1
                while j >= 0:
                    op = ops[j]
                    if mode == "local" and (isinstance(op, Wait) or
                                            (isinstance(op, Barrier) and (global_res or op.res is None or op.res in res))):
                        break
                    if isinstance(op, Use) and (global_res or op.res in res) and self._gpu_use(op):
                        if op.stage is None:
                            src_any = True
                        else:
                            src_st.add(op.stage)
                        if op.access == "write":
                            if op.flag is None:
                                src_any = True
                            else:
                                src_ac.add(op.flag)
                    j -= 1
                j = hi + 1
                while j < len(ops):
                    op = ops[j]
                    if e.transition:
                        # a transition must reach its later users itself (obligation 1b)
                        if isinstance(op, Wait) or (isinstance(op, Barrier) and op.res in res and
                                                    op.dst_stages == frozenset({LT_STAGE})):
                            break
                    elif mode == "local" and isinstance(op, Barrier) and (global_res or op.res is None or op.res in res):
                        break
                    if isinstance(op, Use) and (global_res or op.res in res) and self._gpu_use(op):
                        if op.stage is None or op.flag is None:
                            dst_any = True
                        else:
                            dst_st.add(op.stage)
                            dst_ac.add(op.flag)
                    j += 1
            ss = oss if src_any else self._stages(src_st, oss, True)
            sa = osa if src_any else self._access(src_ac, osa, "write")
            ds = ods if dst_any else self._stages(dst_st, ods, False)
            da = oda if dst_any else self._access(dst_ac, oda, None)
            cand = (ss, sa, ds, da)
            if cand != tuple(cur) and self._narrower(cand, cur) and cand not in [c for _, c in out]:
                out.append((mode, cand))
        return out

    @staticmethod
    def _gpu_use(op):
        """Accesses that shape a barrier's masks: not the layout-transition
        pseudo-write, and not descriptor-set objects (written only by the host;
        their device reads are ordered by submission, never by a barrier)."""
        return op.stage != LT_STAGE and not str(op.res).startswith("set:") and op.flag != "DESCRIPTOR_READ"

    @staticmethod
    def _stages(found, orig, src):
        s = set()
        for st in found:
            for x in (st, STAGE_PAIRS.get(st)):
                if x and timing._stage_in(x, orig, src):
                    s.add(x)
        return frozenset(s)

    @staticmethod
    def _access(found, orig, kind):
        s = set()
        for f in found:
            for x in (f, WRITE_IMPLIES_READ.get(f)):
                if not x:
                    continue
                k = kind or ("write" if x.endswith("WRITE") else "read")
                if timing._access_in(x, k, orig):
                    s.add(ACCESS_LEGACY.get(x, x))
        return frozenset(s)

    def _narrower(self, cand, cur):
        """Is cand a strict, bitwise narrowing of cur? (compared as Vulkan bits)"""
        n = self.names
        cb = (n.stage_bits(cand[0]), n.access_bits(cand[1]), n.stage_bits(cand[2]), n.access_bits(cand[3]))
        ub = (self._bits_any(cur[0], "stage"), self._bits_any(cur[1], "access"),
              self._bits_any(cur[2], "stage"), self._bits_any(cur[3], "access"))
        return cb != ub

    def _bits_any(self, s, kind):
        if s is None:
            return -1
        return self.names.stage_bits(s) if kind == "stage" else self.names.access_bits(s)

    # -- sync1: one stage mask per call --------------------------------------
    def normalize(self, call, acts):
        """For a sync1 call, every kept struct gets the union of the stage
        masks of the call's kept structs (sync1 has one stage mask per call).
        A "narrowing" that orders everything the recorded masks order (equal
        Vulkan bits, or logically equivalent / wider scopes: e.g. src
        {COLOR_ATTACHMENT_OUTPUT, FRAGMENT_SHADER} for COLOR_ATTACHMENT_OUTPUT)
        is dropped, i.e. the struct is kept as recorded."""
        return self._drop_noop(call, self._normalize1(call, acts))

    def _drop_noop(self, call, acts):
        out = dict(acts)
        for e in call.elems:
            a = out.get(e.label)
            if isinstance(a, tuple) and not self._orders_less(e.masks(), a[1]):
                del out[e.label]
        return out

    def _orders_less(self, rec, new):
        """Does `new` order less than `rec` in some scope? (False = no change on hardware.)"""
        kinds = ("stage", "access", "stage", "access")
        if any(x is None for x in rec) or any(x is None for x in new):
            return not all(self._bits_any(x, k) == self._bits_any(y, k) for x, y, k in zip(rec, new, kinds))
        covers = lambda n, r: n[0] or (not r[0] and r[1] <= n[1])
        (rs, ra, rd, rda), (ns, na, nd, nda) = rec, new
        if not (covers(timing._es(frozenset(ns), True), timing._es(frozenset(rs), True))
                and covers(timing._es(frozenset(nd), False), timing._es(frozenset(rd), False))
                and covers(timing._as(frozenset(ns)), timing._as(frozenset(rs)))
                and covers(timing._as(frozenset(nd)), timing._as(frozenset(rd)))):
            return True
        for r, n in ((ra, na), (rda, nda)):
            rb, nb = self._bits_any(r, "access"), self._bits_any(n, "access")
            if rb & nb != rb:
                return True
        return False

    def _normalize1(self, call, acts):
        if not call.sync1:
            return acts
        kept = [e for e in call.elems if acts.get(e.label) != "remove"]
        if not kept:
            return acts
        ss, ds = set(), set()
        for e in kept:
            m = self._elem_masks(call, e, acts.get(e.label))
            ss |= set(m[0]) if m[0] is not None else {"ALL_COMMANDS"}
            ds |= set(m[2]) if m[2] is not None else {"ALL_COMMANDS"}
        ss = frozenset(ss) or frozenset({"TOP_OF_PIPE"})
        ds = frozenset(ds) or frozenset({"BOTTOM_OF_PIPE"})
        out = dict(acts)
        for e in kept:
            m = self._elem_masks(call, e, acts.get(e.label))
            if (ss, ds) != (m[0], m[2]) or isinstance(acts.get(e.label), tuple):
                out[e.label] = ("narrow", (ss, m[1], ds, m[3]))
        return out

    # -- the greedy search ---------------------------------------------------
    def run(self):
        t0 = time.time()
        plan = {}
        h, f2, pos = self.hazards(plan)
        self.base = set(h)
        self.base_items = list(h.values())
        self.log(f"baseline: {len(self.base)} diagnostic(s) on the strengthened frame")
        self.tainted = self._taint(f2) if self.taint_baseline_hazards else {}
        if self.tainted:
            self.log(f"{len(self.tainted)} barrier struct(s) pinned: they order accesses of a baseline hazard")
        self.decisions = OrderedDict()
        # pass 1: removal, in recording order (each step re-checked on top of
        # every removal accepted before it)
        for call in self.calls:
            d = {"why": {}, "witness": None}
            self.decisions[call.key] = d
            if call.pin:
                d["why"]["*"] = "pinned: " + call.pin
                continue
            free = [e for e in call.elems if not e.pin and not e.transition]
            if free and len(free) == len(call.elems):
                trial = dict(plan)
                trial[call.key] = {e.label: "remove" for e in call.elems}
                bad, _, _ = self.new_hazards(trial)
                if not bad:
                    plan = trial
                    d["why"]["*"] = "removed: no new hazard with it gone"
                    continue
                d["witness"] = bad[0]
            for e in call.elems:
                if e.pin:
                    d["why"][e.label] = "kept: " + e.pin
                    continue
                if e.transition or len(call.elems) == 1:
                    continue
                acts = dict(plan.get(call.key, {}))
                acts[e.label] = "remove"
                trial = dict(plan)
                trial[call.key] = self.normalize(call, acts)
                bad, _, _ = self.new_hazards(trial)
                if not bad:
                    plan = trial
                    d["why"][e.label] = "removed: no new hazard with it gone"
                else:
                    d["witness"] = d["witness"] or bad[0]
        # pass 2: narrow what is left, on the frame without the removed barriers
        for call in self.calls:
            d = self.decisions[call.key]
            if call.pin:
                continue
            for e in call.elems:
                if e.pin or plan.get(call.key, {}).get(e.label) == "remove":
                    continue
                f2, pos = self.build(plan)
                done = False
                for mode, cand in self.candidates(call, e, f2, pos, plan):
                    acts = dict(plan.get(call.key, {}))
                    acts[e.label] = ("narrow", cand)
                    trial = dict(plan)
                    trial[call.key] = self.normalize(call, acts)
                    if trial[call.key] == plan.get(call.key, {}):
                        continue                    # no change after sync1 normalization
                    bad, _, _ = self.new_hazards(trial)
                    if not bad:
                        plan = trial
                        d["why"][e.label] = f"narrowed ({mode} accesses): no new hazard"
                        done = True
                        break
                if not done:
                    d["why"][e.label] = ("kept: layout transition, no narrower mask verified" if e.transition
                                         else "kept: needed, no narrower mask verified")
        bad, _, _ = self.new_hazards(plan)
        assert not bad, f"final plan re-check failed: {bad[0]}"
        # re-check once more in pairwise (exhaustive) mode,
        # where every racing pair is reported, not one representative per epoch
        self.pairwise = None
        if self.pairwise_recheck:
            self.pairwise = self._pairwise_verify(plan)
            assert not self.pairwise["new"] and not self.pairwise["new_pairs"], \
                f"final pairwise re-check failed: {self.pairwise}"
        self.plan = plan
        # sanity: the unstrengthened frame (plain timing.check) sees nothing new either
        self.raw_check = self._raw_check(plan)
        self.chain_plan, self.chain_note = self._chain_merge(plan)
        self.seconds = time.time() - t0
        return plan

    def _pairwise_verify(self, plan):
        """The elision theorem's premise, Haz_pw(F*(tau_P)) <= Haz_pw(F*(tau)), compared by event
        identity, plus the diagnostic-key comparison."""
        if getattr(self, "_pw_base", None) is None:
            hb, _, _ = self.hazards({}, exhaustive=True)
            self._pw_base = (hb, self.last_pairs)
        hb, pb = self._pw_base
        hp, _, _ = self.hazards(plan, exhaustive=True)
        pp = self.last_pairs
        new = [d for k, d in hp.items() if k not in hb]
        new_pairs = pp - pb
        return {"baseline": len(hb), "with_plan": len(hp), "new": len(new),
                "baseline_pairs": len(pb), "with_plan_pairs": len(pp), "new_pairs": len(new_pairs),
                "example": (new[0].where if new else sorted(map(sorted, new_pairs), key=str)[0] if new_pairs
                            else None)}

    def _raw_check(self, plan):
        def apply(fr):
            q2 = {}
            for qn, ops in fr.queues.items():
                out = []
                for op in ops:
                    ce = self.op_elem.get(id(op)) if isinstance(op, Barrier) else None
                    if ce:
                        act = plan.get(ce[0].key, {}).get(ce[1].label)
                        if act == "remove":
                            continue
                        if isinstance(act, tuple):
                            op = Barrier(op.res, op.old, op.new, op.where, *act[1], call=op.call, struct=op.struct,
                                         region=op.region, fb_local=op.fb_local, fb=op.fb, auto=op.auto)
                    out.append(op)
                q2[qn] = out
            return Frame(queues=q2, initial_layouts=dict(fr.initial_layouts), swapchains=dict(fr.swapchains),
                         semaphore_reuse=list(fr.semaphore_reuse), findings=list(fr.findings))
        before = {(d.code, d.where, d.message) for d in timing.check(self.frame, Report()).items}
        after = {(d.code, d.where, d.message) for d in timing.check(apply(self.frame), Report()).items}
        return {"baseline": len(before), "with_plan": len(after), "new": len(after - before)}

    # -- chain-keyed merge ------------------------------------------------------
    def _chain_merge(self, plan):
        groups = OrderedDict()
        for c in self.calls:
            groups.setdefault(c.chain, []).append(c)

        def merged_for(group, unanimous):
            acts = [plan.get(c.key, {}) for c in group]
            c0 = group[0]
            out = {}
            if unanimous:
                ref = [self._act_repr(c0, e, acts[0].get(e.label)) for e in c0.elems]
                for c, a in zip(group[1:], acts[1:]):
                    if [self._act_repr(c, e, a.get(e.label)) for e in c.elems] != ref:
                        return None
                return dict(acts[0])
            for e in c0.elems:
                per = [a.get(e.label) for a in acts]
                if any(p is None or p == "keep" for p in per):
                    continue                            # some instance keeps it
                if all(p == "remove" for p in per):
                    out[e.label] = "remove"
                    continue
                u = [set(), set(), set(), set()]
                for p in per:
                    if isinstance(p, tuple):
                        for i in range(4):
                            u[i] |= set(p[1][i])
                out[e.label] = ("narrow", tuple(frozenset(x) for x in u))
            return self.normalize(c0, out)

        for unanimous in (False, True):
            cplan, chain_acts = {}, {}
            ok = True
            for chain, group in groups.items():
                m = merged_for(group, unanimous)
                if m is None:
                    continue
                chain_acts[chain] = m
                for c in group:
                    if m:
                        cplan[c.key] = m
            bad, _, _ = self.new_hazards(cplan)
            if not bad and self.pairwise_recheck:
                # an empty epoch result does not imply an empty pairwise one (e.g. with
                # non-uniform access regions), so the chain plan is
                # re-verified in pairwise mode by event identity, like the strict plan
                pw = self._pairwise_verify(cplan)
                if pw["new"] or pw["new_pairs"]:
                    self.log(f"chain merge ({'unanimous' if unanimous else 'union'}) failed the pairwise "
                             f"re-check: {pw}")
                    continue
            if not bad:
                note = ("merged conservatively over all instances of each key; re-verified on the whole trace"
                        + (" (epoch and pairwise, by event identity)" if self.pairwise_recheck else " (epoch)")
                        if not unanimous else "only keys whose instances all agree; re-verified on the whole trace"
                        + (" (epoch and pairwise, by event identity)" if self.pairwise_recheck else " (epoch)"))
                return chain_acts, note
            self.log(f"chain merge ({'unanimous' if unanimous else 'union'}) failed re-check: {bad[0].code} at {bad[0].where}")
        return {}, "no chain-keyed plan verified; use strict matching"

    def _act_repr(self, call, e, a):
        if a is None or a == "keep":
            return "keep"
        if a == "remove":
            return "remove"
        return ("narrow",) + tuple(tuple(sorted(x)) for x in a[1])

    # -- output -------------------------------------------------------------------
    def _elem_json(self, call, e, act):
        n = self.names
        r = e.raw
        out = OrderedDict(kind={"M": "memory", "B": "buffer", "I": "image", "E": "exec"}[e.kind], index=e.i)
        if e.kind == "I":
            out["image"] = self.model.name(r["image"])
            out["layout"] = f"{host_trace.short_layout(r['old_layout'])} -> {host_trace.short_layout(r['new_layout'])}"
        elif e.kind == "B":
            out["buffer"] = self.model.name(r["buffer"])
        oss, osa, ods, oda = e.masks()
        out["recorded"] = {"src_stage": sorted(oss or []), "src_access": sorted(osa or []),
                           "dst_stage": sorted(ods or []), "dst_access": sorted(oda or [])}
        if act is None or act == "keep":
            out["action"] = "keep"
        elif act == "remove":
            out["action"] = "remove"
        else:
            ss, sa, ds, da = act[1]
            out["action"] = "narrow"
            out["new"] = {"src_stage": sorted(ss), "src_access": sorted(sa), "dst_stage": sorted(ds), "dst_access": sorted(da)}
            out["src_stage_mask"] = f"0x{n.stage_bits(ss):x}"
            out["src_access_mask"] = f"0x{n.access_bits(sa):x}"
            out["dst_stage_mask"] = f"0x{n.stage_bits(ds):x}"
            out["dst_access_mask"] = f"0x{n.access_bits(da):x}"
        return out

    def _call_json(self, call, acts, why=None, witness=None):
        n = self.names
        elems = [self._elem_json(call, e, acts.get(e.label)) for e in call.elems]
        kinds = [x["action"] for x in elems]
        if kinds and all(k == "remove" for k in kinds):
            decision = "remove"
        elif all(k == "keep" for k in kinds):
            decision = "required"
        else:
            decision = "narrow"
        out = OrderedDict(rec=call.rec_ord, idx=call.idx, seq=call.seq, fn=call.fn, sync=1 if call.sync1 else 2,
                          cb=self.model.name(call.cb), sig=call.sig, chain=call.chain, decision=decision)
        if call.sync1 and decision == "narrow":
            kept = [e for e in call.elems if acts.get(e.label) != "remove"]
            m = self._elem_masks(call, kept[0], acts.get(kept[0].label))
            out["sync1_src_stage_mask"] = f"0x{n.stage_bits(m[0]):x}"
            out["sync1_dst_stage_mask"] = f"0x{n.stage_bits(m[2]):x}"
        out["elements"] = elems
        if why is not None:
            out["why"] = why
        if witness is not None:
            out["not_removable_because"] = {"code": witness.code, "at": witness.where, "message": witness.message}
        return out

    def plan_json(self):
        barriers = []
        for c in self.calls:
            d = self.decisions[c.key]
            acts = self.plan.get(c.key, {})
            removed = bool(c.elems) and all(acts.get(e.label) == "remove" for e in c.elems)
            j = self._call_json(c, acts, d["why"], None if removed else d["witness"])
            j["occurrences"] = c.occurrences
            barriers.append(j)
        chain = []
        seen = set()
        for c in self.calls:
            if c.chain in self.chain_plan and c.chain not in seen:
                seen.add(c.chain)
                j = self._call_json(c, self.chain_plan[c.chain])
                for k in ("rec", "idx", "seq", "cb"):
                    j.pop(k, None)
                j["instances"] = sum(1 for x in self.calls if x.chain == c.chain)
                if j["decision"] != "required":
                    chain.append(j)
        summ = OrderedDict()
        for b in barriers:
            summ[b["decision"]] = summ.get(b["decision"], 0) + 1
        structs = OrderedDict(total=0, removed=0, narrowed=0, kept=0)
        for b in barriers:
            for e in b["elements"]:
                structs["total"] += 1
                structs[{"remove": "removed", "narrow": "narrowed", "keep": "kept"}[e["action"]]] += 1
        with open(self.trace, "rb") as f:
            sha = hashlib.sha1(f.read()).hexdigest()
        return OrderedDict([
            ("format", "covenant-elide-plan/1"),
            ("trace", os.path.basename(self.trace)),
            ("trace_sha1", sha),
            ("id_scheme", OrderedDict(
                strict="(rec, idx): rec = 0-based vkBeginCommandBuffer ordinal in the process; idx = 0-based "
                       "barrier-call index within that recording (sync1, sync2 and 2KHR all count)",
                chain="FNV-1a 64 over the signatures of the recording's barrier calls so far (state continues "
                      "from the previous barrier's chain value; recording start = 0xcbf29ce484222325)",
                sig="FNV-1a 64 of the canonical content string; the layer never touches a call whose sig differs",
                sig_format="<1|2>|<depFlags> then |M:ss:sa:ds:da, |B:ss:sa:ds:da:sfam:dfam:off:size, "
                           "|I:ss:sa:ds:da:sfam:dfam:old:new:aspect:baseMip:levels:baseLayer:layers, |E:ss:ds; decimal")),
            ("proof", OrderedDict(
                oracle="covenant.timing.check (vector-clock happens-before over all queues; every schedule the "
                       "recorded synchronization allows)",
                obligation="the planned frame reports no diagnostic the recorded frame does not; layout transitions "
                           "and ownership transfers are never removed",
                strengthening=["layout transitions modelled as writes between the src and dst scopes",
                               "a layout transition must reach later accesses through its own dst scope "
                               "(or a chain from it), not by a later barrier capturing it",
                               "write-after-write requires the later write in the dst access scope",
                               "attachment load/store operations modelled as accesses",
                               "the presentation engine reads a swapchain image before every acquire signal"],
                baseline_diagnostics=len(self.base),
                baseline=[{"code": d.code, "at": d.where} for d in self.base_items][:20],
                new_diagnostics_with_plan=0,
                plain_timing_check=self.raw_check,
                pairwise_recheck=self.pairwise,
                pinned_for_baseline_hazards=len(getattr(self, "tainted", {}) or {}),
                oracle_runs=self.checks,
                oracle_source_sha1=_oracle_sha1(),
                seconds=round(self.seconds, 2))),
            ("summary", OrderedDict(calls=len(barriers), decisions=summ, structs=structs,
                                    chain_entries=len(chain), chain_note=self.chain_note)),
            ("barriers", barriers),
            ("chain", chain),
        ])


def _oracle_sha1():
    """Version of the oracle the plan was proved with (timing.py + host_trace.py + this file)."""
    h = hashlib.sha1()
    for mod in (timing, host_trace, sys.modules[__name__]):
        with open(mod.__file__, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def naive_control(el):
    """UNSAFE control experiment: test each removable barrier call ALONE
    against the recorded frame and remove every one that passes, all together,
    without re-verifying the combination. This is what a per-barrier
    redundancy analysis would do. Returns (plan, number of hazards the union
    creates); the plan is written with "unsafe_control": true."""
    el.base = set(el.hazards({})[0])
    alone = {}
    for c in el.calls:
        if c.pin or any(e.pin or e.transition for e in c.elems):
            continue
        trial = {c.key: {e.label: "remove" for e in c.elems}}
        bad, _, _ = el.new_hazards(trial)
        if not bad:
            alone[c.key] = trial[c.key]
    bad, _, _ = el.new_hazards(alone)
    return alone, bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("trace")
    ap.add_argument("--spirv-dir")
    ap.add_argument("-o", "--out")
    ap.add_argument("--header", help="vulkan_core.h (default: /usr/include/vulkan/vulkan_core.h)")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--naive-control", metavar="OUT",
                    help="also write an UNSAFE plan that removes every barrier that is redundant on its own "
                         "(no joint re-check), as a control experiment")
    a = ap.parse_args(argv)
    log = (lambda *x: None) if a.quiet else (lambda *x: print(*x, file=sys.stderr))
    el = Elider(a.trace, a.spirv_dir, a.header, log)
    el.run()
    if a.naive_control:
        good_plan, good_base = el.plan, el.base
        nplan, bad = naive_control(Elider(a.trace, a.spirv_dir, a.header, log))
        el.plan = nplan
        nj = el.plan_json()
        el.plan, el.base = good_plan, good_base
        nj["unsafe_control"] = True
        nj["chain"] = []
        nj["proof"] = OrderedDict(note="NOT VERIFIED: each removed barrier was redundant on its own; the "
                                       "combination was not re-checked", hazards_created=len(bad),
                                  examples=[{"code": d.code, "at": d.where} for d in bad[:10]])
        with open(a.naive_control, "w") as f:
            json.dump(nj, f, indent=1)
        log(f"naive control: removes {len(nplan)} calls together; the checker finds {len(bad)} hazard(s) in the union")
    pj = el.plan_json()
    if a.out:
        with open(a.out, "w") as f:
            json.dump(pj, f, indent=1)
    if not a.quiet:
        print(format_summary(pj))
    return 0


def format_summary(pj):
    s = pj["summary"]
    lines = [f"{pj['trace']}: {s['calls']} barrier calls -> {dict(s['decisions'])}; structs {dict(s['structs'])}; "
             f"chain entries {s['chain_entries']} ({s['chain_note']}); oracle runs {pj['proof']['oracle_runs']}, "
             f"{pj['proof']['seconds']} s"]
    for b in pj["barriers"]:
        el = ", ".join(f"{e['kind']}[{e['index']}]{':' + e['image'] if 'image' in e else ''} {e['action']}"
                       + (f" {'/'.join(e['new']['src_stage']) or 'NONE'}->{'/'.join(e['new']['dst_stage']) or 'NONE'}"
                          if e["action"] == "narrow" else "")
                       for e in b["elements"])
        rb = b.get("not_removable_because")
        lines.append(f"  rec {b['rec']:>2} idx {b['idx']:>2} seq {b['seq']:>4} {b['decision']:<8} {el}"
                     + (f"   [needed: {rb['code']} at {rb['at'][:60]}]" if rb and b["decision"] != "remove" else ""))
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
