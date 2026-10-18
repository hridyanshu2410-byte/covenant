"""Window sufficiency: is the traced frame long enough?

Window theorem. If after a warm-up of w frames the trace is periodic with period
p frames, and every access of frame i happens-after the host's fence wait for
frame i - N, then every race of an arbitrarily long run of the program is
between accesses at most N + 1 frames apart, and is already present in any
N + 2 consecutive steady-state frames. Checking w + N + 2 frames is complete.

This module finds w, p and N on a timing.Frame built by host_trace:

* frames: the host queue is cut after every Present op; device and present
  queue ops belong to the frame of the vkQueueSubmit ("submit@N" wait) that
  made them run.
* periodicity: each frame gets a signature (per host/device queue, the ops
  with their resources, stages, access flags, masks and layouts; semaphores
  and fences only by kind, and the swapchain image index dropped: the driver
  picks the acquire order, so a steady state must hold for any image). The smallest p, then the
  smallest w, with sig[i] == sig[i + p] for every complete frame i >= w (and at
  least two periods after w) is reported. The last frame and the ops after
  the final present (teardown) are not required to repeat.
* N (frames in flight): for frame i, the newest frame j whose work on every
  device queue precedes a fence or idle wait the host performed before frame
  i's first submission or host access; N = max(i - j) over steady frames.

Scope: what this module computes is a MEASUREMENT, not an instance of the
formal window theorem. That theorem's bound is
M = w + p + max(N, G+1) - 1 frames, with N counted over the presentation queues too and
G the frame distance from a swapchain access to its acquire or present, and it needs a
shift that preserves the pairing of waits with signals. This module counts N over device
queues only, compares signatures that abstract which signal a wait consumes and the
image index, and checks w + N + 2 frames, which reaches M only when
p + max(N, G+1) <= N + 3. WindowInfo.theorem_instance is therefore always False.

``window(frame)`` returns a WindowInfo; ``window_frame(frame, info)``
truncates a frame to its first w + N + 2 frames (the part the window theorem
above says is enough to check). timing.check records the info in stats["window"] and in
report.meta["timing.window"]; it never adds a diagnostic.
"""
import re
from dataclasses import dataclass, field

from .timing import Acquire, Barrier, Frame, Present, Signal, Use, Wait


@dataclass
class WindowInfo:
    frames: int = 0                 # complete frames in the trace (cut at presents)
    periodic: bool = False
    warmup: int = None              # w
    period: int = None              # p, in frames
    in_flight: int = None           # N (None: no fence bound found)
    window: int = None              # N + 2 frames
    needed: int = None              # w + N + 2 frames
    complete: bool = False          # the trace holds at least w + N + 2 frames
    # never an instance of the formal window theorem (module docstring): a measurement only
    theorem_instance: bool = False
    reason: str = ""
    frame_of: dict = field(default_factory=dict, repr=False)   # (queue, index) -> frame

    def summary(self):
        if not self.periodic:
            return f"window: not established ({self.reason})"
        n = "unbounded" if self.in_flight is None else self.in_flight
        s = (f"window: {self.frames} frames, periodic after {self.warmup} with period {self.period}; "
             f"frames in flight N = {n}")
        if self.in_flight is None:
            return s + f"; no fence bound, so Theorem 3 does not apply ({self.reason})"
        return s + (f"; the trace covers warm-up + N + 2 = {self.needed} frames (a measurement: N counts "
                    "device queues only and the signature abstracts wait/signal pairing, so this is not the "
                    "bound M = w + p + max(N, G+1) - 1 of the window theorem)"
                    if self.complete else
                    f"; the trace has {self.frames} frames but {self.needed} are needed (warm-up + N + 2)")


_SEM_NUM = re.compile(r"([@:=])\d+")
_SWAP_IMG = re.compile(r"\[\d+\]")


def _sem(name):
    """The kind of a semaphore / fence instance, not its identity: per-image
    and per-frame-in-flight objects rotate with periods that need not divide
    the frame period, and the checker sees the pairing anyway."""
    name = str(name)
    for k in ("submit@", "idle@", "presented:", "present-waits@"):
        if name.startswith(k):
            return k
    if ">=" in name:
        return "timeline-wait"
    if "=" in name:
        return "timeline"
    if "@" in name:
        return "fence"
    return "binary"


def _res(name, swap):
    """Swapchain images are interchangeable: the driver picks the acquire order."""
    return _SWAP_IMG.sub("[*]", name) if name and name in swap else name


def _sig(op, swap=frozenset()):
    if isinstance(op, Use):
        return ("U", _res(op.res, swap), op.access, op.stage, op.flag, op.layout, op.region, op.pass_op)
    if isinstance(op, Barrier):
        f = lambda m: None if m is None else tuple(sorted(m))
        return ("B", _res(op.res, swap), op.old, op.new, f(op.src_stages), f(op.src_access), f(op.dst_stages),
                f(op.dst_access), op.region)
    if isinstance(op, Signal):
        return ("S", _sem(op.sem))
    if isinstance(op, Wait):
        return ("W", _sem(op.sem), None if op.stages is None else tuple(sorted(op.stages)))
    if isinstance(op, (Acquire, Present)):
        return (type(op).__name__[0], op.swapchain)
    return (type(op).__name__,)


def _attribute(frame):
    """frame index of every op: {(queue, i): k}, and the number of frames."""
    host = frame.queues.get("host", [])
    of, k, submit_frame = {}, 0, {}
    for i, op in enumerate(host):
        of[("host", i)] = k
        if isinstance(op, Signal):
            submit_frame[op.sem] = k
        if isinstance(op, Present):
            k += 1
    nframes = k
    for q, ops in frame.queues.items():
        if q == "host":
            continue
        cur = 0
        for i, op in enumerate(ops):
            if isinstance(op, Wait) and op.sem in submit_frame:
                cur = submit_frame[op.sem]
            of[(q, i)] = cur
    return of, nframes


def window(frame):
    info = WindowInfo()
    host = frame.queues.get("host", [])
    if not any(isinstance(op, Present) for op in host):
        info.reason = "no vkQueuePresentKHR to cut the trace into frames"
        return info
    of, nframes = _attribute(frame)
    info.frame_of = of
    # frames after the last present are teardown: only frames 0..nframes-1 are complete
    info.frames = nframes
    swap = {f"{name}[{i}]" for name, (n, _) in frame.swapchains.items() for i in range(n or 0)}
    sigs = [dict() for _ in range(nframes + 1)]
    for q, ops in frame.queues.items():
        if q.startswith("present:"):
            continue                    # the host's Acquire / Present ops stand for them
        for i, op in enumerate(ops):
            sigs[of[(q, i)]].setdefault(q, []).append(_sig(op, swap))
    # the last frame runs into the teardown (device-idle waits etc.): it is
    # checked, but not required to repeat
    sig = [tuple(sorted((q, tuple(v)) for q, v in s.items())) for s in sigs[:max(nframes - 1, 0)]]
    K = len(sig)
    best = None
    for p in range(1, K // 2 + 1):
        # smallest w: the suffix from w on repeats with period p
        w = K - p
        while w > 0 and sig[w - 1] == sig[w - 1 + p]:
            w -= 1
        if K - w >= 2 * p:
            best = (p, w)
            break
    if best is None:
        info.reason = f"no period found in {K} frames"
        return info
    info.periodic, (info.period, info.warmup) = True, best
    info.in_flight, why = _in_flight(frame, of, nframes, info.warmup)
    if info.in_flight is None:
        info.reason = why
        return info
    info.window = info.in_flight + 2
    info.needed = info.warmup + info.window
    info.complete = info.frames >= info.needed
    return info


def _in_flight(frame, of, nframes, warmup):
    """N: frame i's accesses are ordered after the host's wait for frame i-N."""
    host = frame.queues.get("host", [])
    # position of every signal on its queue
    sig_at = {}
    last_of_frame = {}              # (queue, frame) -> last index
    for q, ops in frame.queues.items():
        for i, op in enumerate(ops):
            if isinstance(op, Signal):
                sig_at[op.sem] = (q, i)
            if q != "host":
                last_of_frame[(q, of[(q, i)])] = i
    devq = [q for q in frame.queues if q != "host" and not q.startswith("present:")]
    covered = {q: -1 for q in devq}    # host has waited for ops [0..covered] of q
    first_work = {}                     # frame -> covered snapshot at its first submit / host access
    for i, op in enumerate(host):
        k = of[("host", i)]
        if isinstance(op, Wait) and op.sem in sig_at and not op.sem.startswith("submit@"):
            q, j = sig_at[op.sem]
            if q in covered:
                covered[q] = max(covered[q], j)
        if k not in first_work and (isinstance(op, Use) or
                                    (isinstance(op, Signal) and op.sem.startswith("submit@"))):
            first_work[k] = dict(covered)
    worst, unbounded = 0, 0
    for i in range(warmup, nframes):
        cov = first_work.get(i)
        if cov is None:
            continue
        j = -1                          # newest frame whose work (and all before it) the host waited for
        for f in range(i):
            if all(last_of_frame.get((q, f), -1) <= cov[q] for q in devq):
                j = f
            else:
                break
        if j < 0 and i > 0:
            unbounded += 1
            if unbounded >= 3:
                return None, f"frame {i} starts before the host waited for any earlier frame's work"
            continue
        worst = max(worst, i - j)
    return worst, ""


def window_frame(frame, info):
    """The first info.needed frames of `frame` (ops of later frames dropped)."""
    if not (info.periodic and info.needed):
        return frame
    keep = info.needed
    queues = {}
    for q, ops in frame.queues.items():
        queues[q] = [op for i, op in enumerate(ops) if info.frame_of.get((q, i), 0) < keep]
    return Frame(queues=queues, initial_layouts=dict(frame.initial_layouts), swapchains=dict(frame.swapchains),
                 semaphore_reuse=list(frame.semaphore_reuse), findings=list(frame.findings))
