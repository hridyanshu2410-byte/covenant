"""Checker configuration: the verified configuration and the opt-in heuristics.

The *verified* configuration is the default and is what the soundness theorem is
about. Two switches leave it; each is read from the environment so that every
harness (bench/run.py, check_trace.py, report_groups) sees the same setting:

COVENANT_HEURISTIC_DISCARD=1   relaxation R1 (``timing.check(discard_relaxed=True)``):
    a write that precedes an image layout transition from UNDEFINED needs only an
    execution dependency to the transition. This is NOT the specification
    (synchronization.adoc "Image Layout Transitions": "applications must ensure that
    all memory writes have been made available before a layout transition is
    executed"; the UNDEFINED old layout only allows the contents to be discarded), so
    with it set a clean result is outside Theorem 4.3. Default: off (spec-strict).

COVENANT_HOST_CAPTURE=observed  host writes to mapped memory are charged only to the
    bytes that differ from the driver's shadow copy. A store of the
    bytes already there, or a write later restored within one capture interval, is
    then not seen. Default: ``conservative``: every page the page-protection trap saw
    written in an interval is charged, over every mapped binding it overlaps, whether
    or not its bytes changed. Only the conservative mode carries the
    host-write part of the theorem.

COVENANT_LEGACY_SUBPASS_FLAGS=global  for traces whose driver did not record subpass
    dependency flags, read every explicit subpass dependency as framebuffer-global.
    Default: framebuffer-local (the weaker reading, conservative).
"""
import os

_TRUE = ("1", "true", "yes", "on")


def discard_heuristic():
    return os.environ.get("COVENANT_HEURISTIC_DISCARD", "").strip().lower() in _TRUE


def host_capture():
    v = os.environ.get("COVENANT_HOST_CAPTURE", "conservative").strip().lower()
    if v not in ("conservative", "observed"):
        raise ValueError(f"COVENANT_HOST_CAPTURE must be 'conservative' or 'observed', not {v!r}")
    return v


def legacy_subpass_global():
    """COVENANT_LEGACY_SUBPASS_FLAGS=global: traces whose driver did not record subpass
    dependency flags take every explicit subpass dependency as framebuffer-global (the
    stronger reading). Default: framebuffer-local, the weaker reading."""
    return os.environ.get("COVENANT_LEGACY_SUBPASS_FLAGS", "") == "global"


def verified():
    """Is the current configuration the verified one (theorem applies)?"""
    return not discard_heuristic() and host_capture() == "conservative" and not legacy_subpass_global()


def describe():
    return {"discard": "heuristic-R1" if discard_heuristic() else "strict",
            "host_capture": host_capture(),
            "legacy_subpass_flags": "global" if legacy_subpass_global() else "local",
            "verified_configuration": verified()}
