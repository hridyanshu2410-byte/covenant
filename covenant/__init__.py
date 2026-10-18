"""COVENANT research prototype."""
from . import meaning, structural, timing
from .diagnostics import Report


def check_all(pipelines=(), host_writes=(), frame=None, commands=None, layouts=()):
    """Run every checker and return one Report.
    layouts: [(contract, host_fields, where)] hand-written host replicas."""
    r = Report()
    for p in pipelines:
        structural.check_io(p, r)
    for c, hf, w in layouts:
        structural.check_host_layout(c, hf, w, r)
    if commands:
        structural.check_commands(commands, r)
    meaning.check(list(pipelines), host_writes, r)
    if frame is not None:
        timing.check(frame, r)
    return r
