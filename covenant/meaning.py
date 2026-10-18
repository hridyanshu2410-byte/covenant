"""Pillar 2: meaning inference over the contract IR (covenant/ir.py).

Every value has one node per meaning dimension (frame, color, unit). Declared
facts (seed facts, contract tags, host writes) and operation rules add edges
to the labelled solver (meaning_solver.MeaningSolver over
labels.LabelledUF): plain operations are identity edges, srgb_to_linear /
linear_to_srgb are sRGB decode / encode edges, an *_srgb format decodes on
sample and encodes on store. A conflict is a cycle with a non-identity label,
reported with both paths.
"""
from .diagnostics import Report
from .ir import DIMENSIONS
from .meaning_solver import HEURISTIC, WILDCARDS, MeaningSolver  # noqa: F401
from .signatures import SRGB_DEC, SRGB_ENC

WORDS = {"frame": "coordinate frame", "color": "color encoding", "unit": "unit",
         "conv.alpha": "alpha convention"}
CODES = {"conv.alpha": "meaning.convention-mismatch"}


class _UF:
    """Compatibility shim: the old equality union-find API on the labelled
    solver (set / union / value)."""

    def __init__(self, report):
        self.s = MeaningSolver(report)
        self.report = report

    def set(self, k, dim, value, origin, where):
        self.s.set(k, dim, value, origin, where)

    def union(self, a, b, dim, where):
        self.s.union(a, b, dim, where)

    def edge(self, a, b, dim, g, origin, where):
        self.s.edge(a, b, dim, g, origin, where)

    def value(self, k):
        for d in ("color", "frame", "unit"):
            if k and k[-1] == d:
                got = self.s.resolved(k, d)
                if got is None:
                    return None
                if d == "color":
                    return {(): "linear", SRGB_ENC: "srgb"}.get(got[1], str(got[1]))
                return got[0]
        return None

    def finish(self):
        self.s.finish()


def _n(e, d):
    return ("e", e.id, d)


def _c(contract, d, fname=None):
    return ("c", contract.name, fname, d)


def check(pipelines, host_writes=(), report=None):
    """Infer meanings across all pipelines (and therefore across passes,
    because render targets are shared contracts). Returns (report, solver)."""
    report = report or Report()
    uf = _UF(report)
    seen = set()

    def seed_contract(c):
        if c.name in seen:
            return
        seen.add(c.name)
        for d, v in c.tag.items():
            uf.set(_c(c, d), d, v, f"the declaration of '{c.name}'", c.name)
        for f in c.fields:
            for d, v in f.tag.items():
                uf.set(_c(c, d, f.name), d, v, f"the declaration of '{c.name}.{f.name}'",
                       c.name)

    for hw in host_writes:
        seed_contract(hw.contract)
        for fname, tag in hw.tags.items():
            for d, v in tag.items():
                key = _c(hw.contract, d, None if fname == "*" else fname)
                uf.set(key, d, v, f"host data written at {hw.where}", hw.where)

    for p in pipelines:
        for s in p.stages:
            for c in s.imports + s.exports:
                seed_contract(c)
        # stage-to-stage links by NAME (fixes RCGP's positional matching)
        for up, down in zip(p.stages, p.stages[1:]):
            for name, src in up.outputs.items():
                dst = down.inputs.get(name)
                if dst is not None:
                    for d in DIMENSIONS:
                        uf.union(_n(src, d), _n(dst, d), d,
                                 f"{up.name} -> {down.name} varying '{name}'")
        for s in p.stages:
            for eid, tag in s.declared.items():
                for d, v in tag.items():
                    uf.set(("e", eid, d), d, v, f"an annotation in {s.name}", s.name)
            for e in s.exprs:
                _rule(uf, e)
    uf.finish()
    return report, uf


def _same(uf, a, b, e, dims=DIMENSIONS):
    for d in dims:
        uf.union(_n(a, d), _n(b, d), d, e.loc)


def _rule(uf, e):
    op, a = e.op, e.args
    if op == "field":
        c, f = e.attrs["contract"], e.attrs["field"]
        if f.ty != "mat4":
            for d in ("frame", "unit", "color"):
                uf.union(_n(e, d), _c(c, d, f.name), d, e.loc)
    elif op in ("add", "sub", "lt"):
        _same(uf, a[0], a[1], e)
        if op != "lt":
            _same(uf, e, a[0], e)
    elif op == "modulate":                       # color * color, same encoding
        _same(uf, a[0], a[1], e, ("color", "frame"))
        _same(uf, e, a[0], e, ("color", "frame"))
    elif op == "dot":
        _same(uf, a[0], a[1], e, ("frame", "color"))
    elif op == "scale":
        _same(uf, e, a[1], e)
    elif op == "normalize":
        _same(uf, e, a[0], e, ("frame", "color"))
    elif op == "transform":
        m = a[0]
        c, f = m.attrs["contract"], m.attrs["field"]
        src, dst = c.matrix_frames[f.name]
        uf.set(_n(a[1], "frame"), "frame", src,
               f"'{c.name}.{f.name}', which maps {src} to {dst}", e.loc)
        uf.set(_n(e, "frame"), "frame", dst,
               f"'{c.name}.{f.name}', which maps {src} to {dst}", e.loc)
        _same(uf, e, a[1], e, ("unit", "color"))
    elif op == "sample":
        tex = a[0].attrs["contract"]
        stored = _c(tex, "color")
        if tex.format.endswith("_srgb"):
            uf.set(stored, "color", "srgb",
                   f"'{tex.name}' using format {tex.format}, which decodes sRGB on read",
                   e.loc)
            uf.edge(stored, _n(e, "color"), "color", SRGB_DEC,
                    f"hardware sRGB decode of '{tex.name}'", e.loc)
        else:
            uf.union(_n(e, "color"), stored, "color", e.loc)
        uf.union(_n(e, "frame"), _c(tex, "frame"), "frame", e.loc)
    elif op == "srgb_to_linear":
        uf.set(_n(a[0], "color"), "color", "srgb", "the input of srgb_to_linear", e.loc)
        uf.edge(_n(a[0], "color"), _n(e, "color"), "color", SRGB_DEC, "srgb_to_linear", e.loc)
        _same(uf, e, a[0], e, ("frame",))
    elif op == "linear_to_srgb":
        uf.set(_n(a[0], "color"), "color", "linear", "the input of linear_to_srgb", e.loc)
        uf.edge(_n(a[0], "color"), _n(e, "color"), "color", SRGB_ENC, "linear_to_srgb", e.loc)
    elif op == "tonemap":
        uf.set(_n(a[0], "color"), "color", "linear",
               "the tonemapper, which expects linear HDR color", e.loc)
        uf.set(_n(e, "color"), "color", "linear", "the tonemapper output", e.loc)
    elif op == "store":
        tgt = e.attrs["contract"]
        if tgt.format.endswith("_srgb"):
            uf.set(_c(tgt, "color"), "color", "srgb", f"format {tgt.format}", e.loc)
            uf.edge(_n(a[0], "color"), _c(tgt, "color"), "color", SRGB_ENC,
                    f"'{tgt.name}' using format {tgt.format}, which encodes sRGB on write", e.loc)
        else:
            uf.union(_n(a[0], "color"), _c(tgt, "color"), "color", e.loc)
        uf.union(_n(a[0], "frame"), _c(tgt, "frame"), "frame", e.loc)
