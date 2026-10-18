"""Pillar 2: meaning inference.

Every value gets one unknown per meaning dimension (frame, color, unit).
Declared facts (seed facts, contract tags, host writes) and operation rules
add equalities; a union-find solves them. A conflict is reported with the
origin of both sides, so the message says *why* each side has its meaning.

Only a handful of seed facts are needed: everything else is inferred.
"""
from ..diagnostics import Report
from ..ir import DIMENSIONS

WORDS = {"frame": "coordinate frame", "color": "color encoding", "unit": "unit",
         "conv.alpha": "alpha convention"}
# dimensions whose conflicts are convention mismatches rather than "<dim>-mismatch"
CODES = {"conv.alpha": "meaning.convention-mismatch"}

# Values that declare a deliberate idiom: they agree with anything.
# e.g. Color("data") on a texture that packs non-color data in an RGBA8 image.
WILDCARDS = {"data", "any"}
HEURISTIC = "inferred: "     # origin prefix for facts that come from a heuristic


class _UF:
    def __init__(self, report):
        self.parent, self.val = {}, {}
        self.report = report
        self.reported = set()

    def find(self, k):
        self.parent.setdefault(k, k)
        while self.parent[k] != k:
            self.parent[k] = self.parent[self.parent[k]]
            k = self.parent[k]
        return k

    def _conflict(self, dim, a, b, where):
        key = (dim, a[0], b[0], where)
        if key in self.reported:
            return
        self.reported.add(key)
        heur = any(o.startswith(HEURISTIC) for o in (a[1], b[1]))
        strip = lambda o: o[len(HEURISTIC):] if o.startswith(HEURISTIC) else o
        self.report.add(
            "meaning", CODES.get(dim, f"meaning.{dim}-mismatch"),
            f"{WORDS.get(dim, dim)} mismatch: one side is '{a[0]}' and the other is '{b[0]}'.",
            where, [f"'{a[0]}' comes from {strip(a[1])}", f"'{b[0]}' comes from {strip(b[1])}"],
            severity="warning" if heur else "error")

    def set(self, k, dim, value, origin, where):
        r = self.find(k)
        if value in WILDCARDS:
            self.val[r] = (value, origin)
            return
        if r in self.val and self.val[r][0] in WILDCARDS:
            return
        if r in self.val and self.val[r][0] != value:
            self._conflict(dim, self.val[r], (value, origin), where)
        else:
            self.val.setdefault(r, (value, origin))

    def union(self, a, b, dim, where):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        va, vb = self.val.get(ra), self.val.get(rb)
        if va and vb and (va[0] in WILDCARDS or vb[0] in WILDCARDS):
            w = va if va[0] in WILDCARDS else vb
            self.parent[rb] = ra
            self.val[ra] = w
            return
        if va and vb and va[0] != vb[0]:
            self._conflict(dim, va, vb, where)
            return
        self.parent[rb] = ra
        if not va and vb:
            self.val[ra] = vb

    def value(self, k):
        v = self.val.get(self.find(k))
        return v[0] if v else None


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
            uf.set(_n(e, "color"), "color", "linear",
                   f"hardware sRGB decode of '{tex.name}'", e.loc)
        else:
            uf.union(_n(e, "color"), stored, "color", e.loc)
        uf.union(_n(e, "frame"), _c(tex, "frame"), "frame", e.loc)
    elif op == "srgb_to_linear":
        uf.set(_n(a[0], "color"), "color", "srgb", "the input of srgb_to_linear", e.loc)
        uf.set(_n(e, "color"), "color", "linear", "the output of srgb_to_linear", e.loc)
        _same(uf, e, a[0], e, ("frame",))
    elif op == "linear_to_srgb":
        uf.set(_n(a[0], "color"), "color", "linear", "the input of linear_to_srgb", e.loc)
        uf.set(_n(e, "color"), "color", "srgb", "the output of linear_to_srgb", e.loc)
    elif op == "tonemap":
        uf.set(_n(a[0], "color"), "color", "linear",
               "the tonemapper, which expects linear HDR color", e.loc)
        uf.set(_n(e, "color"), "color", "linear", "the tonemapper output", e.loc)
    elif op == "store":
        tgt = e.attrs["contract"]
        if tgt.format.endswith("_srgb"):
            uf.set(_n(a[0], "color"), "color", "linear",
                   f"'{tgt.name}' using format {tgt.format}, which encodes sRGB on write",
                   e.loc)
            uf.set(_c(tgt, "color"), "color", "srgb", f"format {tgt.format}", e.loc)
        else:
            uf.union(_n(a[0], "color"), _c(tgt, "color"), "color", e.loc)
        uf.union(_n(a[0], "frame"), _c(tgt, "frame"), "frame", e.loc)
