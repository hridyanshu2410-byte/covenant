"""The meaning solver: one group-labelled union-find per dimension
(labels.LabelledUF), with the vocabulary the front ends use.

    s = MeaningSolver(report)
    s.set(key, "color", "srgb", origin, where)      # key holds sRGB-encoded values
    s.union(a, b, "frame", where)                    # same meaning
    s.edge(a, b, "color", SRGB_DEC, origin, where)   # b = sRGB_decode(a)
    s.finish()                                       # domain checks + diagnostics

A conflict is two paths between the same nodes whose labels differ (a cycle
with a non-identity product); the diagnostic shows both paths (the edges that
carry an origin) and the residual. Conflicts whose paths use a fingerprint or
heuristic edge (weak) are warnings; all others are errors.

Codes: meaning.color-mismatch (color), meaning.frame-mismatch (frame, from,
to), meaning.unit-mismatch (unit, angle), meaning.convention-mismatch
(conv.alpha, alpha.power, clip_y, depth.range, depth.dir).
"""
from .labels import FreeGroup, IntGroup, LabelledUF, Trivial, UnitGroup, Z2
from .signatures import encoding_element

HEURISTIC = "inferred: "

PRETTY = {"srgb": "sRGB", "pq": "PQ", "hlg": "HLG"}

# dimension -> (group, equality?, uf name)
DIM_SPEC = {
    "color": (FreeGroup(PRETTY), False),
    "frame": (Trivial(), True),
    "unit": (UnitGroup(), False),
    "conv.alpha": (IntGroup("premultiply", {0: "straight", 1: "premultiplied"}), False),
    "alpha.power": (IntGroup("coverage", {0: "opaque", 1: "coverage", 2: "coverage squared"}), False),
    "clip_y": (Z2("flip", {0: "+Y up", 1: "+Y down"}), False),
    "depth.range": (IntGroup("[0,1]->[-1,1]", {0: "[0,1]", 1: "[-1,1]"}), False),
    "depth.dir": (Z2("reverse", {0: "forward", 1: "reversed"}), False),
    "angle": (IntGroup("deg", {0: "radians", 1: "degrees"}), False),
}
UF_OF = {"from": "frame", "to": "frame"}
WORDS = {"color": "color encoding", "frame": "coordinate frame", "unit": "unit",
         "conv.alpha": "alpha convention", "alpha.power": "stored alpha",
         "clip_y": "clip-space Y direction", "depth.range": "clip depth range",
         "depth.dir": "depth direction", "angle": "angle unit"}
CODES = {"color": "meaning.color-mismatch", "frame": "meaning.frame-mismatch",
         "unit": "meaning.unit-mismatch", "angle": "meaning.unit-mismatch"}
CONVENTION = "meaning.convention-mismatch"
WILDCARDS = {"data", "any"}

VALUE = {
    "conv.alpha": {"straight": 0, "premultiplied": 1},
    "alpha.power": {"opaque": 0, "coverage": 1},
    "clip_y": {"up": 0, "down": 1},
    "depth.range": {"zero_to_one": 0, "neg_one_to_one": 1},
    "depth.dir": {"forward": 0, "reverse_z": 1, "reversed": 1},
    "angle": {"rad": 0, "radians": 0, "deg": 1, "degrees": 1},
}
ANCHOR = {"color": "linear", "unit": "1", "conv.alpha": "straight", "alpha.power": "opaque",
          "clip_y": "+Y up", "depth.range": "[0,1]", "depth.dir": "forward", "angle": "radians"}


class MeaningSolver:
    def __init__(self, report):
        self.report = report
        self.ufs = {d: LabelledUF(d, g, eq) for d, (g, eq) in DIM_SPEC.items()}
        self.wild = {d: set() for d in self.ufs}
        self.domain = []         # (dim, allowed elements, message)

    # ------------------------------------------------------------ vocabulary
    def uf(self, dim):
        return self.ufs[UF_OF.get(dim, dim)]

    def element(self, dim, value):
        d = UF_OF.get(dim, dim)
        if d == "color":
            return encoding_element(value)
        if d == "unit":
            return UnitGroup().of(value)
        return VALUE.get(d, {}).get(value)

    def set(self, k, dim, value, origin, where, weak=False):
        """k holds `value` (an annotation value: 'srgb', 'premultiplied' ...)."""
        uf = self.uf(dim)
        d = uf.dim
        if value in WILDCARDS:
            uf.find(k)
            self.wild[d].add(k)
            return
        weak = weak or origin.startswith(HEURISTIC)
        origin = origin[len(HEURISTIC):] if origin.startswith(HEURISTIC) else origin
        if uf.equality:
            a = uf.anchor(value)
            return uf.add(a, k, uf.G.identity, f"'{value}' comes from {origin}", where, weak)
        g = self.element(dim, value)
        if g is None:
            return None
        a = uf.anchor(ANCHOR[d])
        return uf.add(a, k, g, f"'{value}' comes from {origin}", where, weak)

    def union(self, a, b, dim, where, origin=""):
        uf = self.uf(dim)
        return uf.add(a, b, uf.G.identity, origin, where)

    def edge(self, a, b, dim, g, origin, where, weak=False):
        """b = g(a)."""
        return self.uf(dim).add(a, b, g, origin, where, weak)

    def resolved(self, k, dim):
        return self.uf(dim).resolved(k)

    def require_domain(self, dim, allowed, why, observable=None):
        """Every observable node of dim resolved against its anchor must have
        a label in `allowed` (e.g. Porter-Duff premultiplies at most once).
        observable(node) -> bool limits the check to values that are stored
        or consumed (intermediate SSA values of a computation are not)."""
        ok = allowed if callable(allowed) else (lambda x, _s=frozenset(allowed): x in _s)
        self.domain.append((dim, ok, why, observable or (lambda n: True)))

    # ------------------------------------------------------------ reporting
    def finish(self):
        for d, uf in self.ufs.items():
            wild_roots = {uf.find(k) for k in self.wild[d]}
            for c in uf.conflicts:
                if uf.find(c.node) in wild_roots or (c.path_a and uf.find(c.path_a[0].a) in wild_roots):
                    continue
                self._emit(d, uf, c)
        for d, allowed, why, observable in self.domain:
            uf = self.uf(d)
            wild_roots = {uf.find(k) for k in self.wild[uf.dim]}
            worst = {}
            for n in list(uf.parent):
                if n[0] == "anchor" or not observable(n):
                    continue
                got = uf.resolved(n)
                if got is None or allowed(got[1]) or uf.find(n) in wild_roots:
                    continue
                r = uf.find(n)
                if r not in worst:              # the first violating node of the class
                    worst[r] = (n, got[1])
            for r, (n, lab) in worst.items():
                a = uf.anchor_of.get(r)
                path, _ = uf.path(a, n)
                path = path or []
                notes = self._notes(path) + [why]
                weak = any(e.weak for e in path)
                self.report.add(
                    "meaning", CODES.get(uf.dim, CONVENTION),
                    f"{WORDS.get(uf.dim, uf.dim)} mismatch: a value is "
                    f"'{self._describe(uf, lab)}' relative to {ANCHOR.get(uf.dim, 'its anchor')}, "
                    f"which is not a valid state.",
                    path[-1].where if path else "", notes,
                    severity="warning" if weak else "error")

    @staticmethod
    def _notes(path):
        out = []
        for e in path:
            if e.origin and e.origin not in out:
                out.append(e.origin)
        return out

    def _describe(self, uf, lab):
        if uf.dim == "color":
            inv = {v: k for k, v in {"linear": (), "srgb": (("srgb", 1),), "pq": (("pq", 1),),
                                     "hlg": (("hlg", 1),)}.items()}
            if lab in inv:
                return inv[lab]
        return uf.G.fmt(lab)

    def _emit(self, d, uf, c):
        code = CODES.get(d, CONVENTION)
        sev = "warning" if c.weak else "error"
        word = WORDS.get(d, d)
        if c.residual is None:                    # two named anchors (equality dimension)
            a1, a2 = c.anchors
            notes = self._notes(c.path_a) + self._notes(c.path_b)
            self.report.add("meaning", code,
                            f"{word} mismatch: one side is '{a1}' and the other is '{a2}'.",
                            c.where, notes, severity=sev)
            return
        la, lb = self._describe(uf, c.label_a), self._describe(uf, c.label_b)
        start = c.anchors[0] if c.anchors else None
        notes = []
        pa, pb = self._notes(c.path_a), self._notes(c.path_b)
        if pa:
            notes.append(f"path 1 ({la}): " + "; then ".join(pa))
        if pb:
            notes.append(f"path 2 ({lb}): " + "; then ".join(pb))
        notes.append(f"residual: {uf.G.fmt(c.residual)}")
        rel = f" relative to {start}" if start and uf.G is not None else ""
        self.report.add("meaning", code,
                        f"{word} mismatch: one side is '{la}' and the other is '{lb}'{rel}.",
                        c.where, notes, severity=sev)
