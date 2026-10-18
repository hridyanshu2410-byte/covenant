"""Group-labelled (potential / weighted) union-find: the core of the meaning layer.

Every meaning dimension is a group G. A dataflow edge x -> y carries an element
g of G and states y = g(x): the value at y is the value at x after g has been
applied (an encode, a premultiplication, a Y flip, a unit conversion ...).
Known facts are edges from an *anchor* node (for example LINEAR light, or the
named frame 'world'). The solver keeps, for every node, its potential
relative to the root of its class; adding an edge between two nodes of one
class closes a cycle, and the cycle is consistent iff the product of its
labels is the identity. A conflict is a cycle with a non-identity product
(paper: a meaning bug is a non-trivial holonomy of the program's meaning
connection). The diagnostic shows the two paths and the residual.

Anchors: a class may contain anchor nodes. In a *group* dimension all facts
are stated relative to one anchor per dimension (LINEAR light, STRAIGHT alpha,
...), so only cycles can conflict. In an *equality* dimension (frames: 'world',
'view'; unit names) the group is trivial and the anchors are distinct named
constants; two different anchors in one class are a conflict.

Groups (one per dimension):
  FreeGroup        color transfer: free group on {sRGB, PQ, HLG, ...} modulo
                   enc . dec = id (elements are reduced words)
  IntGroup         alpha premultiplication count (Z), alpha power, angle scale
                   (powers of pi/180), depth range remaps
  Z2               clip-space Y / handedness, depth direction
  UnitGroup        Z^n exponents over base units with a decimal scale
  Trivial          frames (equality with named anchors)
"""
from collections import deque


# ------------------------------------------------------------------ groups
class Trivial:
    name = "trivial"
    identity = ()

    def mul(self, a, b):
        return ()

    def inv(self, a):
        return ()

    def is_id(self, a):
        return True

    def fmt(self, a):
        return "id"


class IntGroup:
    """(Z, +). fmt names the generator."""
    identity = 0

    def __init__(self, gen="g", names=None):
        self.gen, self.names = gen, names or {}

    def mul(self, a, b):
        return a + b

    def inv(self, a):
        return -a

    def is_id(self, a):
        return a == 0

    def fmt(self, a):
        if a in self.names:
            return self.names[a]
        if a == 0:
            return "id"
        return self.gen if a == 1 else f"{self.gen}^{a}"


class Z2(IntGroup):
    def mul(self, a, b):
        return (a + b) % 2

    def inv(self, a):
        return a % 2

    def is_id(self, a):
        return a % 2 == 0


SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


class FreeGroup:
    """Free group on named generators; an element is a reduced word, a tuple of
    (generator, exponent) in *application order* (first applied first).
    Exponent +1 = encode, -1 = decode (for transfer functions)."""
    identity = ()

    def __init__(self, pretty=None, pos="enc", neg="dec"):
        self.pretty, self.pos, self.neg = pretty or {}, pos, neg

    @staticmethod
    def word(*items):
        return FreeGroup.reduce(tuple(items))

    @staticmethod
    def reduce(w):
        out = []
        for g, e in w:
            if e == 0:
                continue
            if out and out[-1][0] == g:
                e2 = out[-1][1] + e
                out.pop()
                if e2:
                    out.append((g, e2))
            else:
                out.append((g, e))
        return tuple(out)

    def mul(self, a, b):                   # a then b
        return self.reduce(tuple(a) + tuple(b))

    def inv(self, a):
        return tuple((g, -e) for g, e in reversed(a))

    def is_id(self, a):
        return not a

    def fmt(self, a):
        if not a:
            return "id (linear)"
        parts = []
        for g, e in a:
            name = self.pretty.get(g, g)
            kind = self.pos if e > 0 else self.neg
            parts.append(f"{name}_{kind}" + (str(abs(e)).translate(SUP) if abs(e) > 1 else ""))
        return " then ".join(parts)


class UnitGroup:
    """Kennedy-style units: an element is (exponents over base units, log10
    scale): 'cm' = ({'m': 1}, -2). Abelian."""
    identity = ((), 0.0)
    SCALE = {"m": ("m", 0), "cm": ("m", -2), "mm": ("m", -3), "km": ("m", 3),
             "s": ("s", 0), "ms": ("s", -3), "us": ("s", -6), "ns": ("s", -9),
             "kg": ("kg", 0), "g": ("kg", -3)}

    def of(self, name):
        base, sc = self.SCALE.get(name, (name, 0))
        return (((base, 1),), float(sc))

    def mul(self, a, b):
        d = dict(a[0])
        for k, v in b[0]:
            d[k] = d.get(k, 0) + v
        return (tuple(sorted((k, v) for k, v in d.items() if v)), round(a[1] + b[1], 6))

    def inv(self, a):
        return (tuple((k, -v) for k, v in a[0]), -a[1])

    def is_id(self, a):
        return not a[0] and abs(a[1]) < 1e-9

    def fmt(self, a):
        for n, (base, sc) in self.SCALE.items():
            if a == (((base, 1),), float(sc)):
                return n
        if self.is_id(a):
            return "dimensionless"
        dims = "·".join(k if v == 1 else f"{k}^{v}" for k, v in a[0]) or "1"
        return dims if not a[1] else f"{dims}·10^{a[1]:g}"


# ------------------------------------------------------------------ solver
class Edge:
    __slots__ = ("a", "b", "g", "origin", "where", "weak")

    def __init__(self, a, b, g, origin, where, weak):
        self.a, self.b, self.g, self.origin, self.where, self.weak = a, b, g, origin, where, weak


class Conflict:
    """Two paths between the same pair of nodes with different labels."""

    def __init__(self, dim, group, node, path_a, label_a, path_b, label_b, residual, where,
                 anchors=None):
        self.dim, self.group, self.node = dim, group, node
        self.path_a, self.label_a, self.path_b, self.label_b = path_a, label_a, path_b, label_b
        self.residual, self.where, self.anchors = residual, where, anchors

    @property
    def weak(self):
        return any(e.weak for e in self.path_a + self.path_b)

    def edges(self):
        return self.path_a + self.path_b


class LabelledUF:
    """One labelled union-find per dimension.

    add(x, y, g): y = g(x). anchor(name) returns the node of a named anchor.
    Conflicts are collected in self.conflicts (one per class pair / cycle)."""

    def __init__(self, dim, group, equality=False):
        self.dim, self.G, self.equality = dim, group, equality
        self.parent, self.pot = {}, {}       # node -> parent, label parent -> node
        self.anchor_of = {}                  # root -> (anchor node, label anchor -> root ... via pot)
        self.adj = {}                        # node -> [(edge, forward?)]
        self.conflicts = []
        self._seen = set()

    # -- nodes
    def anchor(self, name):
        n = ("anchor", self.dim, name)
        self._node(n)
        r = self.find(n)
        self.anchor_of.setdefault(r, n)
        return n

    def _node(self, n):
        if n not in self.parent:
            self.parent[n] = n
            self.pot[n] = self.G.identity

    def find(self, n):
        """-> root; updates pot[n] to the label root -> n (path compression)."""
        self._node(n)
        path = []
        while self.parent[n] != n:
            path.append(n)
            n = self.parent[n]
        root = n
        # compress from the top: pot(x) = pot(parent) then w(x)
        for x in reversed(path):
            p = self.parent[x]
            if p != root:
                self.pot[x] = self.G.mul(self.pot[p], self.pot[x])
                self.parent[x] = root
        return root

    def label(self, n):
        """(root, label root -> n)"""
        r = self.find(n)
        return r, self.pot[n]

    def resolved(self, n):
        """(anchor name, label anchor -> n) if n's class contains an anchor."""
        r, p = self.label(n)
        a = self.anchor_of.get(r)
        if a is None:
            return None
        _, pa = self.label(a)
        return a[2], self.G.mul(self.G.inv(pa), p)

    # -- constraints
    def add(self, x, y, g, origin="", where="", weak=False):
        """Constrain y = g(x). Returns a Conflict or None."""
        if x == y and self.G.is_id(g):
            return None
        self._node(x)
        self._node(y)
        e = Edge(x, y, g, origin, where, weak)
        rx, px = self.label(x)
        ry, py = self.label(y)
        if rx == ry:
            via_new = self.G.mul(px, g)
            if self.G.mul(via_new, self.G.inv(py)) == self.G.identity or \
                    self.G.is_id(self.G.mul(via_new, self.G.inv(py))):
                self._link(e)
                return None
            return self._report(e, x, y, g)
        ax, ay = self.anchor_of.get(rx), self.anchor_of.get(ry)
        if ax is not None and ay is not None and ax != ay and self.equality:
            return self._report(e, x, y, g, anchors=(ax, ay))
        # attach ry under rx: w = (px then g) then inverse(py)
        w = self.G.mul(self.G.mul(px, g), self.G.inv(py))
        self.parent[ry] = rx
        self.pot[ry] = w
        if ax is None and ay is not None:
            self.anchor_of[rx] = ay
        self.anchor_of.pop(ry, None)
        self._link(e)
        return None

    def _link(self, e):
        self.adj.setdefault(e.a, []).append((e, True))
        self.adj.setdefault(e.b, []).append((e, False))

    def path(self, src, dst):
        """Edges of a path src -> dst in the constraint graph (BFS), with the
        label accumulated along it."""
        if src == dst:
            return [], self.G.identity
        prev = {src: None}
        dq = deque([src])
        while dq:
            n = dq.popleft()
            if n == dst:
                break
            for e, fwd in self.adj.get(n, ()):
                m = e.b if fwd else e.a
                if m not in prev:
                    prev[m] = (n, e, fwd)
                    dq.append(m)
        if dst not in prev:
            return None, None
        out, lab = [], []
        n = dst
        while prev[n] is not None:
            p, e, fwd = prev[n]
            out.append(e)
            lab.append(e.g if fwd else self.G.inv(e.g))
            n = p
        out.reverse()
        lab.reverse()
        acc = self.G.identity
        for g in lab:
            acc = self.G.mul(acc, g)
        return out, acc

    def _report(self, e, x, y, g, anchors=None):
        r = self.find(x)
        if anchors:
            a1, a2 = anchors
            pa, la = self.path(a1, x)
            pb, lb = self.path(a2, y)
            pa = (pa or []) + [e]
            key = (self.dim, a1, a2, e.where)
            c = Conflict(self.dim, self.G, y, pa, la, pb or [], lb, None, e.where, anchors=(a1[2], a2[2]))
        else:
            start = self.anchor_of.get(r) or x
            pa, la = self.path(start, x)
            pb, lb = self.path(start, y)
            la = self.G.mul(la if la is not None else self.G.identity, g)
            pa = (pa or []) + [e]
            residual = self.G.mul(self.G.inv(lb if lb is not None else self.G.identity), la)
            key = (self.dim, start, residual, e.where)
            c = Conflict(self.dim, self.G, y, pa, la, pb or [], lb, residual, e.where,
                         anchors=(start[2],) if start[0] == "anchor" else None)
        if key in self._seen:
            return None
        self._seen.add(key)
        self.conflicts.append(c)
        return c


class System:
    """All dimensions together: system.add(dim, x, y, g, ...)."""

    def __init__(self, dims):
        """dims: {name: (group, equality?)}"""
        self.ufs = {d: LabelledUF(d, g, eq) for d, (g, eq) in dims.items()}

    def __getitem__(self, d):
        return self.ufs[d]

    def add(self, dim, x, y, g=None, origin="", where="", weak=False):
        uf = self.ufs[dim]
        return uf.add(x, y, uf.G.identity if g is None else g, origin, where, weak)

    def same(self, dims, x, y, origin="", where="", weak=False):
        for d in dims:
            if d in self.ufs:
                self.add(d, x, y, None, origin, where, weak)

    def conflicts(self):
        return [c for uf in self.ufs.values() for c in uf.conflicts]
