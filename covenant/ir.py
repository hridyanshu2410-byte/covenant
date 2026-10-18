"""Contract IR: resources (contracts), shader modules, passes and a frame.

This is the language-neutral middle layer of COVENANT. Front ends (Slang,
HLSL, C++ ...) are future work; here programs are built with a small Python
builder so the checkers can be exercised end to end.

A contract is RCGP's (type, layout) pair plus a *meaning tag*: a partial map
from meaning dimensions (frame, color, unit) to values.
"""
from dataclasses import dataclass, field
from itertools import count

DIMENSIONS = ("frame", "color", "unit")

_ids = count()


# ---------------------------------------------------------------- contracts
@dataclass
class Field:
    name: str
    ty: str                     # float, vec2, vec3, vec4, mat4, int
    tag: dict = field(default_factory=dict)


@dataclass
class Contract:
    """A resource contract.

    kind: storage_buffer | uniform | push_constant | texture | render_target
          | attribute_stream
    fields: for buffer-like kinds, the reflected struct fields
    tag: meaning tag of the whole resource (textures, targets, streams)
    format: for images, e.g. "rgba8_srgb" (hardware decode) or "rgba8_unorm"
    per_frame: number of copies for frames in flight (1 = shared)
    matrix_frames: for mat4 fields, {"field": ("world", "view")}
    """
    name: str
    kind: str
    ty: str = ""
    fields: list = field(default_factory=list)
    layout: str = "std430"
    tag: dict = field(default_factory=dict)
    format: str = ""
    per_frame: int = 1
    matrix_frames: dict = field(default_factory=dict)

    def field(self, name):
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"{self.name} has no field {name}")


# ------------------------------------------------------------ shader exprs
@dataclass
class Expr:
    op: str
    args: tuple
    ty: str
    loc: str
    attrs: dict = field(default_factory=dict)
    id: int = field(default_factory=lambda: next(_ids))

    def __repr__(self):
        return f"<{self.op}:{self.ty} @{self.loc}>"


class Shader:
    """Builder for a shader module. Every operation records a source label so
    diagnostics can point at it."""

    def __init__(self, name, stage):
        self.name, self.stage = name, stage
        self.inputs, self.outputs = {}, {}
        self.imports = []          # contracts read
        self.exports = []          # contracts written
        self.exprs = []
        self.declared = {}         # expr id -> tag written by the author

    def _e(self, op, args, ty, loc, **attrs):
        e = Expr(op, tuple(args), ty, f"{self.name}:{loc}", attrs)
        self.exprs.append(e)
        return e

    # leaves
    def input(self, name, ty, tag=None):
        e = self._e("input", [], ty, name, name=name)
        self.inputs[name] = e
        if tag:
            self.declared[e.id] = tag
        return e

    def read(self, contract, fname=None, loc=None):
        if contract not in self.imports:
            self.imports.append(contract)
        if fname is None:
            return self._e("resource", [], contract.ty or "vec4", loc or contract.name,
                           contract=contract)
        f = contract.field(fname)
        return self._e("field", [], f.ty, loc or f"{contract.name}.{fname}",
                       contract=contract, field=f)

    def const(self, value, ty="float", tag=None, loc="const"):
        e = self._e("const", [], ty, loc, value=value)
        if tag:
            self.declared[e.id] = tag
        return e

    # operations
    def add(self, a, b, loc):  return self._e("add", [a, b], a.ty, loc)
    def sub(self, a, b, loc):  return self._e("sub", [a, b], a.ty, loc)
    def dot(self, a, b, loc):  return self._e("dot", [a, b], "float", loc)
    def scale(self, s, v, loc): return self._e("scale", [s, v], v.ty, loc)
    def modulate(self, a, b, loc): return self._e("modulate", [a, b], a.ty, loc)
    def normalize(self, v, loc): return self._e("normalize", [v], v.ty, loc)
    def transform(self, m, v, loc): return self._e("transform", [m, v], v.ty, loc)
    def sample(self, tex, uv, loc): return self._e("sample", [tex, uv], "vec4", loc)
    def srgb_to_linear(self, c, loc): return self._e("srgb_to_linear", [c], c.ty, loc)
    def linear_to_srgb(self, c, loc): return self._e("linear_to_srgb", [c], c.ty, loc)
    def tonemap(self, c, loc): return self._e("tonemap", [c], c.ty, loc)
    def lt(self, a, b, loc): return self._e("lt", [a, b], "bool", loc)

    def reinterpret(self, v, tag, loc):
        """Escape hatch: logged, never silent."""
        e = self._e("reinterpret", [v], v.ty, loc)
        self.declared[e.id] = tag
        return e

    def output(self, name, expr, tag=None):
        self.outputs[name] = expr
        if tag:
            self.declared[expr.id] = {**self.declared.get(expr.id, {}), **tag}

    def write(self, contract, expr):
        """Write expr to a render target / storage resource."""
        if contract not in self.exports:
            self.exports.append(contract)
        e = self._e("store", [expr], expr.ty, f"store {contract.name}", contract=contract)
        return e


@dataclass
class Pipeline:
    name: str
    stages: list               # [vertex, fragment] or [compute]

    @property
    def imports(self):
        out = []
        for s in self.stages:
            for c in s.imports:
                if c not in out:
                    out.append(c)
        return out


@dataclass
class HostWrite:
    """Host code filling a contract, with the meaning of the data it writes
    (e.g. light positions loaded from a file authored in centimeters)."""
    contract: Contract
    tags: dict                 # field name (or "*") -> tag
    where: str
