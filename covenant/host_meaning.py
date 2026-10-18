"""Whole-host meaning inference: principal convention types over C++ host code.

The shader side (shader_meaning) labels every SPIR-V value; this module does
the same for the host program, so a convention fact that lives only in host
data (an angle converted to degrees, a glTF colour that is linear) can meet a
signature that states the opposite (std::cos takes radians, a shader field is
declared sRGB).

Model (the "typing half" of the checker):

* Every host l-value of interest has a node: locals and parameters per
  *calling context* (call strings of length <= MAX_CTX: let-polymorphism by
  expansion, so a helper called once with degrees and once with radians is not
  a conflict), struct fields per (record, field) (field-based), globals once.
  Straight-line local stores create new versions (x *= DEG_TO_RAD does not
  merge x's old and new meaning).
* Expressions evaluate to (node, scale, constant): multiplying a value by a
  folded constant only changes its scale; the scale is turned into a group
  element when the value is stored or passed. A scale of (pi/180) * r or
  (180/pi) * r with r in {1, 2, 1/2, 4, 1/4} is a degree <-> radian
  conversion (angle element -1 / +1); every other constant scale keeps the
  convention (fov * 0.5 is still in degrees, x / 255 is still sRGB).
* Signatures (covenant/signatures) give the anchors: <cmath> trigonometry
  takes radians, the inverse functions return radians, glm::radians /
  glm::degrees are conversion edges, glm projections take radians,
  std::pow with a display-gamma exponent is an encode / decode, glTF 2.0
  material factors are linear.
* The constraints are solved by the labelled union-find (labels.LabelledUF):
  a type error is a cycle with a non-identity product (two calling contexts
  that need different units at one field), or a node outside its domain
  (an angle is in radians or degrees: element 0 or 1).

This is Kennedy-style dimension inference specialised to abelian convention
groups: unification over Z has most general solutions, so a typable program
has a principal typing; the solver reports exactly the untypable cycles.
"""
import ctypes
import math
from pathlib import Path

import clang.cindex as ci

from .meaning_solver import MeaningSolver
from .signatures import std_trig, glm as glm_sig, projection as proj_sig

K = ci.CursorKind
MAX_CTX = 3
DEG = math.pi / 180.0
NICE = (1.0, 2.0, 0.5, 4.0, 0.25)

# glTF 2.0 (section 5.19 material.pbrMetallicRoughness.baseColorFactor, 5.18
# emissiveFactor): "The factors ... are linear multipliers"; the colour textures
# are sRGB-encoded. Field names as cgltf / tinygltf / fastgltf spell them.
GLTF_LINEAR_FIELDS = {"base_color_factor", "baseColorFactor", "emissive_factor", "emissiveFactor"}

_lib_ready = False


def _evaluate(c):
    """Fold a constant expression with clang (macros such as M_PI, const
    globals, constexpr)."""
    global _lib_ready
    lib = ci.conf.lib
    if not _lib_ready:
        lib.clang_Cursor_Evaluate.restype = ctypes.c_void_p
        lib.clang_Cursor_Evaluate.argtypes = [ci.Cursor]
        lib.clang_EvalResult_getKind.restype = ctypes.c_int
        lib.clang_EvalResult_getKind.argtypes = [ctypes.c_void_p]
        lib.clang_EvalResult_getAsDouble.restype = ctypes.c_double
        lib.clang_EvalResult_getAsDouble.argtypes = [ctypes.c_void_p]
        lib.clang_EvalResult_getAsLongLong.restype = ctypes.c_longlong
        lib.clang_EvalResult_getAsLongLong.argtypes = [ctypes.c_void_p]
        lib.clang_EvalResult_dispose.argtypes = [ctypes.c_void_p]
        _lib_ready = True
    try:
        r = lib.clang_Cursor_Evaluate(c)
    except Exception:
        return None
    if not r:
        return None
    try:
        kind = lib.clang_EvalResult_getKind(r)
        if kind == 1:
            return float(lib.clang_EvalResult_getAsLongLong(r))
        if kind == 2:
            return float(lib.clang_EvalResult_getAsDouble(r))
        return None
    finally:
        lib.clang_EvalResult_dispose(r)


def angle_element(scale):
    """Angle element (power of pi/180 removed) of multiplying by `scale`."""
    if scale is None or scale == 0:
        return 0
    s = abs(scale)
    for k, base in ((-1, DEG), (1, 1.0 / DEG)):
        r = s / base
        if any(abs(r - n) <= 1e-3 * n for n in NICE):
            return k
    return 0


def _where(c):
    loc = c.location
    return f"{Path(loc.file.name).name}:{loc.line}" if loc.file else "?"


def _strip(e):
    while e is not None and e.kind in (K.UNEXPOSED_EXPR, K.PAREN_EXPR, K.CSTYLE_CAST_EXPR,
                                       K.CXX_STATIC_CAST_EXPR, K.CXX_FUNCTIONAL_CAST_EXPR):
        kids = [k for k in e.get_children() if k.kind != K.TYPE_REF]
        if len(kids) != 1:
            break
        e = kids[0]
    return e


CASTS = (K.UNEXPOSED_EXPR, K.CSTYLE_CAST_EXPR, K.CXX_STATIC_CAST_EXPR,
         K.CXX_FUNCTIONAL_CAST_EXPR)
_INT_KINDS = {ci.TypeKind.UCHAR, ci.TypeKind.CHAR_U, ci.TypeKind.SCHAR, ci.TypeKind.CHAR_S,
              ci.TypeKind.USHORT, ci.TypeKind.SHORT, ci.TypeKind.UINT, ci.TypeKind.INT,
              ci.TypeKind.ULONG, ci.TypeKind.LONG, ci.TypeKind.ULONGLONG, ci.TypeKind.LONGLONG}
_FLOAT_KINDS = {ci.TypeKind.FLOAT, ci.TypeKind.DOUBLE, ci.TypeKind.LONGDOUBLE}


def _integral(t):
    return t is not None and t.get_canonical().kind in _INT_KINDS


def _floating(t):
    return t is not None and t.get_canonical().kind in _FLOAT_KINDS


def _ns(ref):
    p = ref.semantic_parent
    out = []
    while p is not None and p.kind == K.NAMESPACE:
        out.append(p.spelling)
        p = p.semantic_parent
    return "::".join(reversed(out))


def _binop(e):
    kids = list(e.get_children())
    if len(kids) != 2:
        return None
    end = kids[0].extent.end.offset
    for tok in e.get_tokens():
        if tok.extent.start.offset >= end and tok.kind == ci.TokenKind.PUNCTUATION:
            return tok.spelling
    return None


def _unop(e):
    toks = [t.spelling for t in e.get_tokens()]
    return toks[0] if toks else None


def _ident(x):
    return x


def _compose(f, g):
    """g after f; None (unknown values) absorbs."""
    if f is None or g is None:
        return None
    return lambda x, _f=f, _g=g: _g(_f(x))


def _trunc(x):
    return float(math.trunc(x))


class Val:
    """An expression's value: node (or None when it carries no tracked meaning),
    the multiplicative constant scale applied to the node (for units), the
    constant value, and fn: the expression's value as a function of the node's
    value (for the value domain; None when not a known function)."""
    __slots__ = ("node", "scale", "const", "why", "fn")

    def __init__(self, node=None, scale=1.0, const=None, why="", fn=_ident):
        self.node, self.scale, self.const, self.why, self.fn = node, scale, const, why, fn

    def then(self, g, scale=1.0, why=None):
        """The value after a known numeric function g (scale: its constant
        factor, for units)."""
        if self.node is None:
            if self.const is None:
                return NONE
            try:
                return Val(None, 1.0, g(self.const))
            except (ValueError, OverflowError, ZeroDivisionError, TypeError):
                return NONE
        return Val(self.node, self.scale * scale, None, self.why if why is None else why,
                   _compose(self.fn, g))


NONE = Val()
GAMMA = "a gamma-coded value"


ROUNDING = {"round": lambda x: float(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1),
            "lround": lambda x: float(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1),
            "roundf": lambda x: float(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1),
            "lrint": lambda x: float(round(x)), "rint": lambda x: float(round(x)),
            "floor": lambda x: float(math.floor(x)), "floorf": lambda x: float(math.floor(x)),
            "ceil": lambda x: float(math.ceil(x)), "ceilf": lambda x: float(math.ceil(x)),
            "trunc": _trunc, "truncf": _trunc}


class HostMeaning:
    def __init__(self, tus, files, solver):
        self.tus, self.files = tus, [Path(f).resolve() for f in files]
        self.s = solver
        self.defs = {}             # usr -> function definition cursor
        self.called = set()
        self.version = {}          # (ctx, usr) -> version number
        self.n_expr = 0
        self.stack = []
        self.field_nodes = set()
        self.vals = {}             # node -> set of concrete values (None: too many / unknown)
        self.names = {}            # local node -> source name
        self.vedges = []           # (src, dst, fn): dst takes fn(src's values)
        self.param_of = {}         # parameter node -> (function usr, index, cursor)
        self.fov_sites = []        # (half-angle node, what, where): perspective half-angles

    # ------------------------------------------------------------ driver
    def _mine(self, c):
        return c.location.file is not None and Path(c.location.file.name).resolve() in self.files

    def field_initializers(self):
        """Default member initializers (struct S { float fov = 45.0f; }): every
        object of S starts with that value, so it flows into the field node."""
        for tu in self.tus:
            for c in tu.cursor.walk_preorder():
                if c.kind != K.FIELD_DECL or not self._mine(c):
                    continue
                kids = [x for x in c.get_children() if x.kind.is_expression() or x.kind == K.UNEXPOSED_EXPR]
                if not kids:
                    continue
                init = kids[-1]
                if _strip(init) is not None and _strip(init).kind == K.INIT_LIST_EXPR:
                    self.init_list(c.type, _strip(init), (), self.var_node((), c))
                    continue
                self.flow(self.expr(init, ()), self.var_node((), c), _where(c))

    def run(self):
        self.field_initializers()
        fns = []
        for tu in self.tus:
            for c in self._walk_defs(tu.cursor):
                fns.append(c)
                self.defs[c.get_usr()] = c
        for f in fns:
            for c in f.walk_preorder():
                if c.kind == K.CALL_EXPR and c.referenced is not None:
                    d = c.referenced.get_definition()
                    if d is not None:
                        self.called.add(d.get_usr())
        for f in fns:
            if f.get_usr() not in self.called:
                self.function(f, ("root", f.spelling), [])

    def _walk_defs(self, root):
        for c in root.get_children():
            if not self._mine(c):
                continue
            if c.kind in (K.FUNCTION_DECL, K.CXX_METHOD, K.CONSTRUCTOR) and c.is_definition():
                yield c
            elif c.kind in (K.NAMESPACE, K.STRUCT_DECL, K.CLASS_DECL, K.UNEXPOSED_DECL):
                yield from self._walk_defs(c)

    # ------------------------------------------------------------ nodes
    def var_node(self, ctx, decl):
        if decl.kind == K.FIELD_DECL:
            n = ("f", decl.semantic_parent.spelling, decl.spelling)
            self.field_nodes.add(n)
            return n
        if decl.kind == K.VAR_DECL and decl.semantic_parent is not None and \
                decl.semantic_parent.kind in (K.TRANSLATION_UNIT, K.NAMESPACE, K.STRUCT_DECL,
                                              K.CLASS_DECL):
            return ("g", decl.get_usr() or decl.spelling)
        key = (ctx, decl.get_usr() or decl.spelling)
        n = ("v", ctx, key[1], self.version.get(key, 0))
        self.names[n] = decl.spelling
        return n

    def new_version(self, ctx, decl):
        if decl.kind != K.VAR_DECL and decl.kind != K.PARM_DECL:
            return self.var_node(ctx, decl)
        if decl.semantic_parent is not None and decl.semantic_parent.kind in (
                K.TRANSLATION_UNIT, K.NAMESPACE):
            return self.var_node(ctx, decl)
        key = (ctx, decl.get_usr() or decl.spelling)
        self.version[key] = self.version.get(key, 0) + 1
        return self.var_node(ctx, decl)

    def fresh(self, why):
        self.n_expr += 1
        return ("e", self.n_expr, why)

    @staticmethod
    def name_of(decl):
        if decl.kind == K.FIELD_DECL:
            return f"{decl.semantic_parent.spelling}.{decl.spelling}"
        return decl.spelling

    # ------------------------------------------------------------ edges
    def flow(self, v, dst, where, what=""):
        """dst = v (with v's scale): an angle edge, colour identity, and a
        value edge."""
        if v is None or dst is None:
            return
        if v.node is None:
            if v.const is not None:
                self._add_value(dst, v.const)
            return
        self.vedges.append((v.node, dst, v.fn))
        k = angle_element(v.scale)
        origin = what
        if k:
            origin = (f"{v.why or 'a value'} multiplied by {v.scale:.6g} at {where} "
                      f"({'radians -> degrees' if k > 0 else 'degrees -> radians'})")
        self.s.edge(v.node, dst, "angle", k, origin, where)
        self.s.edge(v.node, dst, "color", (), "", where)

    def _add_value(self, n, x):
        got = self.vals.setdefault(n, set())
        if got is None:
            return False
        try:
            x = float(x)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(x) or x in got:
            return False
        if len(got) >= 64:
            self.vals[n] = None
            return True
        got.add(x)
        return True

    def propagate_values(self):
        """Fixpoint of the value edges (bounded sets)."""
        for _ in range(30):
            changed = False
            for a, b, fn in self.vedges:
                src = self.vals.get(a)
                if not src or fn is None:
                    if src is None and a in self.vals and self.vals.get(b) is not None:
                        self.vals[b] = None
                        changed = True
                    continue
                for x in list(src):
                    try:
                        y = fn(x)
                    except (ValueError, OverflowError, ZeroDivisionError, TypeError):
                        continue
                    changed |= self._add_value(b, y)
            if not changed:
                break

    def materialize(self, v, where):
        if v is None or v.node is None:
            return None
        if v.scale == 1.0 and v.fn is _ident:
            return v.node
        n = self.fresh("scaled")
        self.flow(v, n, where)
        return n

    # ------------------------------------------------------------ functions
    def function(self, fn, ctx, args):
        """Analyse fn's body in calling context ctx; args are Val of the
        actual arguments. Returns the return node."""
        usr = fn.get_usr()
        if usr in self.stack or len(self.stack) > 12:
            return None
        self.stack.append(usr)
        try:
            params = [c for c in fn.get_children() if c.kind == K.PARM_DECL]
            site = ctx[-1] if ctx and isinstance(ctx[-1], str) else _where(fn)
            for i, p in enumerate(params):
                pn = self.new_version(ctx, p)
                self.param_of[pn] = (usr, i, fn)
                if i < len(args):
                    a = args[i]
                    self.flow(a, pn, site,
                              f"{a.why or 'a value'} passed as parameter '{p.spelling}' of "
                              f"{fn.spelling}() at {site}")
            ret = ("ret", ctx, usr)
            pending = None
            for c in fn.get_children():
                if c.kind == K.MEMBER_REF and c.referenced is not None:
                    pending = c.referenced          # constructor member initializer
                elif pending is not None and (c.kind.is_expression() or c.kind == K.UNEXPOSED_EXPR):
                    self.flow(self.expr(c, ctx), self.var_node(ctx, pending), _where(c))
                    pending = None
                elif c.kind == K.COMPOUND_STMT:
                    self.stmt(c, ctx, ret)
            return ret
        finally:
            self.stack.pop()

    def stmt(self, c, ctx, ret):
        k = c.kind
        if k == K.COMPOUND_STMT or k in (K.IF_STMT, K.FOR_STMT, K.WHILE_STMT, K.DO_STMT,
                                         K.SWITCH_STMT, K.CASE_STMT, K.DEFAULT_STMT,
                                         K.CXX_FOR_RANGE_STMT):
            for ch in c.get_children():
                self.stmt(ch, ctx, ret)
            return
        if k == K.DECL_STMT:
            for d in c.get_children():
                if d.kind == K.VAR_DECL:
                    self.var_decl(d, ctx)
            return
        if k == K.RETURN_STMT:
            kids = list(c.get_children())
            if kids:
                v = self.expr(kids[0], ctx)
                self.flow(v, ret, _where(c))
            return
        if k == K.VAR_DECL:
            self.var_decl(c, ctx)
            return
        if k.is_expression() or k == K.UNEXPOSED_EXPR:
            self.expr(c, ctx)
            return
        for ch in c.get_children():
            self.stmt(ch, ctx, ret)

    def var_decl(self, d, ctx):
        kids = [x for x in d.get_children() if x.kind not in (K.TYPE_REF, K.NAMESPACE_REF,
                                                                 K.TEMPLATE_REF)]
        if not kids:
            return
        init = kids[-1]
        if _strip(init).kind == K.INIT_LIST_EXPR:
            self.init_list(d.type, _strip(init), ctx, self.new_version(ctx, d))
            return
        v = self.expr(init, ctx)
        self.flow(v, self.new_version(ctx, d), _where(d))

    def init_list(self, typ, il, ctx, dst=None):
        rec = typ.get_canonical().get_declaration()
        if rec is None or rec.kind not in (K.STRUCT_DECL, K.CLASS_DECL):
            for e in il.get_children():
                v = self.expr(e, ctx)
                if dst is not None:
                    self.flow(v, dst, _where(e))     # arrays are element-insensitive
            return
        fields = [f for f in rec.get_children() if f.kind == K.FIELD_DECL]
        for f, e in zip(fields, il.get_children()):
            if _strip(e).kind == K.INIT_LIST_EXPR:
                self.init_list(f.type, _strip(e), ctx, self.var_node(ctx, f))
                continue
            v = self.expr(e, ctx)
            self.flow(v, self.var_node(ctx, f), _where(e))

    # ------------------------------------------------------------ expressions
    def const(self, e):
        s = _strip(e)
        if s is None:
            return None
        if s.kind in (K.INTEGER_LITERAL, K.FLOATING_LITERAL, K.DECL_REF_EXPR, K.BINARY_OPERATOR,
                      K.UNARY_OPERATOR, K.PAREN_EXPR, K.CXX_BOOL_LITERAL_EXPR):
            if s.kind == K.DECL_REF_EXPR and s.referenced is not None and \
                    s.referenced.kind not in (K.VAR_DECL, K.ENUM_CONSTANT_DECL):
                return None
            if s.kind == K.DECL_REF_EXPR and s.referenced is not None and \
                    s.referenced.kind == K.VAR_DECL and \
                    not s.referenced.type.is_const_qualified():
                return None
            return _evaluate(e)
        return None

    def expr(self, e, ctx):
        """Evaluate expression e in context ctx -> Val."""
        if e is None:
            return NONE
        c = self.const(e)
        if c is not None:
            return Val(None, 1.0, c)
        if e.kind in CASTS:
            kids = [x for x in e.get_children() if x.kind not in (K.TYPE_REF, K.NAMESPACE_REF)]
            if len(kids) == 1:
                v = self.expr(kids[0], ctx)
                if _integral(e.type) and _floating(kids[0].type):
                    return v.then(_trunc)          # float -> integer conversion truncates
                return v
        s = e
        while s is not None and s.kind == K.PAREN_EXPR:
            kids = list(s.get_children())
            if len(kids) != 1:
                break
            s = kids[0]
        if s.kind in CASTS and s != e:
            return self.expr(s, ctx)
        k = s.kind
        w = _where(s)
        if k in (K.DECL_REF_EXPR, K.MEMBER_REF_EXPR):
            ref = s.referenced
            if k == K.MEMBER_REF_EXPR:
                for ch in s.get_children():          # evaluate the object for side effects
                    if ch.kind == K.CALL_EXPR:
                        self.expr(ch, ctx)
            if ref is not None and ref.kind in (K.VAR_DECL, K.PARM_DECL, K.FIELD_DECL):
                n = self.var_node(ctx, ref)
                return Val(n, 1.0, None, f"'{self.name_of(ref)}'")
            return NONE
        if k == K.ARRAY_SUBSCRIPT_EXPR:
            kids = list(s.get_children())
            if kids:
                for x in kids[1:]:
                    self.expr(x, ctx)
                return self.expr(kids[0], ctx)
            return NONE
        if k == K.BINARY_OPERATOR:
            return self.binary(s, ctx, w)
        if k == K.COMPOUND_ASSIGNMENT_OPERATOR:
            return self.compound(s, ctx, w)
        if k == K.UNARY_OPERATOR:
            kids = list(s.get_children())
            v = self.expr(kids[0], ctx) if kids else NONE
            op = _unop(s)
            if op == "-" and v.node is not None:
                return v.then(lambda x: -x, -1.0)
            if op in ("++", "--"):
                return v
            return v if op in ("+", "*", "&") else NONE
        if k == K.CONDITIONAL_OPERATOR:
            kids = list(s.get_children())
            if len(kids) == 3:
                self.expr(kids[0], ctx)
                a, b = self.expr(kids[1], ctx), self.expr(kids[2], ctx)
                if a.node is None and b.node is None:
                    return NONE
                r = self.fresh("select")
                coded = [x for x in (a, b) if x.node is not None and x.why == GAMMA]
                for x in (a, b):
                    if x.node is None:
                        continue
                    if coded and x.why != GAMMA:
                        # a piecewise transfer function (the sRGB curve's linear
                        # toe next to its power segment): the curve is decided by
                        # its nonlinear piece
                        self.s.edge(x.node, r, "angle", angle_element(x.scale), "", w)
                        continue
                    self.flow(x, r, w)
                return Val(r, 1.0, None, coded[0].why if coded else "a conditional value")
            return NONE
        if k == K.CALL_EXPR:
            return self.call(s, ctx, w)
        if k == K.INIT_LIST_EXPR:
            for ch in s.get_children():
                self.expr(ch, ctx)
            return NONE
        if k == K.LAMBDA_EXPR:
            # a lambda's body runs in the enclosing context (captures by
            # reference name the enclosing locals)
            for ch in s.get_children():
                if ch.kind == K.COMPOUND_STMT:
                    self.stmt(ch, ctx, self.fresh("lambda-ret"))
            return NONE
        for ch in s.get_children():
            if ch.kind.is_expression() or ch.kind == K.UNEXPOSED_EXPR:
                self.expr(ch, ctx)
        return NONE

    def assign_to(self, lhs, v, ctx, w):
        t = _strip(lhs)
        while t is not None and t.kind == K.ARRAY_SUBSCRIPT_EXPR:
            kids = list(t.get_children())
            for x in kids[1:]:
                self.expr(x, ctx)
            t = _strip(kids[0]) if kids else None
        if t is None or t.kind not in (K.DECL_REF_EXPR, K.MEMBER_REF_EXPR):
            if t is not None:
                self.expr(t, ctx)
            return
        ref = t.referenced
        if ref is None or ref.kind not in (K.VAR_DECL, K.PARM_DECL, K.FIELD_DECL):
            return
        array = ref.type.get_canonical().kind in (ci.TypeKind.CONSTANTARRAY,
                                                    ci.TypeKind.INCOMPLETEARRAY)
        dst = self.var_node(ctx, ref) if (array or ref.kind == K.FIELD_DECL) else \
            self.new_version(ctx, ref)
        self.flow(v, dst, w)

    def binary(self, s, ctx, w):
        kids = list(s.get_children())
        op = _binop(s)
        if len(kids) != 2:
            return NONE
        if op == "=":
            v = self.expr(kids[1], ctx)
            self.assign_to(kids[0], v, ctx, w)
            return v
        a, b = self.expr(kids[0], ctx), self.expr(kids[1], ctx)
        if op in ("*", "/"):
            if a.node is not None and b.const is not None and b.const != 0:
                c = b.const if op == "*" else 1.0 / b.const
                return a.then(lambda x, _c=c: x * _c, c)
            if b.node is not None and a.const is not None and op == "*":
                return b.then(lambda x, _c=a.const: x * _c, a.const)
            return NONE
        if op in ("+", "-"):
            if a.node is None and b.node is None:
                return NONE
            if a.node is None or b.node is None:
                if a.node is not None and b.const is not None:
                    c = b.const if op == "+" else -b.const
                    return a.then(lambda x, _c=c: x + _c)
                if b.node is not None and a.const is not None:
                    return b.then((lambda x, _c=a.const: _c + x) if op == "+" else
                                  (lambda x, _c=a.const: _c - x), 1.0 if op == "+" else -1.0)
                return a if a.node is not None else b
            r = self.fresh("sum")
            self.flow(a, r, w)
            self.flow(b, r, w)
            self.vals[r] = None                     # the sum's values are not tracked
            return Val(r, 1.0, None, a.why)
        return NONE

    def compound(self, s, ctx, w):
        kids = list(s.get_children())
        op = _binop(s)
        if len(kids) != 2:
            return NONE
        cur = self.expr(kids[0], ctx)
        b = self.expr(kids[1], ctx)
        if op in ("*=", "/=") and cur.node is not None and b.const:
            c = b.const if op == "*=" else 1.0 / b.const
            v = cur.then(lambda x, _c=c: x * _c, c)
        elif op in ("+=", "-=") and cur.node is not None:
            if b.node is not None:
                self.flow(b, cur.node, w)
            return cur
        else:
            return cur
        self.assign_to(kids[0], v, ctx, w)
        return v

    def call(self, s, ctx, w):
        ref = s.referenced
        args = list(s.get_arguments() or [])
        vals = [self.expr(a, ctx) for a in args]
        if ref is None:
            return NONE
        name, ns = ref.spelling, _ns(ref)
        if ref.kind == K.CONSTRUCTOR:
            d = ref.get_definition()
            if d is not None and self._mine(d) and len(vals) and not \
                    (len(vals) == 1 and d.spelling == d.semantic_parent.spelling and
                     ref.type.argument_types() and
                     ref.type.argument_types()[0].get_pointee().get_canonical().spelling ==
                     d.semantic_parent.type.get_canonical().spelling):
                self.function(d, ctx + (w,) if len(ctx) <= MAX_CTX else ctx, vals)
                return NONE
            if len(vals) == 1:
                return vals[0]                      # copy / conversion
            return NONE
        # <cmath> and glm trigonometry: radians in, radians out
        if name in std_trig.ANGLE_IN and ns in ("std", "", "glm"):
            if vals and vals[0].node is not None:
                arg = self.materialize(vals[0], w)
                self.s.set(arg, "angle", "rad",
                           f"{(ns + '::') if ns else ''}{name}() at {w}, which takes radians", w)
                self._half_angle_site(vals[0], arg, name, w)
            return NONE
        if name in std_trig.ANGLE_OUT and ns in ("std", "", "glm"):
            r = self.fresh(name)
            self.s.set(r, "angle", "rad", f"{(ns + '::') if ns else ''}{name}() at {w}, which "
                       "returns radians", w)
            return Val(r, 1.0, None, f"the result of {name}()")
        if ns == "glm" and name in glm_sig.ANGLE_CONVERT:
            if vals and vals[0].node is not None:
                r = self.fresh(name)
                src = self.materialize(vals[0], w)
                self.s.edge(src, r, "angle", glm_sig.ANGLE_CONVERT[name],
                            f"glm::{name}() at {w} "
                            f"({'degrees -> radians' if name == 'radians' else 'radians -> degrees'})", w)
                return Val(r, 1.0, None, f"glm::{name}()")
            return NONE
        if ns == "glm" and name.split("_")[0] in glm_sig.ANGLE_PARAMS:
            for i in glm_sig.ANGLE_PARAMS[name.split("_")[0]]:
                if i < len(vals) and vals[i].node is not None:
                    arg = self.materialize(vals[i], w)
                    self.s.set(arg, "angle", "rad",
                               f"glm::{name}() at {w}, whose angle parameter is in radians", w)
                    if name.split("_")[0] in proj_sig.GLM_FOV:
                        self._fov_arg(vals[i], 0.5, f"glm::{name}() (fovy, radians)", w)
                elif i < len(vals) and vals[i].const is not None and name.split("_")[0] in proj_sig.GLM_FOV:
                    self._fov_arg(vals[i], 0.5, f"glm::{name}() (fovy, radians)", w)
            return NONE
        lib = proj_sig.lookup(name, ns)
        if lib is not None and (ref.get_definition() is None or not self._mine(ref.get_definition())):
            idx, unit, cite = lib
            if idx < len(vals):
                self._fov_arg(vals[idx], 0.5 if unit == "rad" else math.pi / 360.0,
                              f"{(ns + '::') if ns else ''}{name}() (field of view in {unit}; {cite})", w)
        if name in ("pow", "powf") and ns in ("std", "") and len(vals) == 2:
            return self._pow(vals, w)
        if name in ("min", "max", "fmin", "fmax", "fminf", "fmaxf", "clamp", "abs", "fabs",
                    "fabsf") and ns in ("std", "", "glm"):
            nodes = [v for v in vals if v.node is not None]
            consts = [v.const for v in vals if v.node is None and v.const is not None]
            if not nodes:
                return NONE
            if len(nodes) == 1:
                fn = {"min": min, "fmin": min, "fminf": min, "max": max, "fmax": max,
                      "fmaxf": max}.get(name)
                if fn and len(consts) == 1:
                    return nodes[0].then(lambda x, _c=consts[0], _f=fn: _f(x, _c))
                if name == "clamp" and len(consts) == 2 and vals[0].node is not None:
                    return nodes[0].then(lambda x, _a=consts[0], _b=consts[1]: min(max(x, _a), _b))
                if name in ("abs", "fabs", "fabsf"):
                    return nodes[0].then(abs)
            r = self.fresh(name)
            for v in nodes[:1] if name in ("clamp",) else nodes:
                self.flow(v, r, w)
            self.vals[r] = None
            return Val(r, 1.0, None, nodes[0].why)
        if name in ROUNDING and ns in ("std", "") and len(vals) == 1:
            return vals[0].then(ROUNDING[name])
        d = ref.get_definition()
        if d is None or not self._mine(d):
            return NONE
        site = f"{w}:{s.location.column}"         # one context per call expression
        if len(ctx) >= MAX_CTX + 1:
            cctx = ctx
        else:
            cctx = ctx + (site,)
        ret = self.function(d, cctx, vals)
        if ret is None:
            return NONE
        return Val(ret, 1.0, None, f"the result of {name}()")

    # -------------------------------------------- perspective field of view
    def _projection_like(self, fn):
        """A function that builds a 4x4 matrix from at least four parameters
        (fov, aspect, near, far): it returns a record of >= 16 floats or
        writes through a float pointer / array parameter."""
        params = [c for c in fn.get_children() if c.kind == K.PARM_DECL]
        if len(params) < 4:
            return False
        try:
            if fn.result_type.get_canonical().get_size() >= 64:
                return True
        except Exception:
            pass
        for p in params:
            t = p.type.get_canonical()
            if t.kind == ci.TypeKind.POINTER and _floating(t.get_pointee()):
                return True
            if t.kind in (ci.TypeKind.CONSTANTARRAY, ci.TypeKind.INCOMPLETEARRAY):
                return True
        return False

    def _half_angle_site(self, v, arg, trig, w):
        """tan(k * p), cos(k * p) or sin(k * p) of a parameter p of a projection
        builder, with k = 1/2 (p in radians) or pi/360 (p in degrees, converted
        inline): p is a vertical field of view and k * p its half angle, which
        a perspective projection needs in (0, pi/2) (the field of view is in
        (0, pi)). Structural signature: it holds for any library's builder."""
        if v.node not in self.param_of or not self.stack:
            return
        usr, idx, fn = self.param_of[v.node]
        if usr != self.stack[-1] or not self._projection_like(fn):
            return
        k = abs(v.scale)
        if not (abs(k - 0.5) < 1e-6 or abs(k - math.pi / 360.0) < 1e-6 * math.pi / 360.0):
            return
        p = [c for c in fn.get_children() if c.kind == K.PARM_DECL][idx]
        self.fov_sites.append((arg, f"{trig}() of {k:.6g} * parameter '{p.spelling}' of {fn.spelling}() "
                               f"(a perspective half angle)", w))

    def _fov_arg(self, v, k, what, w):
        """A field-of-view argument of a library projection: its half angle is k * value."""
        if v.node is None and v.const is None:
            return
        if v.node is None:
            n = self.fresh("fov")
            self._add_value(n, v.const * k)
        else:
            n = self.fresh("fov")
            self.flow(v.then(lambda x, _k=k: x * _k, k), n, w)
        self.fov_sites.append((n, what, w))

    def check_fov(self, report):
        """Domain of a perspective half angle: (0, pi/2). A known value outside
        it whose reading as degrees lies inside is degrees passed as radians."""
        seen = set()
        for n, what, w in self.fov_sites:
            vals = self.vals.get(n)
            if not vals:
                continue
            bad = sorted(x for x in vals if not (0.0 < x < math.pi / 2))
            if not bad:
                continue
            x = bad[0]
            if not (0.0 < x * DEG < math.pi / 2) or (w, x) in seen:
                continue
            seen.add((w, x))
            report.add("meaning", "meaning.unit-mismatch",
                       f"angle unit mismatch: {what} at {w} is {x:.6g}, but a perspective half angle lies "
                       f"in (0, pi/2) radians (the field of view in (0, pi)); {x:.6g} degrees "
                       f"(= {x * DEG:.4g} rad) is in range, so the angle was given in degrees.",
                       w, [f"witness: the host value reaching {what} is {x:.6g} (field of view {2 * x:.6g})",
                           "perspective projections take the vertical field of view in radians "
                           "(tan(fovy / 2) or cos/sin(fovy / 2)); convert degrees with pi / 180"])

    def _pow(self, vals, w):
        x, e = vals
        if e.const is None:
            return NONE
        if x.node is None:
            return x.then(lambda v, _e=e.const: v ** _e)
        g = None
        for gamma in (2.2, 2.4):
            if abs(e.const - 1.0 / gamma) < 2e-3:
                g = (("srgb", 1),)
            elif abs(e.const - gamma) < 2e-3:
                g = (("srgb", -1),)
        if g is None:
            return x.then(lambda v, _e=e.const: v ** _e if v >= 0 else float("nan"))
        r = self.fresh("pow")
        src = self.materialize(x, w)
        self.s.edge(src, r, "color", g, f"std::pow(x, {e.const:.4g}) at {w} "
                    f"({'gamma encode' if g[0][1] > 0 else 'gamma decode'})", w, weak=True)
        self.vedges.append((src, r, lambda v, _e=e.const: v ** _e if v >= 0 else float("nan")))
        return Val(r, 1.0, None, GAMMA)


# ------------------------------------------------------------ host -> GPU join
PLAIN = {0.0, 1.0, -1.0, 0.5, 0.25, 0.75, 2.0, 0.125}


def _distinctive(x):
    return math.isfinite(x) and abs(x) < 1e6 and all(abs(x - p) > 1e-6 for p in PLAIN)


def _same(a, b):
    return abs(a - b) <= max(2e-6 * max(abs(a), abs(b)), 1e-7)


def gpu_fields(use):
    """{(block struct name, member name): [floats]} for the uniform and
    push-constant blocks of one pipeline use, decoded from the bytes the model
    driver recorded (top-level float scalars and vectors)."""
    import numpy as np
    from . import probe
    out = {}
    for m, _ in use.stages:
        for vid, d in m.defs.items():
            if d.op != "OpVariable" or not d.args:
                continue
            sc = d.args[0]
            data = None
            if sc == 9:
                data = (use.values or {}).get("push")
            elif sc in (2, 12):
                dd = m.decos.get(vid, {})
                if "Binding" in dd:
                    data = (use.values or {}).get((dd.get("DescriptorSet", [0])[0], dd["Binding"][0]))
            if data is None:
                continue
            blk = data if isinstance(data, tuple) else (bytes(data), b"\x01" * len(data))
            mem = probe.Memory(m, {vid: blk})
            tid = m.pointee(d.type)
            if m.tkind(tid) != "struct":
                continue
            st = m.struct_name(tid)
            for i, mt in enumerate(m.types[tid].args):
                k = m.tkind(mt)
                if k not in ("float", "vector"):
                    continue
                if k == "vector" and m.tkind(m.types[mt].args[0]) != "float":
                    continue
                try:
                    v = mem.decode(mt, vid, (i,))
                except Exception:
                    v = None
                if v is None:
                    continue
                vals = [float(x) for x in np.asarray(v, dtype=np.float64).reshape(-1)]
                out[(st, m.member_names.get((tid, i), str(i)))] = vals
    return out


def _label_name(lab):
    from .signatures import ENCODING
    for name, el in ENCODING.items():
        if tuple(el) == tuple(lab):
            return name
    return None


def check(tus, files, report):
    """Run the host meaning inference and report conflicts into `report`.
    Returns the solver (callers may add GPU-side facts before finish)."""
    s = MeaningSolver(report)
    s.require_domain("angle", {0, 1},
                     "an angle is stored in radians or in degrees (one conversion, not two)",
                     lambda n: n[0] in ("f", "g", "v"))
    hm = HostMeaning(tus, files, s)
    # glTF material factors are linear (glTF 2.0 spec)
    for tu in tus:
        for c in tu.cursor.walk_preorder():
            if c.kind == K.FIELD_DECL and c.spelling in GLTF_LINEAR_FIELDS and hm._mine(c):
                s.set(hm.var_node((), c), "color", "linear",
                      f"the glTF 2.0 field {c.semantic_parent.spelling}.{c.spelling} "
                      "(material factors are linear)", _where(c))
    hm.run()
    hm.propagate_values()
    hm.check_fov(report)
    s.finish()
    return hm


def gpu_seeds(hm, uses):
    """Value provenance: a host value whose convention the host typing knows
    and whose concrete numbers are exactly the numbers a shader block member
    holds at a draw is that member's value (the join of host and shader facts). Returns
    shader_meaning.HostSeed facts for those members."""
    from .shader_meaning import HostSeed
    if hm is None:
        return []
    typed = []
    for n, vals in hm.vals.items():
        if not vals:
            continue
        got = hm.s.resolved(n, "color")
        if got is None:
            continue
        name = _label_name(got[1])
        if name is None:
            continue
        typed.append((n, vals, name))
    if not typed:
        return []
    seeds, seen = [], set()
    for use in uses:
        for (st, mem), fv in gpu_fields(use).items():
            dist = [x for x in fv if _distinctive(x)]
            if not dist or (len(fv) > 1 and len(dist) < 2):
                continue
            for n, vals, name in typed:
                if all(any(_same(x, y) for y in vals) for x in dist):
                    key = (st, mem, name)
                    if key in seen:
                        continue
                    seen.add(key)
                    what = hm.names.get(n, n[2]) if n[0] == "v" else \
                        (f"{n[1]}.{n[2]}" if n[0] == "f" else str(n[1]))
                    uf = hm.s.uf("color")
                    a = uf.anchor_of.get(uf.find(n))
                    path, _ = uf.path(a, n) if a is not None else (None, None)
                    why = next((e.origin for e in (path or []) if e.origin), "the host typing")
                    seeds.append(HostSeed(
                        {"color": name},
                        f"host value '{what}' ({name}: {why}), whose numbers "
                        f"{', '.join(f'{x:.4g}' for x in dist[:4])} are exactly what the shader reads "
                        f"from {st}.{mem} at '{use.name}'", struct=st, member=mem,
                        where=f"{use.name}: {st}.{mem}"))
    return seeds
