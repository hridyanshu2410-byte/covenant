"""C++ host front end (static part), built on libclang.

Reads two things a run of the program cannot show:
* the byte layout of host structs that mirror shader blocks, and
* meaning annotations (COV("unit=cm") etc.) and where annotated values flow,
  so a centimeter value stored into a field the shader reads as meters is
  caught.

The dataflow is deliberately small: flow-insensitive, through local
variables, struct fields, function return values and arithmetic. Multiplying
or dividing by anything drops the unit (that is how conversions are written).
"""
import glob
import math
from dataclasses import dataclass, field
from pathlib import Path

import clang.cindex as ci

from . import spirv
from .diagnostics import Report
from .meaning_solver import MeaningSolver
from .shader_meaning import HostSeed
from .signatures import glm as glm_sig, std_trig

K = ci.CursorKind
INCLUDE = Path(__file__).parent / "include"


def _libclang_major():
    """Major version of the libclang the Python bindings loaded (None if unknown)."""
    import re
    try:
        f = ci.conf.lib.clang_getClangVersion
        f.restype = ci._CXString
        m = re.search(r"version (\d+)", ci._CXString.from_result(f()))
        return int(m.group(1)) if m else None
    except Exception:
        return None


def _resource_includes():
    """Clang builtin headers (stddef.h, intrinsics) matching the loaded
    libclang. A newer clang's headers use builtins an older libclang lacks
    (clang-20's avx512bitalgintrin.h needs __builtin_elementwise_popcount, which
    libclang 18 rejects), so take the resource dir of the same major version,
    else the newest one not newer than it, else none."""
    dirs = {}
    for d in glob.glob("/usr/lib/llvm-*/lib/clang/*/include") + glob.glob("/usr/lib/clang/*/include"):
        try:
            dirs.setdefault(int(Path(d).parent.name.split(".")[0]), d)
        except ValueError:
            continue
    ver = _libclang_major()
    if ver is None:
        return [dirs[max(dirs)]] if dirs else []
    ok = [v for v in dirs if v <= ver]
    return [dirs[max(ok)]] if ok else []


CLANG_INC = _resource_includes()


class HostParseError(RuntimeError):
    """libclang could not parse a host source file (the host-static checks
    did not run on it)."""


@dataclass
class HostField:
    name: str
    offset: int
    size: int
    type: str
    tag: dict = field(default_factory=dict)


@dataclass
class HostStruct:
    name: str
    size: int
    fields: list
    block: str = None
    where: str = ""


@dataclass
class HostFacts:
    structs: dict = field(default_factory=dict)       # name -> HostStruct
    seeds: list = field(default_factory=list)         # HostSeed
    flows: list = field(default_factory=list)         # (struct, member, tag, where, source)
    clip_range: dict = field(default_factory=dict)    # file -> GLM's clip depth range, if GLM is used
    values: dict = field(default_factory=dict)        # (struct, member) -> set of numbers stored ("?" = unknown)
    angle_uses: list = field(default_factory=list)    # (call, where, origin): a degree value reaches a radian argument
    size_exprs: list = field(default_factory=list)    # (comps, scalar kind, sizeof bytes, type, where): sizeof(vec) * n
    copied: set = field(default_factory=set)          # record types whose bytes are copied (memcpy, casts)


def _tag(cursor):
    out = {}
    for ch in cursor.get_children():
        if ch.kind == K.ANNOTATE_ATTR and ch.spelling.startswith("covenant:"):
            for part in ch.spelling[len("covenant:"):].split(";"):
                k, _, v = part.partition("=")
                out[k.strip()] = v.strip()
    return out


def _where(c):
    loc = c.location
    return f"{Path(loc.file.name).name}:{loc.line}" if loc.file else "?"


def parse(files, args=()):
    files = [Path(f) for f in files]
    idx = ci.Index.create()
    facts = HostFacts()
    base = ["-x", "c++", "-std=c++20", f"-I{INCLUDE}"] + \
           [f"-isystem{i}" for i in CLANG_INC] + list(args)
    tus = []
    for f in files:
        tu = idx.parse(str(f), args=base + [f"-I{f.parent}"],
                       options=ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
        errs = [d for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]
        if errs:
            raise HostParseError(f"{f}: " + "; ".join(str(e) for e in errs[:5]))
        tus.append((f, tu))
    for f, tu in tus:
        _collect_structs(tu.cursor, f, facts, files)
        _collect_size_exprs(tu.cursor, f, facts, files)
        _collect_byte_copies(tu.cursor, f, facts, files)
        rng = _glm_clip_range(tu, list(args))
        if rng:
            facts.clip_range[tu.spelling] = rng
    _Flow(facts, [tu for _, tu in tus], files).run()
    facts.tus = [tu for _, tu in tus]           # for host_meaning (whole-host inference)
    facts.files = files
    return facts


GLM_ZO = glm_sig.FORCE_ZO


def _glm_clip_range(tu, args):
    """GLM builds projections for [-1,1] clip depth unless GLM_FORCE_DEPTH_ZERO_TO_ONE
    is defined before its first include. Returns "zero_to_one", "neg_one_to_one",
    or None when the file does not include GLM."""
    if any(a in (f"-D{GLM_ZO}", f"-D{GLM_ZO}=1") for a in args):
        return "zero_to_one"
    defined = False
    for c in tu.cursor.get_children():
        if c.kind == K.MACRO_DEFINITION and c.spelling == GLM_ZO:
            defined = True
        elif c.kind == K.INCLUSION_DIRECTIVE:
            inc = c.get_included_file()
            if inc is not None and "/glm/" in inc.name.replace("\\", "/"):
                return "zero_to_one" if defined else "neg_one_to_one"
    return None


GLM_PROJECTIONS = glm_sig.PROJECTIONS


def _preorder(root, files):
    """Pre-order walk that skips everything declared outside our files
    (system headers are most of a translation unit)."""
    resolved = {f.resolve() for f in files}
    stack = [root]
    while stack:
        c = stack.pop()
        if c is not root:
            if c.location.file is not None and \
                    Path(c.location.file.name).resolve() not in resolved:
                continue
            yield c
        stack.extend(reversed(list(c.get_children())))


def _in(c, files):
    return c.location.file is not None and Path(c.location.file.name).resolve() in \
        {f.resolve() for f in files}


def _collect_structs(root, f, facts, files):
    for c in _preorder(root, files):
        if c.kind in (K.STRUCT_DECL, K.CLASS_DECL) and c.is_definition() and \
                c.location.file and Path(c.location.file.name).resolve() == f.resolve():
            t = c.type
            fields = []
            for ch in c.get_children():
                if ch.kind == K.FIELD_DECL:
                    fields.append(HostField(ch.spelling, t.get_offset(ch.spelling) // 8,
                                            ch.type.get_size(), ch.type.spelling, _tag(ch)))
            tag = _tag(c)
            facts.structs[c.spelling] = HostStruct(c.spelling, t.get_size(), fields,
                                                   tag.get("block"), _where(c))


_GLM_VEC = __import__("re").compile(r"glm::(?:vec<(\d), |tvec(\d)<)(float|double|int|unsigned int|bool)")
_SCALAR_KIND = {"float": "float", "double": "float", "int": "int", "unsigned int": "uint",
                "uint32_t": "uint", "int32_t": "int", "bool": "uint"}


def _vec_shape(t):
    """(components, scalar kind) of a host vector element type: a GLM vector,
    or a C array of scalars (float[3]); None for anything else."""
    c = t.get_canonical()
    m = _GLM_VEC.search(c.spelling)
    if m:
        return int(m.group(1) or m.group(2)), _SCALAR_KIND.get(m.group(3), m.group(3))
    if c.kind == ci.TypeKind.CONSTANTARRAY:
        e = c.element_type.get_canonical().spelling
        if e in _SCALAR_KIND and c.element_count > 1:
            return c.element_count, _SCALAR_KIND[e]
    return None


def _sizeof_type(e):
    """The type argument of a `sizeof(T)` expression (through parens/casts), or None."""
    while e is not None and e.kind in (K.UNEXPOSED_EXPR, K.PAREN_EXPR, K.CSTYLE_CAST_EXPR,
                                       K.CXX_STATIC_CAST_EXPR, K.CXX_FUNCTIONAL_CAST_EXPR):
        kids = [k for k in e.get_children() if k.kind.is_expression()]
        e = kids[0] if len(kids) == 1 else None
    if e is None or e.kind != K.CXX_UNARY_EXPR:
        return None
    toks = [t.spelling for t in e.get_tokens()][:1]
    if toks != ["sizeof"]:
        return None
    kids = list(e.get_children())
    refs = [k for k in kids if k.kind == K.TYPE_REF]
    if refs:
        return refs[-1].type
    exprs = [k for k in kids if k.kind.is_expression()]
    return exprs[0].type if exprs else None


BYTE_COPY = {"memcpy", "memmove", "vkCmdUpdateBuffer", "vkCmdPushConstants", "vkCmdPushConstants2",
             "vkCmdPushConstants2KHR"}


def _record_name(t):
    t = t.get_canonical()
    while t.kind in (ci.TypeKind.POINTER, ci.TypeKind.LVALUEREFERENCE, ci.TypeKind.CONSTANTARRAY,
                     ci.TypeKind.INCOMPLETEARRAY):
        t = (t.get_pointee() if t.kind in (ci.TypeKind.POINTER, ci.TypeKind.LVALUEREFERENCE)
             else t.element_type).get_canonical()
    if t.kind == ci.TypeKind.RECORD:
        d = t.get_declaration()
        return d.spelling if d is not None else None
    return None


def _collect_byte_copies(root, f, facts, files):
    """Host record types whose object bytes reach memory as they are: an
    argument of memcpy / memmove / vkCmdUpdateBuffer / vkCmdPushConstants
    (the object, its address, or sizeof(T)), or the target of a pointer cast
    (T* p = (T*)mapped). Only such a struct is a byte image of a shader block;
    a struct whose fields are copied one by one is not."""
    for c in _preorder(root, files):
        if c.kind == K.CALL_EXPR and c.spelling in BYTE_COPY:
            for arg in list(c.get_arguments()):
                for e in arg.walk_preorder():
                    if e.kind in (K.DECL_REF_EXPR, K.MEMBER_REF_EXPR, K.ARRAY_SUBSCRIPT_EXPR):
                        n = _record_name(e.type)
                    elif e.kind == K.CXX_UNARY_EXPR:
                        t = _sizeof_type(e)
                        n = _record_name(t) if t is not None else None
                    else:
                        continue
                    if n:
                        facts.copied.add(n)
        elif c.kind in (K.CSTYLE_CAST_EXPR, K.CXX_REINTERPRET_CAST_EXPR, K.CXX_STATIC_CAST_EXPR):
            if c.type.get_canonical().kind == ci.TypeKind.POINTER:
                n = _record_name(c.type)
                if n:
                    facts.copied.add(n)


def _collect_size_exprs(root, f, facts, files):
    """`sizeof(T) * n` with T a vector type: an element size used to size a
    buffer. Matched against shader array strides in check_layouts."""
    for c in _preorder(root, files):
        if c.kind != K.BINARY_OPERATOR or not c.location.file or \
                Path(c.location.file.name).resolve() != f.resolve():
            continue
        kids = list(c.get_children())
        if len(kids) != 2:
            continue
        ops = [t.spelling for t in c.get_tokens()]
        n0 = len(list(kids[0].get_tokens()))
        if n0 >= len(ops) or ops[n0] != "*":
            continue
        for e in kids:
            t = _sizeof_type(e)
            shape = _vec_shape(t) if t is not None else None
            if shape and t.get_size() > 0:
                facts.size_exprs.append((shape[0], shape[1], t.get_size(), t.spelling, _where(c)))


class _Flow:
    def __init__(self, facts, tus, files):
        self.facts, self.tus, self.files = facts, tus, files
        self.tags = {}         # decl key -> (tag, origin)
        self.dvalues = {}      # decl key -> set of numbers ("?" = unknown)
        self.final = False     # unknown values are recorded only in the last pass

    def key(self, c):
        if c is None:
            return None
        if c.kind == K.FIELD_DECL:
            return ("field", c.semantic_parent.spelling, c.spelling)
        return ("decl", c.get_usr() or c.spelling)

    def num(self, e, depth=0):
        """Numeric value of a host expression when it is a literal or a copy of
        a variable/field with one known value; None otherwise."""
        if e is None or depth > 10:
            return None
        k = e.kind
        if k in (K.FLOATING_LITERAL, K.INTEGER_LITERAL):
            toks = [t.spelling for t in e.get_tokens()]
            try:
                return float(toks[0].rstrip("fFlLuU")) if toks else None
            except ValueError:
                return None
        if k == K.CXX_BOOL_LITERAL_EXPR:
            return 1.0 if [t.spelling for t in e.get_tokens()][:1] == ["true"] else 0.0
        if k in (K.DECL_REF_EXPR, K.MEMBER_REF_EXPR):
            k = self.key(e.referenced)
            vals = self.facts.values.get(k[1:]) if k and k[0] == "field" else self.dvalues.get(k)
            if vals and len(vals) == 1 and "?" not in vals:
                return next(iter(vals))
            return self._clang_eval(e)
        kids = list(e.get_children())
        if k == K.CONDITIONAL_OPERATOR and len(kids) == 3:
            c = self.num(kids[0], depth + 1)
            return None if c is None else self.num(kids[1] if c else kids[2], depth + 1)
        if k == K.UNARY_OPERATOR and len(kids) == 1:
            v = self.num(kids[0], depth + 1)
            op = [t.spelling for t in e.get_tokens()][:1]
            if v is not None and op == ["!"]:
                return 0.0 if v else 1.0
            if v is not None and op == ["-"]:
                return -v
        if k == K.BINARY_OPERATOR and len(kids) == 2:
            a, b = self.num(kids[0], depth + 1), self.num(kids[1], depth + 1)
            op = self._binop(e)
            if a is not None and b is not None:
                f = {"==": lambda: float(a == b), "!=": lambda: float(a != b), "*": lambda: a * b,
                     "+": lambda: a + b, "-": lambda: a - b, "/": lambda: a / b if b else None,
                     "&&": lambda: float(bool(a and b)), "||": lambda: float(bool(a or b))}.get(op)
                if f:
                    return f()
            return self._clang_eval(e)
        if k in (K.UNEXPOSED_EXPR, K.PAREN_EXPR, K.CXX_FUNCTIONAL_CAST_EXPR,
                 K.CSTYLE_CAST_EXPR, K.CXX_STATIC_CAST_EXPR) and len(kids) == 1:
            return self.num(kids[0], depth + 1)
        return None

    @staticmethod
    def _clang_eval(e):
        """Constant-fold with clang (enums, constexpr, static const). Shares
        host_meaning's typed ctypes binding: the two must not declare
        clang_Cursor_Evaluate differently in one process (a mismatched result
        type truncates the EvalResult pointer and crashes the next call)."""
        from . import host_meaning
        return host_meaning._evaluate(e)

    def store_value(self, k, rhs):
        if k is None:
            return False
        v = self.num(rhs)
        if v is None and not self.final:
            return False                # may become known once other values settle
        if k[0] != "field":
            vals = self.dvalues.setdefault(k, set())
            new = v if v is not None else "?"
            if new in vals:
                return False
            vals.add(new)
            return True
        vals = self.facts.values.setdefault(k[1:], set())
        new = v if v is not None else "?"
        if new in vals:
            return False
        vals.add(new)
        return True

    def seed(self):
        for tu in self.tus:
            for c in _preorder(tu.cursor, self.files):
                if c.kind == K.FIELD_DECL and "=" in [t.spelling for t in c.get_tokens()]:
                    init = [x for x in c.get_children()
                            if x.kind not in (K.TYPE_REF, K.ANNOTATE_ATTR, K.TEMPLATE_REF,
                                              K.NAMESPACE_REF)]
                    if init:
                        self.store_value(self.key(c), init[-1])
                if c.kind in (K.FIELD_DECL, K.VAR_DECL, K.PARM_DECL, K.FUNCTION_DECL,
                              K.CXX_METHOD):
                    t = {k: v for k, v in _tag(c).items() if k in ("unit", "frame", "color")}
                    if t:
                        name = (f"{c.semantic_parent.spelling}.{c.spelling}"
                                if c.kind == K.FIELD_DECL else c.spelling)
                        self.tags[self.key(c)] = (t, f"'{name}' (COV at {_where(c)})")

    def expr_tag(self, e, depth=0):
        """(tag, origin) carried by an expression, or None."""
        if e is None or depth > 30:
            return None
        k = e.kind
        if k in (K.INTEGER_LITERAL, K.FLOATING_LITERAL):
            return None
        if k in (K.MEMBER_REF_EXPR, K.DECL_REF_EXPR):
            ref = e.referenced
            got = self.tags.get(self.key(ref))
            if got:
                return got
            if k == K.MEMBER_REF_EXPR:
                return None
        if k == K.CALL_EXPR:
            ref = e.referenced
            if ref is not None and ref.kind in (K.FUNCTION_DECL, K.CXX_METHOD, K.FUNCTION_TEMPLATE):
                conv = ANGLE_CONVERT.get(ref.spelling)
                if conv and _ns(ref) in ("glm", ""):
                    return ({"unit": conv}, f"{_ns(ref) + '::' if _ns(ref) else ''}{ref.spelling}() "
                                            f"({_where(e)})")
                if _ns(ref) == "glm" and ref.spelling in ("inverse", "transpose", "affineInverse"):
                    # the inverse of a projection undoes the same clip depth range
                    return self._merge(e, depth, arith="operator*")
                got = self.tags.get(self.key(ref)) or self.projection_tag(e, ref)
                if got:
                    return got
                if ref.spelling.startswith("operator") and ref.spelling not in ("operator=",):
                    return self._merge(e, depth, arith=ref.spelling)
                return None
            return self._merge(e, depth)       # constructors, conversions
        if k == K.BINARY_OPERATOR:
            op = self._binop(e)
            if op in ("*", "/"):
                return self._angle_scale(e, op, depth)
            return self._merge(e, depth)
        return self._merge(e, depth)

    def _angle_scale(self, e, op, depth):
        """Scaling drops units (that is how conversions are written), except an
        angle scaled by a number that is not a degree/radian factor: fov * 0.5
        is still in degrees."""
        kids = list(e.get_children())
        if len(kids) != 2:
            return None
        for i, ch in enumerate(kids):
            t = self.expr_tag(ch, depth + 1)
            if not t or t[0].get("unit") not in ("deg", "rad"):
                continue
            if op == "/" and i == 1:
                return None
            f = self.num(kids[1 - i])
            if not isinstance(f, (int, float)) or f == 0:
                return None
            f = abs(f) if op == "*" else 1.0 / abs(f)
            if std_trig.scale_element(f, 0.01) != 0:      # a degree <-> radian conversion
                return None
            return ({"unit": t[0]["unit"]}, t[1])
        return None

    def check_angle_call(self, c):
        """A trig function or a glm projection takes radians."""
        ref = c.referenced
        if ref is None:
            return
        name, ns = ref.spelling, _ns(ref)
        base = name.split("_")[0]
        if name in TRIG and ns in ("std", "", "glm"):
            args = list(c.get_arguments() or [])[:1]
        elif ns == "glm" and base in glm_sig.ANGLE_PARAMS:
            got = list(c.get_arguments() or [])
            args = [got[i] for i in glm_sig.ANGLE_PARAMS[base] if i < len(got)]
        else:
            return
        for a in args:
            t = self.expr_tag(a)
            if t and t[0].get("unit") == "deg":
                self.facts.angle_uses.append((f"{ns + '::' if ns else ''}{name}", _where(c), t[1]))

    def projection_tag(self, e, ref):
        name = ref.spelling
        ns = ref.semantic_parent.spelling if ref.semantic_parent else ""
        if ns != "glm":
            return None
        base = name.rsplit("_", 1)
        if len(base) == 2 and base[1] in ("ZO", "NO") and glm_sig.is_projection(name):
            rng = "zero_to_one" if glm_sig.depth_range(name, False) == 0 else "neg_one_to_one"
            why = f"glm::{name}"
        elif glm_sig.is_projection(name):
            rng = self.facts.clip_range.get(e.translation_unit.spelling)
            if rng is None:
                return None
            why = (f"glm::{name} with {GLM_ZO} "
                   f"{'in effect' if rng == 'zero_to_one' else 'not defined before GLM is included'}")
        else:
            return None
        return ({"depth_range": rng}, f"{why} ({_where(e)})")

    def _merge(self, e, depth, arith=None):
        if arith in ("operator*", "operator/", "operator*=", "operator/="):
            # scaling drops units, but a projection times a view matrix is still
            # a projection: keep only the clip-depth convention
            for ch in e.get_children():
                t = self.expr_tag(ch, depth + 1)
                if t and "depth_range" in t[0]:
                    return ({"depth_range": t[0]["depth_range"]}, t[1])
            return None
        for ch in e.get_children():
            t = self.expr_tag(ch, depth + 1)
            if t:
                return t
        return None

    def _binop(self, e):
        kids = list(e.get_children())
        if len(kids) != 2:
            return None
        end = kids[0].extent.end.offset
        for tok in e.get_tokens():
            if tok.extent.start.offset >= end and tok.kind == ci.TokenKind.PUNCTUATION:
                return tok.spelling
        return None

    def run(self):
        self.seed()
        for _ in range(6):
            changed = False
            for tu in self.tus:
                changed |= self._walk(tu.cursor, None)
            if not changed:
                break
        self.final = True
        for tu in self.tus:
            self._walk(tu.cursor, None)
        for tu in self.tus:
            for c in _preorder(tu.cursor, self.files):
                if c.kind == K.CALL_EXPR:
                    self.check_angle_call(c)

    def _walk(self, c, fn):
        changed = False
        for ch in c.get_children():
            if ch.location.file is not None and not _in(ch, self.files):
                continue
            f = ch if ch.kind in (K.FUNCTION_DECL, K.CXX_METHOD) else fn
            changed |= self.visit(ch, f)
            changed |= self._walk(ch, f)
        return changed

    def assign(self, target, rhs, where):
        if target.kind == K.UNEXPOSED_EXPR:
            kids = list(target.get_children())
            target = kids[0] if kids else target
        changed = False
        if target.kind == K.MEMBER_REF_EXPR and target.referenced is not None:
            changed = self.store_value(self.key(target.referenced), rhs)
        t = self.expr_tag(rhs)
        if not t:
            return changed
        ref = target.referenced
        k = self.key(ref)
        if k is None or k in self.tags:
            return False
        tag, origin = t
        self.tags[k] = (tag, origin)
        if ref.kind == K.FIELD_DECL:
            self.facts.flows.append((ref.semantic_parent.spelling, ref.spelling, tag,
                                     where, origin))
        return True

    def visit(self, c, fn):
        changed = False
        if c.kind == K.VAR_DECL:
            kids = [x for x in c.get_children() if x.kind not in (K.TYPE_REF, K.NAMESPACE_REF)]
            if kids and kids[-1].kind == K.INIT_LIST_EXPR:
                rec = c.type.get_canonical().get_declaration()
                fields = [f for f in rec.get_children() if f.kind == K.FIELD_DECL]
                for f, v in zip(fields, kids[-1].get_children()):
                    changed |= self.store_value(self.key(f), v)
            elif kids:
                changed |= self.store_value(self.key(c), kids[-1])
        if c.kind == K.CALL_EXPR and c.referenced is not None and \
                c.referenced.spelling == "radians" and _ns(c.referenced) == "glm":
            # glm::radians(x) says x is in degrees, with no annotation needed
            args = list(c.get_arguments() or [])
            ref = _strip(args[0]).referenced if args else None
            if ref is not None and _strip(args[0]).kind in (K.MEMBER_REF_EXPR, K.DECL_REF_EXPR) \
                    and ref.kind in (K.FIELD_DECL, K.VAR_DECL, K.PARM_DECL) \
                    and self.key(ref) not in self.tags:
                name = (f"{ref.semantic_parent.spelling}.{ref.spelling}"
                        if ref.kind == K.FIELD_DECL else ref.spelling)
                self.tags[self.key(ref)] = ({"unit": "deg"},
                                            f"'{name}', which {_where(c)} converts with glm::radians")
                changed = True
        if c.kind == K.CALL_EXPR and c.referenced is not None and \
                c.referenced.kind in (K.FUNCTION_DECL, K.CXX_METHOD):
            for prm, arg in zip(c.referenced.get_arguments() or [], c.get_arguments() or []):
                changed |= self.store_value(self.key(prm), arg)
        return self._visit(c, fn) or changed

    def _visit(self, c, fn):
        if c.kind == K.VAR_DECL:
            kids = [x for x in c.get_children() if x.kind != K.TYPE_REF]
            if kids and self.key(c) not in self.tags:
                t = self.expr_tag(kids[-1])
                if t:
                    self.tags[self.key(c)] = t
                    return True
        elif c.kind == K.BINARY_OPERATOR and self._binop(c) == "=":
            lhs, rhs = list(c.get_children())
            return self.assign(lhs, rhs, _where(c))
        elif c.kind == K.CALL_EXPR and c.spelling == "operator=":
            kids = list(c.get_children())
            if len(kids) == 3:
                return self.assign(kids[0], kids[2], _where(c))
        elif c.kind == K.RETURN_STMT:
            kids = list(c.get_children())
            if fn is not None and kids and self.key(fn) not in self.tags:
                t = self.expr_tag(kids[0])
                if t:
                    self.tags[self.key(fn)] = t
                    return True
        return False


def _strip(e):
    while e is not None and e.kind == K.UNEXPOSED_EXPR:
        kids = list(e.get_children())
        if len(kids) != 1:
            break
        e = kids[0]
    return e


def _ns(ref):
    p = ref.semantic_parent
    return p.spelling if p is not None and p.kind == K.NAMESPACE else ""


ANGLE_CONVERT = {"radians": "rad", "degrees": "deg"}
TRIG = std_trig.ANGLE_IN


def check_angle_units(facts, report=None):
    """Angle units as a labelled dimension (Z: powers of pi/180): the value a
    degree annotation or a glm::radians argument starts from is 'degrees';
    the trig / projection signature asserts its argument is 'radians'."""
    report = report or Report()
    s = MeaningSolver(report)
    seen = set()
    for call, where, origin in facts.angle_uses:
        if (call, where) in seen:
            continue
        seen.add((call, where))
        node = ("angle-arg", call, where)
        s.set(node, "angle", "deg", origin, where)
        s.set(node, "angle", "rad", f"{call}, which takes an angle in radians (convert with "
              "glm::radians() first)", where)
    s.finish()
    return report


# ---------------------------------------------------------------- checking
def spirv_blocks(modules):
    """{struct name: {"members": {name: offset}, "stride": array stride or None}}"""
    out = {}
    for m in modules:
        for tid, t in m.types.items():
            if t.op != "OpTypeStruct":
                continue
            offs = {}
            for i in range(len(t.args)):
                o = m.member_decos.get((tid, i), {}).get("Offset")
                if o is not None:
                    offs[m.member_names.get((tid, i), str(i))] = o[0]
            if not offs:
                continue
            stride = None
            for aid, at in m.types.items():
                if at.op in ("OpTypeRuntimeArray", "OpTypeArray") and at.args[0] == tid:
                    s = m.decos.get(aid, {}).get("ArrayStride")
                    if s:
                        stride = s[0]
            out[m.struct_name(tid)] = {"members": offs, "stride": stride}
    return out


def name_tokens(name):
    """Normalised words of a member name, so that the spellings host and shader
    code use for one field meet: camelCase / PascalCase / snake_case split and
    lower-cased, and a one-letter scope prefix dropped (Shadertoy 'iTime',
    'uColor', 'm_proj', 'g_view': i / u / m / g)."""
    import re
    parts = []
    for chunk in re.split(r"[_\W]+", name or ""):
        parts += re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", chunk)
    parts = [p.lower() for p in parts if p]
    if len(parts) > 1 and parts[0] in ("i", "u", "m", "g", "s"):
        parts = parts[1:]
    return tuple(parts)


def _match_block(hs, blocks, min_members=4, frac=0.6):
    """(shader struct, {host field name: shader member name}) for the shader
    struct whose members the host struct's fields name, by normalised name
    (most members, and at least `frac` of the shader struct's members and of
    the host's fields). Used when the two structs are named differently."""
    hn = {}
    for f in hs.fields:
        hn.setdefault(name_tokens(f.name), f.name)
    best = None
    for bname, b in blocks.items():
        pairs = {}
        for mem in b["members"]:
            t = name_tokens(mem)
            if t in hn and hn[t] not in pairs:
                pairs[hn[t]] = mem
        k = len(pairs)
        if k < min_members or k < frac * len(b["members"]) or k < frac * len(hs.fields):
            continue
        if best is None or k > len(best[1]):
            best = (bname, pairs)
    return best


def check_layouts(facts, modules, report=None):
    report = report or Report()
    blocks = spirv_blocks(modules)
    used = {}
    for m in modules:
        for st, mems in spirv.used_members(m).items():
            if st in used and used[st] is None:
                continue
            used[st] = None if mems is None else (used.get(st) or set()) | mems
    for hs in facts.structs.values():
        bname = hs.block or (hs.name if hs.name in blocks else None)
        names = None                     # host field -> shader member (None: same spelling)
        if bname is None or bname not in blocks:
            if hs.name not in getattr(facts, "copied", ()):
                continue                 # fields copied one by one: not a byte image
            got = _match_block(hs, blocks)
            if got is None:
                continue
            bname, names = got
        b = blocks[bname]
        base = 0
        if names is not None:
            # a host struct matched by name may be a sub-range of the block,
            # copied at a base offset (memcpy(ubo + 128, &params, ...)): the
            # offset most matched fields agree on is that base (0 on a tie)
            from collections import Counter
            deltas = Counter(b["members"][names[f.name]] - f.offset for f in hs.fields
                             if names.get(f.name) in b["members"])
            if deltas:
                top = max(deltas.values())
                base = 0 if deltas.get(0, 0) == top else deltas.most_common(1)[0][0]
        for f in hs.fields:
            sname = f.name if names is None else names.get(f.name)
            so = b["members"].get(sname) if sname else None
            if so is None:
                continue
            u = used.get(bname, set()) if bname in used else None
            if u is not None and sname not in u:
                continue                # no shader reads this member (padding): harmless
            if so != f.offset + base:
                same = "" if names is None else f" (member '{sname}', matched by name" + \
                    (f"; struct copied at byte {base})" if base else ")")
                report.add("structure", "structure.host-layout",
                           f"field '{f.name}' is at byte {f.offset} in host struct "
                           f"'{hs.name}' but at byte {so} in shader block '{bname}'{same}; the "
                           "GPU reads shifted data.", hs.where,
                           ["match the shader layout (std430 aligns vec3 to 16 bytes), "
                            "or generate the host struct from the shader"])
        if b["stride"] and b["stride"] != hs.size:
            report.add("structure", "structure.host-layout",
                       f"host struct '{hs.name}' is {hs.size} bytes but each element of "
                       f"'{bname}' in the shader array is {b['stride']} bytes; every element "
                       "after the first is misread.", hs.where)
    _check_vector_strides(facts, modules, report)
    return report


def shader_vector_arrays(modules):
    """Arrays of vectors inside buffer blocks: [(block, member, comps, scalar kind, stride)]."""
    out = []
    for m in modules:
        for tid, t in m.types.items():
            if t.op != "OpTypeStruct":
                continue
            d = m.decos.get(tid, {})
            if "Block" not in d and "BufferBlock" not in d:
                continue
            for i, mt in enumerate(t.args):
                at = m.types.get(mt)
                if at is None or at.op not in ("OpTypeRuntimeArray", "OpTypeArray"):
                    continue
                stride = m.decos.get(mt, {}).get("ArrayStride")
                et = m.types.get(at.args[0])
                if not stride or et is None or et.op != "OpTypeVector":
                    continue
                st = m.types.get(et.args[0])
                kind = {"OpTypeFloat": "float"}.get(st.op) if st is not None else None
                if st is not None and st.op == "OpTypeInt":
                    kind = "int" if st.args[1] else "uint"
                out.append((m.struct_name(tid), m.member_names.get((tid, i), str(i)),
                            et.args[1], kind, stride[0]))
    return out


def _check_vector_strides(facts, modules, report):
    """A buffer sized as sizeof(vector) * n on the host, for a shader array of
    the same vector type whose ArrayStride is larger (std430/std140 round a
    3-component vector up to 16 bytes): the buffer is too small and the last
    elements are out of bounds. Linked by element type only, so a warning."""
    arrays = shader_vector_arrays(modules)
    seen = set()
    for comps, kind, size, tname, where in facts.size_exprs:
        for block, mem, sc, sk, stride in arrays:
            if (sc, sk) != (comps, kind) or stride <= size or (block, mem, where) in seen:
                continue
            seen.add((block, mem, where))
            short = round(100 * (1 - size / stride))
            report.add("structure", "structure.host-layout",
                       f"a buffer is sized at {where} as sizeof({tname}) = {size} bytes per element, but "
                       f"'{block}.{mem}' in the shader is an array of the same vector type with a stride of "
                       f"{stride} bytes; if this buffer backs it, it is {short}% too small and the last "
                       "elements are written or read out of bounds.", where,
                       [f"size the buffer with the shader's ArrayStride ({stride} bytes per element), or "
                        "use a 4-component host type"],
                       severity="warning")


def check_clip_depth(facts, modules, report=None, api_range="zero_to_one"):
    """A projection built for one clip-depth range stored into a shader matrix
    that maps to clip space on an API with the other range (Vulkan: [0,1])."""
    report = report or Report()
    clip = {}
    for m in modules:
        for (st, mem), tag in m.meta.get("fields", {}).items():
            if tag.get("to") == "clip":
                clip[(st, mem)] = tag
    host_to_block = {h.name: (h.block or h.name) for h in facts.structs.values()}
    seen = set()
    for st, mem, tag, where, origin in facts.flows:
        rng = tag.get("depth_range")
        b = host_to_block.get(st, st)
        if rng and rng != api_range and (b, mem) in clip and (b, mem) not in seen:
            seen.add((b, mem))
            pretty = {"zero_to_one": "[0,1]", "neg_one_to_one": "[-1,1]"}
            report.add("meaning", "meaning.convention-mismatch",
                       f"'{st}.{mem}' is stored at {where} from {origin}, which maps depth to "
                       f"{pretty[rng]}, but the shader uses it as a transform to clip space "
                       f"and this API clips depth to {pretty[api_range]}: the near half of the "
                       "depth range is lost and depth comparisons shift.",
                       where, [f"define {GLM_ZO} before the first GLM include (e.g. as a "
                               "compile definition), or use the _ZO projection functions"])
    return report


def host_values(facts, modules):
    """{(shader struct, member): number} for fields every host store sets to one
    known number (e.g. a gamma of 2.2 from a settings default)."""
    blocks = _shader_structs(modules)
    layout = spirv_blocks(modules)
    host_to_block = {h.name: (h.block or h.name) for h in facts.structs.values()}
    out = {}
    for (st, mem), vals in facts.values.items():
        b = host_to_block.get(st, st)
        if b not in blocks or len(vals) != 1 or "?" in vals:
            continue
        smem = mem
        h = facts.structs.get(st)
        if b in layout and mem not in layout[b]["members"] and h:
            # names differ between host and shader: pair the fields by byte offset
            off = next((f.offset for f in h.fields if f.name == mem), None)
            smem = next((n for n, o in layout[b]["members"].items() if o == off), None)
        if smem is not None:
            out[(b, smem)] = (next(iter(vals)), f"{st}.{mem}")
    return out


def _shader_structs(modules):
    names = set(spirv_blocks(modules))
    for m in modules:
        names |= {st for st, _ in m.meta.get("fields", {})}
        names |= {st for st, _ in m.meta.get("varyings", {}).values()}
    return names


def meaning_seeds(facts, modules):
    """Turn host flows into seeds on shader struct members."""
    blocks = _shader_structs(modules)
    host_to_block = {h.name: (h.block or h.name) for h in facts.structs.values()}
    seeds = []
    members = {b: v["members"] for b, v in spirv_blocks(modules).items()}
    norm = lambda n: n.replace("_", "").lower()
    for st, mem, tag, where, origin in facts.flows:
        b = host_to_block.get(st, st)
        if b in members and mem not in members[b]:
            # projection_inverse on the host is projectionInverse in the shader
            mem = next((x for x in members[b] if norm(x) == norm(mem)), mem)
        if b in blocks:
            seeds.append(HostSeed(tag, f"host data from {origin}, stored into "
                                       f"{st}.{mem} at {where}", struct=b, member=mem,
                                  where=where))
    for h in facts.structs.values():
        b = h.block or h.name
        if b not in blocks:
            continue
        for f in h.fields:
            t = {k: v for k, v in f.tag.items() if k in ("unit", "frame", "color")}
            if t:
                seeds.append(HostSeed(t, f"the host declaration of {h.name}.{f.name} "
                                         f"({h.where})", struct=b, member=f.name))
    return seeds


def vec3_tails(module, tid, off=0, deco=None, depth=0, out=None):
    """Byte offsets of the 4-byte word after each 3-component vector of a block
    whose next member or array element starts 16 bytes after the vector (std140
    / std430 round vec3 up to 16 bytes there). A host that mirrors such a member
    with a 4-component array writes its 4th component into that word: the
    offsets agree and the shader never reads the word, so it is not a layout
    error."""
    m = module
    out = set() if out is None else out
    if depth > 8:
        return out
    k = m.tkind(tid)
    if k == "struct":
        for i, mt in enumerate(m.types[tid].args):
            md = m.member_decos.get((tid, i), {})
            vec3_tails(m, mt, off + md.get("Offset", [0])[0], md, depth + 1, out)
    elif k == "array":
        n = m.consts.get(m.types[tid].args[1])
        stride = m.decos.get(tid, {}).get("ArrayStride", [0])[0]
        if isinstance(n, int) and stride:
            for i in range(min(n, 4096)):
                vec3_tails(m, m.types[tid].args[0], off + i * stride, deco, depth + 1, out)
    elif k == "vector" and m.types[tid].args[1] == 3:
        out.add(off + 12)
    return out
