"""Byte ranges a shader can access in its buffer descriptors.

A descriptor covers a whole buffer range, but a shader often reads or writes
only a part of it that the command selects: a ring half chosen by a push
constant (``data[pc.base + i]``), one dispatch chunk's elements
(``data[gl_GlobalInvocationID.x + pc.first]``), the instances of a direct draw
(``data[gl_InstanceIndex]``). Without that, a host write to one half of a ring
races a GPU read of the other half.

``binding_exprs(module, entry)`` finds, once per shader, every access to every
non-arrayed uniform / storage buffer binding and builds an expression for the
byte offset it touches, in terms of

* constants (``OpConstant``; specialization constants are unknown),
* push-constant words (read from the command's push-constant bytes),
* compute / vertex builtins (bounded by the command's group counts, local
  size, vertex and instance ranges),
* anything else: unknown, bounded only by its integer type.

``Ctx.eval`` evaluates these with interval arithmetic for one command. A
binding is refined only when every access to it is understood: pointers that
escape into function calls, phis, selects, pointer casts or stores make the
binding "whole" (``None``).

Assumptions (stated in the paper): index arithmetic does not wrap around
(the usual no-overflow assumption of range analyses), and accesses stay within
the descriptor's range (outside it they are undefined behaviour without
robustness features), so every range is clipped to the descriptor range.
"""
from . import spirv as _spirv

INF = float("inf")
_CHAINS = ("OpAccessChain", "OpInBoundsAccessChain")
_ATOMIC_SIZE_OPS = None

# SPIR-V BuiltIn numbers
BI = {24: "NumWorkgroups", 26: "WorkgroupId", 27: "LocalInvocationId", 28: "GlobalInvocationId",
      29: "LocalInvocationIndex", 42: "VertexIndex", 43: "InstanceIndex"}


def _int_range(m, t):
    d = m.types.get(t)
    if d is None or d.op != "OpTypeInt":
        return (-INF, INF)
    w, signed = d.args[0], d.args[1]
    return (-(1 << (w - 1)), (1 << (w - 1)) - 1) if signed else (0, (1 << w) - 1)


def type_size(m, t, mstride=None, depth=0):
    """Bytes a value of type t occupies in an explicitly laid out block (None: unbounded/unknown)."""
    d = m.types.get(t)
    if d is None or depth > 32:
        return None
    op = d.op
    if op in ("OpTypeInt", "OpTypeFloat"):
        return d.args[0] // 8
    if op == "OpTypeBool":
        return 4
    if op == "OpTypeVector":
        s = type_size(m, d.args[0], None, depth + 1)
        return None if s is None else s * d.args[1]
    if op == "OpTypeMatrix":
        col = type_size(m, d.args[0], None, depth + 1)
        if col is None:
            return None
        n = d.args[1]
        if mstride:
            # column- or row-major: at most n * stride (+ the last column's size)
            return max(n * mstride, col)
        return col * n
    if op == "OpTypeArray":
        n = m.consts.get(d.args[1])
        dn = m.defs.get(d.args[1])
        st = (m.decos.get(t) or {}).get("ArrayStride")
        if not isinstance(n, int) or dn is None or dn.op != "OpConstant":
            return None
        if st:
            return n * st[0]
        es = type_size(m, d.args[0], mstride, depth + 1)
        return None if es is None else n * es
    if op == "OpTypeRuntimeArray":
        return None
    if op == "OpTypeStruct":
        end = 0
        for i, mt in enumerate(d.args):
            md = m.member_decos.get((t, i)) or {}
            off = md.get("Offset", [None])[0]
            if off is None:
                return None
            ms = md.get("MatrixStride", [None])[0]
            s = type_size(m, mt, ms, depth + 1)
            if s is None:
                return None
            end = max(end, off + s)
        return end
    return None


class _Builder:
    """Expression trees for SSA ids of one module (shared by all entries)."""

    def __init__(self, m, reach):
        self.m = m
        self.memo = {}
        self.busy = set()
        self.body = [i for f in reach for i in m.functions[f]]
        # stores into each pointer (function / private variables)
        self.stores = {}
        self.calls = {}             # callee -> [call instructions]
        self.returns = {}           # function -> [returned value ids]
        for f in reach:
            for i in m.functions[f]:
                if i.op == "OpStore":
                    self.stores.setdefault(i.args[0], []).append(i.args[1])
                elif i.op == "OpFunctionCall":
                    self.calls.setdefault(i.args[0], []).append(i)
                elif i.op == "OpReturnValue":
                    self.returns.setdefault(f, []).append(i.args[0])
        self.param_of = {}          # param id -> (function, index)
        for f in reach:
            for k, p in enumerate(m.params.get(f, [])):
                self.param_of[p] = (f, k)
        self.chained_into = set()   # variables written through an access chain
        for i in self.body:
            if i.op in _CHAINS and i.args:
                self.chained_into.add(i.args[0])
        self.builtin = {v: BI.get(d["BuiltIn"][0]) for v, d in m.decos.items() if "BuiltIn" in d}
        self.pc_vars = {i.id for i in m.defs.values() if i.op == "OpVariable" and i.args and i.args[0] == 9}

    def unk(self, t):
        return ("unk",) + _int_range(self.m, t)

    def expr(self, vid):
        if vid in self.memo:
            return self.memo[vid]
        if vid in self.busy:
            d = self.m.defs.get(vid)
            return self.unk(d.type if d is not None else None)
        self.busy.add(vid)
        try:
            e = self._expr(vid)
        finally:
            self.busy.discard(vid)
        self.memo[vid] = e
        return e

    def _expr(self, vid):
        m = self.m
        d = m.defs.get(vid)
        if d is None:
            if vid in self.param_of:        # a by-value parameter: join of the call sites' arguments
                f, k = self.param_of[vid]
                args = [c.args[1 + k] for c in self.calls.get(f, []) if len(c.args) > 1 + k]
                return ("join",) + tuple(self.expr(a) for a in args) if args else ("unk", -INF, INF)
            return ("unk", -INF, INF)
        op, a = d.op, d.args
        if op == "OpConstant":
            t = m.types.get(d.type)
            if t is not None and t.op == "OpTypeInt":
                v = a[0] if len(a) == 1 else a[0] | (a[1] << 32)
                w, signed = t.args[0], t.args[1]
                if signed and v >= 1 << (w - 1):
                    v -= 1 << w
                return ("const", v)
            return self.unk(d.type)
        if op == "OpConstantNull":
            return ("const", 0)
        if op == "OpConstantComposite":
            return ("vec",) + tuple(self.expr(x) for x in a)
        if op == "OpFunctionParameter":
            f, k = self.param_of.get(vid, (None, None))
            args = [c.args[1 + k] for c in self.calls.get(f, []) if len(c.args) > 1 + k]
            return ("join",) + tuple(self.expr(x) for x in args) if args else self.unk(d.type)
        if op == "OpFunctionCall":
            rets = self.returns.get(a[0], [])
            return ("join",) + tuple(self.expr(x) for x in rets) if rets else self.unk(d.type)
        if op == "OpLoad":
            return self.load(a[0], d.type)
        if op in ("OpCopyObject", "OpBitcast", "OpUConvert", "OpSConvert", "OpSatConvertSToU", "OpSatConvertUToS"):
            return self.expr(a[0]) if m.tkind(d.type) in ("int", "vector") else self.unk(d.type)
        if op == "OpCompositeExtract":
            return ("extract", self.expr(a[0])) + tuple(a[1:])
        if op == "OpCompositeConstruct":
            return ("vec",) + tuple(self.expr(x) for x in a)
        if op == "OpVectorShuffle":
            return ("shuffle", self.expr(a[0]), self.expr(a[1])) + tuple(a[2:])
        if op == "OpPhi":
            return ("join",) + tuple(self.expr(a[k]) for k in range(0, len(a), 2))
        if op == "OpSelect":
            return ("join", self.expr(a[1]), self.expr(a[2]))
        if op in ("OpIAdd", "OpISub", "OpIMul", "OpUDiv", "OpSDiv", "OpUMod", "OpSMod", "OpSRem",
                  "OpShiftLeftLogical", "OpShiftRightLogical", "OpShiftRightArithmetic",
                  "OpBitwiseAnd", "OpBitwiseOr"):
            return ("op", op, self.expr(a[0]), self.expr(a[1]), _int_range(m, d.type))
        if op == "OpSNegate":
            return ("op", "OpISub", ("const", 0), self.expr(a[0]), _int_range(m, d.type))
        if op == "OpExtInst" and a and a[0] == m.glsl_ext:
            name = _spirv._grammar()[1].get(a[1])
            if name in ("UMin", "SMin", "UMax", "SMax") and len(a) >= 4:
                return ("op", name, self.expr(a[2]), self.expr(a[3]), _int_range(m, d.type))
            if name in ("UClamp", "SClamp") and len(a) >= 5:
                return ("op", "Max", ("op", "Min", self.expr(a[2]), self.expr(a[4]), (-INF, INF)),
                        self.expr(a[3]), _int_range(m, d.type))
        return self.unk(d.type)

    def load(self, ptr, t):
        """Value loaded from pointer ptr (type t)."""
        m = self.m
        d = m.defs.get(ptr)
        if d is not None and d.op == "OpVariable":
            sc = d.args[0]
            if self.builtin.get(ptr):
                return ("bi", self.builtin[ptr], None)
            if sc in (6, 7) and ptr not in self.chained_into:      # Private / Function: join of stores
                vals = list(self.stores.get(ptr, []))
                if len(d.args) > 1:
                    vals.append(d.args[1])
                elif sc == 6:
                    return self.unk(t)
                return ("join",) + tuple(self.expr(v) for v in vals) if vals else self.unk(t)
            return self.unk(t)
        if d is not None and d.op in _CHAINS and d.args:
            base = m.defs.get(d.args[0])
            if base is not None and base.op == "OpVariable":
                if self.builtin.get(d.args[0]) and len(d.args) == 2:
                    c = self._const(d.args[1])
                    if c is not None:
                        return ("bi", self.builtin[d.args[0]], c)
                if d.args[0] in self.pc_vars:
                    off = chain_offset(m, d.args[0], d.args[1:], self)
                    if off is not None and off[0] == "const":
                        tt = m.types.get(t)
                        if tt is not None and tt.op == "OpTypeInt":
                            return ("pc", off[1], tt.args[0] // 8, bool(tt.args[1]))
            return self.unk(t)
        if ptr in self.param_of:                # pointer parameter: what the call sites pass
            f, k = self.param_of[ptr]
            args = [c.args[1 + k] for c in self.calls.get(f, []) if len(c.args) > 1 + k]
            if args and ptr not in self.stores and ptr not in self.chained_into:
                return ("join",) + tuple(self.load(x, t) for x in args)
        return self.unk(t)

    def _const(self, vid):
        d = self.m.defs.get(vid)
        if d is not None and d.op == "OpConstant":
            return self.m.consts.get(vid)
        return None


def chain_offset(m, var, idxs, b):
    """("const", bytes) or an expression of the byte offset of an access chain
    from block variable var; None when the layout is unknown. Returns the
    offset expression; the accessed size is chain_size."""
    d = m.defs.get(var)
    t = m.pointee(d.type)
    terms = []
    parent_struct, parent_member = None, None
    for ix in idxs:
        td = m.types.get(t)
        if td is None:
            return None
        if td.op == "OpTypeStruct":
            c = b._const(ix)
            if not isinstance(c, int) or c >= len(td.args):
                return None
            off = (m.member_decos.get((t, c)) or {}).get("Offset")
            if off is None:
                return None
            terms.append(("const", off[0]))
            parent_struct, parent_member = t, c
            t = td.args[c]
        elif td.op in ("OpTypeArray", "OpTypeRuntimeArray"):
            st = (m.decos.get(t) or {}).get("ArrayStride")
            if not st:
                return None
            terms.append(("op", "OpIMul", b.expr(ix), ("const", st[0]), (-INF, INF)))
            t = td.args[0]
        elif td.op in ("OpTypeMatrix", "OpTypeVector"):
            # inside a matrix / vector: the whole of it (row-major layouts make the column offset unreliable)
            break
        else:
            return None
    e = ("const", 0)
    for x in terms:
        e = ("op", "OpIAdd", e, x, (-INF, INF))
    return _fold(e)


def _fold(e):
    if e[0] == "op" and e[2][0] == "const" and e[3][0] == "const":
        x, y = e[2][1], e[3][1]
        if e[1] == "OpIAdd":
            return ("const", x + y)
        if e[1] == "OpIMul":
            return ("const", x * y)
    if e[0] == "op":
        a, c = _fold(e[2]), _fold(e[3])
        if a[0] == "const" and c[0] == "const":
            return _fold(("op", e[1], a, c, e[4]))
        return ("op", e[1], a, c, e[4])
    return e


def _access_type_size(m, ptr_type, var, idxs, b):
    """Size of the value at the end of the chain (matrix/vector: whole)."""
    t = m.pointee(m.defs.get(var).type)
    ms = None
    for ix in idxs:
        td = m.types.get(t)
        if td is None:
            return None
        if td.op == "OpTypeStruct":
            c = b._const(ix)
            if not isinstance(c, int):
                return None
            ms = (m.member_decos.get((t, c)) or {}).get("MatrixStride", [None])[0]
            t = td.args[c]
        elif td.op in ("OpTypeArray", "OpTypeRuntimeArray"):
            t = td.args[0]
        else:
            break
    return type_size(m, t, ms)


def binding_exprs(m, entry):
    """{(set, binding): [(offset expr, size)] or None (whole)} for every
    non-arrayed uniform / storage buffer block the entry point reaches."""
    fid = next((f for _, f, n, _ in m.entries if n == entry), None)
    if fid is None:
        return {}
    reach, todo = set(), [fid]
    while todo:
        f = todo.pop()
        if f in reach or f not in m.functions:
            continue
        reach.add(f)
        todo += [i.args[0] for i in m.functions[f] if i.op == "OpFunctionCall"]
    b = _Builder(m, reach)
    blocks = {}
    for v, dd in m.decos.items():
        if "Binding" not in dd:
            continue
        vd = m.defs.get(v)
        if vd is None or vd.op != "OpVariable" or vd.args[0] not in (2, 12):
            continue
        t = m.pointee(vd.type)
        if m.tkind(t) != "struct":
            continue                        # descriptor arrays, images, ...: not refined
        blocks[v] = (dd.get("DescriptorSet", [0])[0], dd["Binding"][0])
    if not blocks:
        return {}
    # pointers derived from each block variable through constant-structured chains
    root = {v: (v, ()) for v in blocks}     # ptr id -> (block var, chain index ids)
    for _ in range(6):
        n = len(root)
        for i in b.body:
            if i.op in _CHAINS and i.args and i.args[0] in root:
                bv, ix = root[i.args[0]]
                root[i.id] = (bv, ix + tuple(i.args[1:]))
            elif i.op == "OpCopyObject" and i.args and i.args[0] in root:
                root[i.id] = root[i.args[0]]
        if len(root) == n:
            break
    out = {}
    bad = set()
    for i in b.body:
        if i.op in _spirv._NO_ACCESS or (i.op == "OpExtInst" and i.args and i.args[0] != m.glsl_ext):
            continue
        refs = [k for k, x in enumerate(i.args) if x in root]
        if not refs:
            continue
        if i.op in _CHAINS and refs == [0] or i.op == "OpCopyObject" or i.op == "OpArrayLength":
            continue
        bv = root[i.args[refs[0]]][0]
        if i.op == "OpLoad" and refs == [0]:
            ptrs = [i.args[0]]
        elif i.op == "OpStore" and refs == [0]:
            ptrs = [i.args[0]]
        elif i.op.startswith("OpAtomic") and refs == [0]:
            ptrs = [i.args[0]]
        elif i.op == "OpCopyMemory":
            ptrs = [i.args[k] for k in refs if k < 2]
        else:
            for k in refs:                  # escapes: function call, phi, select, store of a pointer...
                bad.add(root[i.args[k]][0])
            continue
        for p in ptrs:
            bv, ix = root[p]
            off = chain_offset(m, bv, ix, b)
            size = _access_type_size(m, None, bv, ix, b)
            if off is None or size is None:
                bad.add(bv)
                continue
            out.setdefault(bv, []).append((off, size))
    res = {}
    for v, key in blocks.items():
        if v not in bad and v not in out:
            continue                        # not accessed (or only its length)
        cur = res.get(key, [])
        res[key] = None if (v in bad or cur is None) else cur + out[v]
    return res


def local_size(m, entry):
    fid = next((f for _, f, n, _ in m.entries if n == entry), None)
    modes = getattr(m, "exec_modes", {}).get(fid, {})
    if 17 in modes:                         # LocalSize x y z
        return tuple(modes[17][:3])
    if 38 in modes:                         # LocalSizeId: only plain constants
        out = []
        for c in modes[38][:3]:
            d = m.defs.get(c)
            if d is None or d.op != "OpConstant":
                return None
            out.append(m.consts.get(c))
        return tuple(out)
    return None


class Ctx:
    """One command's values: push-constant bytes (+ which were set), builtin ranges."""

    def __init__(self, pc=None, pc_mask=None, builtins=None):
        self.pc, self.pc_mask = pc, pc_mask
        self.bi = builtins or {}
        self.memo = {}

    def eval(self, e):
        k = id(e)
        r = self.memo.get(k)
        if r is None:
            r = self._eval(e)
            self.memo[k] = r
        return r

    def _eval(self, e):
        tag = e[0]
        if tag == "const":
            return (e[1], e[1])
        if tag == "unk":
            return (e[1], e[2])
        if tag == "pc":
            off, w, signed = e[1], e[2], e[3]
            if self.pc is None or off + w > len(self.pc) or not all(self.pc_mask[off:off + w]):
                return (-(1 << (8 * w - 1)), (1 << (8 * w - 1)) - 1) if signed else (0, (1 << (8 * w)) - 1)
            v = int.from_bytes(bytes(self.pc[off:off + w]), "little", signed=signed)
            return (v, v)
        if tag == "bi":
            r = self.bi.get((e[1], e[2]))
            if r is None and e[2] is None:
                return (-INF, INF)          # a whole vector used as a scalar: should not happen
            return r if r is not None else (0, (1 << 32) - 1)
        if tag == "join":
            rs = [self.eval(x) for x in e[1:]]
            rs = [r for r in rs if r is not None]
            if not rs:
                return (-INF, INF)
            return (min(r[0] for r in rs), max(r[1] for r in rs))
        if tag == "extract":
            v = e[1]
            if v[0] == "vec" and len(e) == 3 and e[2] + 1 < len(v):
                return self.eval(v[1 + e[2]])
            if v[0] == "bi" and v[2] is None and len(e) == 3:
                r = self.bi.get((v[1], e[2]))
                return r if r is not None else (0, (1 << 32) - 1)
            if v[0] == "join" and len(e) == 3:
                return self.eval(("join",) + tuple(("extract", x, e[2]) for x in v[1:]))
            return (-INF, INF)
        if tag == "shuffle":
            return (-INF, INF)
        if tag == "vec":
            return (-INF, INF)
        if tag == "op":
            return _arith(e[1], self.eval(e[2]), self.eval(e[3]), e[4])
        return (-INF, INF)


def _arith(op, a, b, trange):
    lo, hi = trange
    (a0, a1), (b0, b1) = a, b
    try:
        if op == "OpIAdd":
            r = (a0 + b0, a1 + b1)
        elif op == "OpISub":
            r = (a0 - b1, a1 - b0)
        elif op == "OpIMul":
            ps = [0 if (x == 0 or y == 0) else x * y for x in (a0, a1) for y in (b0, b1)]
            r = (min(ps), max(ps))
        elif op in ("OpUDiv", "OpSDiv"):
            if b0 > 0 and a0 >= 0:
                r = (a0 // b1 if b1 != INF else 0, a1 // b0 if a1 != INF else INF)
            else:
                r = (-INF, INF)
        elif op in ("OpUMod", "OpSMod", "OpSRem"):
            if b0 > 0 and a0 >= 0 and b1 != INF:
                r = (0, min(a1, b1 - 1))
            else:
                r = (-INF, INF)
        elif op == "OpShiftLeftLogical":
            if b0 == b1 and 0 <= b0 < 64:
                r = (a0 * (1 << b0), a1 * (1 << b0))
            else:
                r = (-INF, INF)
        elif op in ("OpShiftRightLogical", "OpShiftRightArithmetic"):
            if b0 >= 0 and b1 < 64 and a0 >= 0:
                r = (a0 // (1 << int(b1)) if a0 != INF else INF, a1 // (1 << int(b0)) if a1 != INF else INF)
            else:
                r = (-INF, INF)
        elif op == "OpBitwiseAnd":
            if a0 >= 0 and b0 >= 0:
                r = (0, min(a1, b1))
            elif b0 >= 0:
                r = (0, b1)
            elif a0 >= 0:
                r = (0, a1)
            else:
                r = (-INF, INF)
        elif op == "OpBitwiseOr":
            if a0 >= 0 and b0 >= 0 and a1 != INF and b1 != INF:
                r = (max(a0, b0), (1 << int(max(a1, b1)).bit_length()) - 1)
            else:
                r = (-INF, INF)
        elif op in ("UMin", "SMin", "Min"):
            r = (min(a0, b0), min(a1, b1))
        elif op in ("UMax", "SMax", "Max"):
            r = (max(a0, b0), max(a1, b1))
        else:
            r = (-INF, INF)
    except (OverflowError, ValueError, TypeError):
        r = (-INF, INF)
    # no-wrap assumption: the value is the mathematical result, which must fit the type
    return (max(r[0], lo), min(r[1], hi)) if r[0] <= hi and r[1] >= lo else (lo, hi)


def ranges(exprs, ctx, limit=8):
    """[(lo, hi)) byte intervals relative to the descriptor start, or None (whole)."""
    if exprs is None:
        return None
    iv = []
    for off, size in exprs:
        lo, hi = ctx.eval(off)
        # in-bounds assumption: below the descriptor start is undefined behaviour
        lo = max(lo, 0)
        if hi < 0:
            continue
        iv.append((lo, INF if hi == INF else hi + size))
    iv.sort()
    out = []
    for lo, hi in iv:
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    if len(out) > limit:
        out = [(out[0][0], max(h for _, h in out))]
    return out
