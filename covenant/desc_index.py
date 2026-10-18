"""Descriptor-array indexing: which descriptor-array elements a draw can dynamically use.

A draw that statically uses a descriptor array binding is charged, by default,
with a read of every written element of the array (host_trace.draw). That is
sound but, for renderers that select the texture per instance (GTK 4's GSK:
one set holding every image of the frame, ``textures[id >> 1]`` with ``id`` a
flat per-instance vertex attribute), it charges reads of images the draw never
samples (descriptorsets.adoc: a descriptor that is not dynamically used is not
"considered to be referenced during command execution").

This module recognises, from the pipeline's SPIR-V only, the one index shape
the refinement accepts:

    index  =  ops(attr)      or a constant
    attr   =  a scalar 32-bit integer vertex input at Location L (no Component),
              passed unchanged or through ``ops`` by the vertex shader into a
              scalar integer Output at Location K, read in the fragment shader
              from a ``Flat`` Input at Location K (or used directly in the vertex
              shader)
    ops    =  a sequence of OpShiftRightLogical / OpShiftRightArithmetic /
              OpShiftLeftLogical / OpBitwiseAnd by OpConstant (never a
              specialization constant), OpCopyObject and same-width OpBitcast

Anything else (a load from a buffer, a push constant, a uniform, a builtin, a
computed value, a spec constant, a pointer to the array escaping into a call,
an Output stored through a pointer, a geometry/tessellation/mesh stage, ...)
makes the binding ``None``: the caller then keeps the default behaviour (every
written element). Control flow is followed (path-insensitively: branch
conditions are ignored, which only adds elements); function-local variables
and pointer parameters are tracked through stores and loads per call context.

The result is per pipeline: ``{(set, binding): None | frozenset(terms)}``, a
term being ``("c", value)`` (a constant element) or ``("attr", location, ops)``.
Decoding the attribute bytes and the schedule-independence conditions are the
caller's job (host_trace.draw).
"""
from . import spirv as _spirv

TOP = "TOP"                 # unknown value
CAP = 16                    # most distinct terms a value may hold before it is TOP
MAX_OPS = 4                 # longest op chain kept in a term
MAX_VISITS = 64             # per block and context: a fixed point not reached by then -> give up
MAX_DEPTH = 24              # call depth

_SHIFTS = {"OpShiftRightLogical": "shr", "OpShiftRightArithmetic": "sar", "OpShiftLeftLogical": "shl"}
_STAGE_MODEL = {0: "VERTEX", 4: "FRAGMENT"}


class GiveUp(Exception):
    pass


def apply_ops(ops, v):
    """Evaluate an op chain on a 32-bit pattern (unsigned)."""
    v &= 0xFFFFFFFF
    for op, k in ops:
        if op == "shr":
            v = v >> (k & 31)
        elif op == "sar":
            s = v - (1 << 32) if v & 0x80000000 else v
            v = (s >> (k & 31)) & 0xFFFFFFFF
        elif op == "shl":
            v = (v << (k & 31)) & 0xFFFFFFFF
        elif op == "and":
            v &= k & 0xFFFFFFFF
        else:
            raise ValueError(op)
    return v


def _join(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if a is TOP or b is TOP:
        return TOP
    u = a | b
    return TOP if len(u) > CAP else u


def _join_state(a, b):
    """States map tracked pointers to values; a missing pointer is TOP."""
    if a is None:
        return dict(b)
    out = {}
    for k, v in a.items():
        if k in b:
            j = _join(v, b[k])
            if j is not TOP:
                out[k] = j
    return out


def _map_terms(val, f):
    if val is TOP:
        return TOP
    out = set()
    for t in val:
        x = f(t)
        if x is TOP:
            return TOP
        out.add(x)
    return frozenset(out) if len(out) <= CAP else TOP


def _push_op(term, op):
    if term[0] == "c":
        return ("c", apply_ops((op,), term[1]))
    kind, loc, ops, flat = term
    if len(ops) >= MAX_OPS:
        return TOP
    return (kind, loc, ops + (op,), flat)


class _ModuleAnalysis:
    def __init__(self, m, entry):
        self.m = m
        self.fid = next((f for _, f, n, _ in m.entries if n == entry), None)
        self.model = next((mod for mod, _, n, _ in m.entries if n == entry), None)
        if self.fid is None:
            raise GiveUp("no entry point")
        self.var_key = {v: (d.get("DescriptorSet", [0])[0], d["Binding"][0])
                        for v, d in m.decos.items() if "Binding" in d}
        self.arrays = {}            # (set, binding) -> value (frozenset | TOP)
        self.inputs, self.outputs = {}, {}     # var id -> (location, flat) of scalar 32-bit int I/O
        for v, d in m.decos.items():
            ins = m.defs.get(v)
            if ins is None or ins.op != "OpVariable" or "Location" not in d or "BuiltIn" in d:
                continue
            if d.get(31, [0])[0] != 0:          # Component decoration (31) other than 0
                continue
            t = m.types.get(m.pointee(ins.type))
            if t is None or t.op != "OpTypeInt" or t.args[0] != 32:
                continue
            sc = ins.args[0]
            flat = 14 in d                     # Flat decoration (14)
            if sc == 1:
                self.inputs[v] = (d["Location"][0], flat)
            elif sc == 3:
                self.outputs[v] = d["Location"][0]
        self.memo = {}
        self.blocks_cache = {}

    # -- helpers --------------------------------------------------------------
    def const(self, i):
        d = self.m.defs.get(i)
        if d is None or d.op != "OpConstant":
            return None
        t = self.m.types.get(d.type)
        if t is None or t.op != "OpTypeInt" or t.args[0] != 32:
            return None
        return d.args[0] & 0xFFFFFFFF

    def is_array_desc(self, var):
        d = self.m.defs.get(var)
        if d is None or var not in self.var_key:
            return False
        t = self.m.types.get(self.m.pointee(d.type))
        return t is not None and t.op in ("OpTypeArray", "OpTypeRuntimeArray")

    def blocks(self, fid):
        if fid in self.blocks_cache:
            return self.blocks_cache[fid]
        blocks, order, cur = {}, [], None
        for ins in self.m.functions.get(fid, []):
            if ins.op == "OpLabel":
                cur = ins.id
                blocks[cur] = []
                order.append(cur)
            elif cur is not None:
                blocks[cur].append(ins)
        succ = {}
        for b, lst in blocks.items():
            t = lst[-1] if lst else None
            s = []
            if t is not None:
                if t.op == "OpBranch":
                    s = [t.args[0]]
                elif t.op == "OpBranchConditional":
                    s = [t.args[1], t.args[2]]
                elif t.op == "OpSwitch":
                    s = [t.args[1]] + t.args[3::2]
            succ[b] = [x for x in s if x in blocks]
        self.blocks_cache[fid] = (blocks, order, succ)
        return self.blocks_cache[fid]

    # -- interpretation -------------------------------------------------------------
    def run(self):
        self.escape_scan()
        out_state, _ = self.call(self.fid, (), {}, 0)
        outs = {}
        for v, loc in self.outputs.items():
            val = out_state.get(v, TOP) if out_state is not None else TOP
            outs[loc] = _join(outs.get(loc), val) if loc in outs else val
        return self.arrays, outs

    def escape_scan(self):
        """A descriptor array used other than as the base of an access chain: unknown index."""
        reach, todo = set(), [self.fid]
        while todo:
            f = todo.pop()
            if f in reach or f not in self.m.functions:
                continue
            reach.add(f)
            todo += [i.args[0] for i in self.m.functions[f] if i.op == "OpFunctionCall"]
        for f in reach:
            for ins in self.m.functions[f]:
                if ins.op in _spirv._NO_ACCESS:
                    continue
                args = ins.args
                if ins.op in ("OpAccessChain", "OpInBoundsAccessChain") and args:
                    args = args[1:]
                for a in args:
                    if a in self.var_key and self.is_array_desc(a):
                        self.arrays[self.var_key[a]] = TOP

    def call(self, fid, args, outs_in, depth):
        """Analyse function fid in a context. args: per parameter ("ptr", value) or ("val", value).
        outs_in: tracked Output var -> value. Returns (state at return (Outputs only), return value)."""
        if depth > MAX_DEPTH or fid not in self.m.functions:
            raise GiveUp("call depth")
        key = (fid, args, tuple(sorted(outs_in.items())))
        if key in self.memo:
            return self.memo[key]
        blocks, order, succ = self.blocks(fid)
        vals, ptr_base = {}, {}
        params = self.m.params.get(fid, [])
        entry = {k: v for k, v in outs_in.items() if v is not TOP}
        for p, a in zip(params, args):
            if a[0] == "ptr":
                if a[1] is not TOP:
                    entry[p] = a[1]
                ptr_base[p] = p
            else:
                vals[p] = a[1]
        tracked = set(self.outputs) | {p for p, a in zip(params, args) if a[0] == "ptr"}
        if not order:
            raise GiveUp("function without body")
        ins_state = {order[0]: entry}
        ret_state, ret_val = None, None
        rounds = 0
        changed = True
        # round-robin to a fixed point over states and SSA values (phis read SSA values of
        # predecessors, so a change of either re-runs every block); values only grow
        while changed:
            rounds += 1
            if rounds > MAX_VISITS:
                raise GiveUp("no fixed point")
            changed = False
            before_vals = dict(vals)
            ret_state, ret_val = None, None
            for b in order:
                if b not in ins_state:
                    continue
                st = self.block(b, blocks[b], dict(ins_state[b]), vals, ptr_base, tracked, depth)
                if st is None:
                    continue
                st, rs, rv, is_ret = st
                if is_ret:
                    ret_state = _join_state(ret_state, rs)
                    ret_val = _join(ret_val, rv)
                for s in succ.get(b, []):
                    new = _join_state(ins_state[s], st) if s in ins_state else dict(st)
                    if s not in ins_state or new != ins_state[s]:
                        ins_state[s] = new
                        changed = True
            if vals != before_vals:
                changed = True
        res = (ret_state if ret_state is not None else {}, ret_val)
        self.memo[key] = res
        return res

    def block(self, b, body, st, vals, ptr_base, tracked, depth):
        ret_state, ret_val, is_ret = None, None, False
        if True:
            for ins in body:
                op = ins.op
                if op == "OpVariable":                     # function-local variable
                    tracked.add(ins.id)
                    ptr_base[ins.id] = ins.id
                    if len(ins.args) > 1:
                        st[ins.id] = self.value(ins.args[1], vals)
                    elif ins.id not in st:
                        st.pop(ins.id, None)
                elif op == "OpPhi":
                    v = None
                    for i in range(0, len(ins.args), 2):
                        x = vals.get(ins.args[i])
                        if x is None and ins.args[i] in self.m.consts:
                            x = self.value(ins.args[i], vals)
                        v = _join(v, x)
                    vals[ins.id] = TOP if v is None else v
                elif op == "OpLoad":
                    p = ins.args[0]
                    base = ptr_base.get(p, p)
                    if p in tracked and ptr_base.get(p, p) == p:
                        vals[ins.id] = st.get(p, TOP)
                    elif p in self.inputs and ptr_base.get(p, p) == p:
                        loc, flat = self.inputs[p]
                        vals[ins.id] = frozenset({("in", loc, (), flat)})
                    else:
                        vals[ins.id] = TOP
                    if base in self.var_key:
                        pass                                   # descriptor load: not an index
                elif op == "OpStore":
                    p, v = ins.args[0], ins.args[1]
                    base = ptr_base.get(p, p)
                    if base in tracked:
                        if base == p:
                            val = self.value(v, vals)
                            if val is TOP:
                                st.pop(p, None)
                            else:
                                st[p] = val
                        else:
                            st.pop(base, None)                 # store through a derived pointer
                elif op in ("OpAccessChain", "OpInBoundsAccessChain", "OpPtrAccessChain",
                            "OpInBoundsPtrAccessChain"):
                    base = ins.args[0]
                    root = ptr_base.get(base, base)
                    ptr_base[ins.id] = root
                    if root == base and base in self.var_key and self.is_array_desc(base) \
                            and op in ("OpAccessChain", "OpInBoundsAccessChain") and len(ins.args) >= 2:
                        k = self.var_key[base]
                        self.arrays[k] = _join(self.arrays.get(k), self.value(ins.args[1], vals))
                    elif root in self.var_key and self.is_array_desc(root):
                        self.arrays[self.var_key[root]] = TOP   # a second chain on a chained pointer
                elif op == "OpCopyObject":
                    a = ins.args[0]
                    if a in ptr_base:
                        ptr_base[ins.id] = ptr_base[a]
                    vals[ins.id] = self.value(a, vals)
                elif op == "OpBitcast":
                    t = self.m.types.get(ins.type)
                    a = ins.args[0]
                    if a in ptr_base:
                        ptr_base[ins.id] = ptr_base[a]
                        vals[ins.id] = TOP
                    elif t is not None and t.op == "OpTypeInt" and t.args[0] == 32:
                        vals[ins.id] = self.value(a, vals)
                    else:
                        vals[ins.id] = TOP
                elif op in _SHIFTS or op == "OpBitwiseAnd":
                    a, c = ins.args[0], ins.args[1]
                    k = self.const(c)
                    if k is None and op == "OpBitwiseAnd":
                        a, c = c, a
                        k = self.const(c)
                    if k is None:
                        vals[ins.id] = TOP
                    else:
                        o = ("and", k) if op == "OpBitwiseAnd" else (_SHIFTS[op], k)
                        vals[ins.id] = _map_terms(self.value(a, vals), lambda t: _push_op(t, o))
                elif op == "OpFunctionCall":
                    callee = ins.args[0]
                    cargs = []
                    for a in ins.args[1:]:
                        if a in ptr_base or self.m.defs.get(a) is not None and self.m.defs[a].op == "OpVariable":
                            base = ptr_base.get(a, a)
                            cargs.append(("ptr", st.get(a, TOP) if base == a and a in tracked else TOP))
                        else:
                            cargs.append(("val", self.value(a, vals)))
                    souts = {v: st.get(v, TOP) for v in self.outputs}
                    c_state, c_ret = self.call(callee, tuple(cargs), souts, depth + 1)
                    for v in self.outputs:
                        x = (c_state or {}).get(v, TOP)
                        if x is TOP:
                            st.pop(v, None)
                        else:
                            st[v] = x
                    for a in ins.args[1:]:
                        base = ptr_base.get(a, a)
                        if base in tracked:
                            st.pop(base, None)                 # the callee may have written it
                    vals[ins.id] = TOP if c_ret is None else c_ret
                elif op in ("OpReturn", "OpReturnValue"):
                    is_ret = True
                    ret_state = {v: st[v] for v in self.outputs if v in st}
                    if op == "OpReturnValue":
                        ret_val = self.value(ins.args[0], vals)
                elif op in ("OpCopyMemory", "OpCopyMemorySized"):
                    base = ptr_base.get(ins.args[0], ins.args[0])
                    st.pop(base, None)
                elif op.startswith("OpAtomic"):
                    base = ptr_base.get(ins.args[0], ins.args[0]) if ins.args else None
                    if base in tracked:
                        st.pop(base, None)
                    if ins.id is not None:
                        vals[ins.id] = TOP
                elif ins.id is not None:
                    if ins.args and ins.args[0] in ptr_base:
                        ptr_base[ins.id] = ptr_base[ins.args[0]]   # other pointer-producing ops
                    vals[ins.id] = TOP
        return st, ret_state, ret_val, is_ret

    def value(self, i, vals):
        if i in vals:
            return vals[i]
        k = self.const(i)
        if k is not None:
            return frozenset({("c", k)})
        return TOP


def module_index_terms(m, entry):
    """(arrays, outputs) of one entry point, or None when the analysis gives up.
    arrays: {(set, binding): frozenset(terms) | TOP} for descriptor arrays it indexes;
    outputs: {location: frozenset(terms) | TOP} of scalar 32-bit integer Outputs at return.
    Terms: ("c", v) or ("in", input location, ops, flat)."""
    try:
        a = _ModuleAnalysis(m, entry)
        return a.run()
    except GiveUp:
        return None
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return None


def pipeline_index_terms(stages, load):
    """stages: [(stage flags, sha1, entry)] of a graphics pipeline; load(sha1) -> spirv.Module.
    Returns {(set, binding): None | frozenset(("c", v) | ("attr", location, ops))}: the index
    terms of every descriptor array the pipeline indexes (None: not understood; the caller
    charges every element), or None when the pipeline is outside the accepted shape."""
    kinds = {}
    for flags, sha, entry in stages:
        names = set(flags.split("|"))
        if names == {"VK_SHADER_STAGE_VERTEX_BIT"}:
            kinds["VERTEX"] = (sha, entry)
        elif names == {"VK_SHADER_STAGE_FRAGMENT_BIT"}:
            kinds["FRAGMENT"] = (sha, entry)
        else:
            return None                         # tessellation / geometry / mesh / combined: not handled
    if "VERTEX" not in kinds:
        return None
    res = {}
    for k, (sha, entry) in kinds.items():
        m = load(sha)
        if m is None:
            return None
        got = module_index_terms(m, entry)
        if got is None:
            return None
        res[k] = got
    v_arrays, v_outs = res["VERTEX"]
    out = {}

    def from_vertex(term):
        if term[0] == "c":
            return term
        _, loc, ops, _flat = term
        return ("attr", loc, ops)

    def from_fragment(term):
        if term[0] == "c":
            return [term]
        _, loc, ops, flat = term
        src = v_outs.get(loc, TOP)
        if not flat or src is TOP:
            return None
        lst = []
        for s in src:
            if s[0] == "c":
                lst.append(("c", apply_ops(ops, s[1])))
            else:
                _, vloc, vops, _ = s
                if len(vops) + len(ops) > MAX_OPS:
                    return None
                lst.append(("attr", vloc, vops + ops))
        return lst

    for key, val in v_arrays.items():
        out[key] = None if val is TOP else frozenset(from_vertex(t) for t in val)
    if "FRAGMENT" in res:
        for key, val in res["FRAGMENT"][0].items():
            if val is TOP or out.get(key, 0) is None:
                out[key] = None
                continue
            terms = set(out.get(key) or ())
            for t in val:
                x = from_fragment(t)
                if x is None:
                    terms = None
                    break
                terms.update(x)
            out[key] = None if terms is None else frozenset(terms)
    return out
