"""A small SPIR-V interpreter and probe fingerprinting.

The interpreter executes one entry point (with its callees) on N lanes at
once (numpy arrays: scalars (N,), vectors (N, k), matrices (N, cols, rows)).
Values that come from outside the shader are *sources*: varyings, sampled
texels, and uniform / push-constant / storage loads whose bytes are not known.
Uniform data the model driver observed (or the host front end knows) is
decoded from the SPIR-V block layout and is constant.

Fingerprinting (fingerprint()):
1. one base run records every executed SSA value and its dynamic operands;
2. dominators of the dependence graph, with a virtual root feeding every
   source: a node whose immediate dominator is the root is a *head* (a
   source, or a value merging several sources); every other value is a
   function of its head alone (a univariate slice);
3. each head is overridden with a ramp (splat over colour components,
   alpha = 1) and the region below it is evaluated in one run;
4. each value v of a region is fitted against the transfer library
   (transfer.fit) as a function of its dominators, nearest to the head first;
   the first unambiguous, monotone fit labels the edge u -> v; otherwise v is
   opaque (no constraint, no alarm).
"""
import struct as _st
from dataclasses import dataclass, field

import numpy as np

from . import transfer

ROW_MAJOR, COL_MAJOR = 4, 5          # SPIR-V decoration numbers
MAX_STEPS = 200000


class Unsupported(Exception):
    pass


class Divergent(Exception):
    pass


class Killed(Exception):
    pass


@dataclass
class Handle:
    var: int


@dataclass
class Ptr:
    var: tuple          # memory key: ("var", ctx, id) or ("glob", id)
    path: tuple
    tid: int            # pointee type
    sc: int             # storage class
    root: int           # the OpVariable id


@dataclass
class Run:
    values: dict = field(default_factory=dict)      # (ctx, id) -> value
    deps: dict = field(default_factory=dict)        # (ctx, id) -> set of keys
    types: dict = field(default_factory=dict)       # (ctx, id) -> type id
    order: list = field(default_factory=list)
    sources: dict = field(default_factory=dict)     # key -> kind ("input", "texel", "uniform", ...)
    stores: list = field(default_factory=list)      # (var id, path, value key, ctx)
    blocks: set = field(default_factory=set)        # executed (ctx-free) block labels
    loads: dict = field(default_factory=dict)       # load key -> (var id, path)


def _f(x):
    return np.asarray(x, dtype=np.float64)


class Memory:
    """Concrete bytes of descriptor-backed blocks: {var id: (bytes, known mask)}."""

    def __init__(self, module, blocks=None, host_values=None):
        self.m = module
        self.blocks = blocks or {}
        self.host_values = host_values or {}

    def decode(self, tid, var, path, ctx_decos=None):
        """Value (unbatched python / numpy) of the member at `path` inside the
        block of variable var, or None if any byte is unknown."""
        m = self.m
        got = self.blocks.get(var)
        if got is None:
            return self._host(var, path)
        data, mask = got
        t = m.pointee(m.defs[var].type)
        off, deco = 0, {}
        for pi, idx in enumerate(path):
            k = m.tkind(t)
            if k == "struct":
                md = m.member_decos.get((t, idx), {})
                off += md.get("Offset", [0])[0]
                deco = md
                t = m.types[t].args[idx]
            elif k in ("array", "runtimearray"):
                stride = m.decos.get(t, {}).get("ArrayStride", [0])[0]
                off += stride * int(idx)
                t = m.types[t].args[0]
            elif k == "matrix":
                ms = deco.get("MatrixStride", [16])[0]
                if ROW_MAJOR in deco:
                    # a column of a RowMajor matrix is strided: its element r is at
                    # r * MatrixStride + 4 * column (SPIR-V Offset / MatrixStride rules)
                    colt = m.types[t].args[0]
                    ct, nrow = m.types[colt].args
                    rest = path[pi + 1:]
                    rows = [int(rest[0])] if len(rest) == 1 else range(nrow)
                    if len(rest) > 1:
                        return None
                    col = [self._read(ct, data, mask, off + r * ms + 4 * int(idx), {}) for r in rows]
                    if any(v is None for v in col):
                        return None
                    return col[0] if rest else col
                off += ms * int(idx)
                t = m.types[t].args[0]
                deco = {}
            elif k == "vector":
                off += 4 * int(idx)
                t = m.types[t].args[0]
            else:
                return None
        return self._read(t, data, mask, off, deco)

    def _host(self, var, path):
        m = self.m
        if not self.host_values:
            return None
        t = m.pointee(m.defs[var].type)
        for idx in path:
            if m.tkind(t) != "struct":
                return None
            key = (m.struct_name(t), m.member_names.get((t, idx)))
            if key in self.host_values and idx == path[-1]:
                v = self.host_values[key][0]
                return float(v) if m.tkind(m.types[t].args[idx]) == "float" else int(v)
            t = m.types[t].args[idx]
        return None

    def _read(self, t, data, mask, off, deco):
        m = self.m
        k = m.tkind(t)
        if k in ("float", "int", "bool"):
            w = m.types[t].args[0] if k != "bool" else 32
            n = w // 8
            if off + n > len(data) or not all(mask[off:off + n]):
                return None
            raw = bytes(data[off:off + n])
            if k == "float":
                return _st.unpack("<f" if n == 4 else "<e", raw)[0] if n in (2, 4) else None
            if k == "bool":
                return bool(int.from_bytes(raw, "little"))
            signed = m.types[t].args[1]
            return int.from_bytes(raw, "little", signed=bool(signed))
        if k == "vector":
            ct, n = m.types[t].args
            vals = [self._read(ct, data, mask, off + 4 * i, {}) for i in range(n)]
            return None if any(v is None for v in vals) else vals
        if k == "matrix":
            colt, ncol = m.types[t].args
            nrow = m.types[colt].args[1]
            ms = deco.get("MatrixStride", [16])[0]
            out = []
            for c in range(ncol):
                col = []
                for r in range(nrow):
                    o = off + (r * ms + 4 * c if ROW_MAJOR in deco else c * ms + 4 * r)
                    v = self._read(m.types[colt].args[0], data, mask, o, {})
                    if v is None:
                        return None
                    col.append(v)
                out.append(col)
            return ("mat", out)
        if k == "struct":
            out = []
            for i, mt in enumerate(m.types[t].args):
                md = m.member_decos.get((t, i), {})
                v = self._read(mt, data, mask, off + md.get("Offset", [0])[0], md)
                if v is None:
                    return None
                out.append(v)
            return ("struct", out)
        if k == "array":
            n = m.consts.get(m.types[t].args[1])
            stride = m.decos.get(t, {}).get("ArrayStride", [0])[0]
            if not isinstance(n, int) or not stride:
                return None
            out = []
            for i in range(n):
                v = self._read(m.types[t].args[0], data, mask, off + i * stride, deco)
                if v is None:
                    return None
                out.append(v)
            return ("array", out)
        return None


class Interp:
    def __init__(self, module, memory=None, source_value=None):
        self.m = module
        self.mem = memory or Memory(module)
        self.source_value = source_value or (lambda key, tid, n, kind: None)
        self._blocks = {}
        from .spirv import _grammar
        self.ext = _grammar()[1]

    # ------------------------------------------------------------ types
    def tk(self, t):
        return self.m.tkind(t)

    def zeros(self, t, n):
        k = self.tk(t)
        a = self.m.types.get(t)
        if k == "float":
            return np.zeros(n)
        if k == "int":
            return np.zeros(n, dtype=np.int64)
        if k == "bool":
            return np.zeros(n, dtype=bool)
        if k == "vector":
            z = self.zeros(a.args[0], n)
            return np.zeros((n, a.args[1]), dtype=z.dtype)
        if k == "matrix":
            rows = self.m.types[a.args[0]].args[1]
            return np.zeros((n, a.args[1], rows))
        if k == "struct":
            return [self.zeros(x, n) for x in a.args]
        if k == "array":
            ln = self.m.consts.get(a.args[1])
            return [self.zeros(a.args[0], n) for _ in range(ln if isinstance(ln, int) else 1)]
        return None

    def from_py(self, v, t, n):
        """Batch a decoded python value of type t to n lanes."""
        k = self.tk(t)
        a = self.m.types.get(t)
        if k == "float":
            return np.full(n, float(v))
        if k == "int":
            return np.full(n, int(v), dtype=np.int64)
        if k == "bool":
            return np.full(n, bool(v))
        if k == "vector":
            return np.tile(np.asarray(v, dtype=np.float64 if self.tk(a.args[0]) == "float" else np.int64), (n, 1))
        if k == "matrix":
            return np.tile(np.asarray(v[1], dtype=np.float64), (n, 1, 1))
        if k == "struct":
            return [self.from_py(x, mt, n) for x, mt in zip(v[1], a.args)]
        if k == "array":
            return [self.from_py(x, a.args[0], n) for x in v[1]]
        raise Unsupported(f"value of type {k}")

    def const(self, i, n):
        m = self.m
        d = m.defs.get(i)
        if d is None:
            raise Unsupported(f"unknown id {i}")
        if d.op == "OpConstant" or d.op == "OpSpecConstant":
            t = self.tk(d.type)
            v = m.consts[i]
            if t == "int" and m.types[d.type].args[1] and v >= 2 ** 31:
                v -= 2 ** 32
            return self.from_py(v, d.type, n)
        if d.op in ("OpConstantTrue", "OpSpecConstantTrue"):
            return np.ones(n, dtype=bool)
        if d.op in ("OpConstantFalse", "OpSpecConstantFalse"):
            return np.zeros(n, dtype=bool)
        if d.op in ("OpConstantNull", "OpUndef"):
            return self.zeros(d.type, n)
        if d.op in ("OpConstantComposite", "OpSpecConstantComposite"):
            parts = [self.const(x, n) for x in d.args]
            return self._construct(d.type, parts, n)
        raise Unsupported(d.op)

    def is_const(self, i):
        d = self.m.defs.get(i)
        return d is not None and d.op.startswith(("OpConstant", "OpSpecConstant")) or \
            (d is not None and d.op == "OpUndef")

    def _construct(self, t, parts, n):
        k = self.tk(t)
        if k == "vector":
            cols = [p.reshape(n, -1) if isinstance(p, np.ndarray) else p for p in parts]
            dt = np.result_type(*[c.dtype for c in cols])
            return np.concatenate([c.astype(dt) for c in cols], axis=1)
        if k == "matrix":
            return np.stack([p.astype(np.float64) for p in parts], axis=1)
        return list(parts)

    # ------------------------------------------------------------ run
    def run(self, fid, n, overrides=None):
        self.n = n
        self.over = overrides or {}
        self.r = Run()
        self.memv = {}          # memory key -> value
        self.memdeps = {}       # memory key -> set of keys
        self.steps = 0
        self._call(fid, (), [])
        return self.r

    def _blocks_of(self, fid):
        if fid not in self._blocks:
            body = self.m.functions[fid]
            blocks, order, cur = {}, [], None
            for ins in body:
                if ins.op == "OpLabel":
                    cur = ins.id
                    blocks[cur] = []
                    order.append(cur)
                elif cur is not None:
                    blocks[cur].append(ins)
            self._blocks[fid] = (blocks, order)
        return self._blocks[fid]

    def _set(self, ctx, ins, val, deps, source=None):
        key = (ctx, ins.id)
        if key in self.over:
            val = self.over[key]
        r = self.r
        if key not in r.values:
            r.order.append(key)
        r.values[key] = val
        r.types[key] = ins.type
        r.deps[key] = deps
        if source:
            r.sources[key] = source

    def _val(self, ctx, i):
        if (ctx, i) in self.r.values:
            return self.r.values[(ctx, i)]
        if ((), i) in self.r.values:
            return self.r.values[((), i)]
        if self.is_const(i):
            return self.const(i, self.n)
        d = self.m.defs.get(i)
        if d is not None and d.op == "OpVariable":
            return self._var_ptr(ctx, i)
        raise Unsupported(f"value %{i} not computed")

    def _dep(self, ctx, i):
        if (ctx, i) in self.r.values:
            return {(ctx, i)}
        if ((), i) in self.r.values:
            return {((), i)}
        return set()

    def _var_ptr(self, ctx, i):
        d = self.m.defs[i]
        sc = d.args[0]
        mk = ("var", ctx, i) if sc == 7 else ("glob", i)
        return Ptr(mk, (), self.m.pointee(d.type), sc, i)

    def _call(self, fid, ctx, args):
        m = self.m
        for p, (v, dp) in zip(m.params.get(fid, []), args):
            self.r.values[(ctx, p)] = v
            self.r.deps[(ctx, p)] = dp
            self.r.types[(ctx, p)] = m.defs[p].type
            self.r.order.append((ctx, p))
        blocks, order = self._blocks_of(fid)
        cur, prev = order[0], None
        while True:
            self.r.blocks.add(cur)
            ins_list = blocks[cur]
            # phis first, all from the predecessor
            phis = [i for i in ins_list if i.op == "OpPhi"]
            pv = []
            for ph in phis:
                pairs = [(ph.args[k], ph.args[k + 1]) for k in range(0, len(ph.args), 2)]
                src = next((v for v, b in pairs if b == prev), None)
                if src is None:
                    raise Unsupported("phi without matching predecessor")
                pv.append((ph, self._val(ctx, src), self._dep(ctx, src)))
            for ph, v, dp in pv:
                self._set(ctx, ph, v, dp)
            nxt = None
            for ins in ins_list:
                self.steps += 1
                if self.steps > MAX_STEPS:
                    raise Unsupported("step limit")
                op = ins.op
                if op == "OpPhi" or op in ("OpSelectionMerge", "OpLoopMerge", "OpLine", "OpNoLine",
                                          "OpNop", "OpControlBarrier", "OpMemoryBarrier"):
                    continue
                if op == "OpBranch":
                    nxt = ins.args[0]
                    break
                if op == "OpBranchConditional":
                    c = self._val(ctx, ins.args[0])
                    c = np.asarray(c, dtype=bool).reshape(-1)
                    if c.all():
                        nxt = ins.args[1]
                    elif not c.any():
                        nxt = ins.args[2]
                    else:
                        raise Divergent()
                    break
                if op == "OpSwitch":
                    s = np.asarray(self._val(ctx, ins.args[0])).reshape(-1)
                    if not (s == s[0]).all():
                        raise Divergent()
                    nxt = ins.args[1]
                    for k in range(2, len(ins.args) - 1, 2):
                        if int(s[0]) == ins.args[k]:
                            nxt = ins.args[k + 1]
                    break
                if op == "OpReturn":
                    return None
                if op == "OpReturnValue":
                    return self._val(ctx, ins.args[0]), self._dep(ctx, ins.args[0])
                if op in ("OpKill", "OpTerminateInvocation", "OpUnreachable"):
                    raise Killed()
                self.exec(ins, ctx)
            if nxt is None:
                return None
            prev, cur = cur, nxt

    # ------------------------------------------------------------ memory
    def _mem_get(self, p):
        if p.var not in self.memv:
            init = self.m.defs[p.root].args[1] if len(self.m.defs[p.root].args) > 1 else None
            self.memv[p.var] = self.const(init, self.n) if init is not None else \
                self.zeros(self.m.pointee(self.m.defs[p.root].type), self.n)
        v = self.memv[p.var]
        for idx in p.path:
            v = self._index(v, idx)
        return v

    def _index(self, v, idx):
        if isinstance(v, list):
            idx = self._scalar_index(idx, len(v))
            return v[idx]
        if isinstance(v, np.ndarray):
            if v.ndim >= 2:
                if isinstance(idx, np.ndarray):
                    return v[np.arange(v.shape[0]), np.clip(idx, 0, v.shape[1] - 1)]
                return v[:, int(idx)]
        raise Unsupported("index into scalar")

    def _scalar_index(self, idx, ln):
        if isinstance(idx, np.ndarray):
            if not (idx == idx.reshape(-1)[0]).all():
                raise Divergent()
            idx = int(idx.reshape(-1)[0])
        return max(0, min(int(idx), ln - 1))

    def _mem_put(self, p, val):
        if not p.path:
            self.memv[p.var] = val
            return
        cur = self._mem_get(Ptr(p.var, (), p.tid, p.sc, p.root))
        self.memv[p.var] = self._update(cur, list(p.path), val)

    def _update(self, v, path, val):
        idx = path[0]
        if isinstance(v, list):
            i = self._scalar_index(idx, len(v))
            v = list(v)
            v[i] = val if len(path) == 1 else self._update(v[i], path[1:], val)
            return v
        v = v.copy()
        if isinstance(idx, np.ndarray):
            idx = self._scalar_index(idx, v.shape[1])
        if len(path) == 1:
            v[:, int(idx)] = val
        else:
            v[:, int(idx)] = self._update(v[:, int(idx)], path[1:], val)
        return v

    # ------------------------------------------------------------ exec
    def exec(self, ins, ctx):
        m, op, a, n = self.m, ins.op, ins.args, self.n
        V = lambda i: self._val(ctx, i)
        D = lambda *ids: set().union(*[self._dep(ctx, i) for i in ids]) if ids else set()
        if op == "OpVariable":
            mk = ("var", ctx, ins.id)
            init = a[1] if len(a) > 1 else None
            self.memv[mk] = self.const(init, n) if init is not None else self.zeros(m.pointee(ins.type), n)
            self.memdeps[mk] = set()
            self.r.values[(ctx, ins.id)] = Ptr(mk, (), m.pointee(ins.type), a[0], ins.id)
            return
        if op in ("OpAccessChain", "OpInBoundsAccessChain"):
            base = V(a[0])
            if not isinstance(base, Ptr):
                raise Unsupported("access chain on non-pointer")
            path = list(base.path)
            t = base.tid
            for idx in a[1:]:
                iv = m.consts.get(idx) if self.is_const(idx) and isinstance(m.consts.get(idx), int) else V(idx)
                if isinstance(iv, np.ndarray) and (iv == iv.reshape(-1)[0]).all():
                    iv = int(iv.reshape(-1)[0])
                path.append(iv)
                k = m.tkind(t)
                t = m.types[t].args[iv if k == "struct" else 0] if k in ("struct", "array", "runtimearray",
                                                                         "vector", "matrix") else t
            self.r.values[(ctx, ins.id)] = Ptr(base.var, tuple(path), t, base.sc, base.root)
            self.r.deps[(ctx, ins.id)] = D(*a[1:]) | self._dep(ctx, a[0])
            return
        if op == "OpLoad":
            p = V(a[0])
            if not isinstance(p, Ptr):
                raise Unsupported("load from non-pointer")
            sc = p.sc
            tk = self.tk(ins.type)
            if sc in (0,) and tk in ("image", "sampledimage", "sampler", "accelerationstructurekhr"):
                self.r.values[(ctx, ins.id)] = Handle(p.root)
                return
            if sc in (2, 9, 12):              # Uniform, PushConstant, StorageBuffer
                if any(isinstance(x, np.ndarray) for x in p.path):
                    v = None
                else:
                    v = self.mem.decode(ins.type, p.root, p.path)
                self.r.loads[(ctx, ins.id)] = (p.root, p.path)
                if v is not None:
                    self._set(ctx, ins, self.from_py(v, ins.type, n), set())
                    return
                # a load at a lane-invariant address is a *parameter* (the
                # same for every invocation of a draw): it does not start a
                # slice; a dynamically indexed one is a per-lane source
                dyn = any(isinstance(x, np.ndarray) for x in p.path)
                self._source(ctx, ins, ("storage" if sc == 12 else "uniform") if dyn else "param")
                return
            if sc == 1:                        # Input
                self._source(ctx, ins, "input")
                return
            if sc == 0:
                self._source(ctx, ins, "uniformconstant")
                return
            v = self._mem_get(p)
            self._set(ctx, ins, v, set(self.memdeps.get(p.var, set())) | self._dep(ctx, a[0]))
            return
        if op == "OpStore":
            p = V(a[0])
            if not isinstance(p, Ptr):
                raise Unsupported("store to non-pointer")
            val = V(a[1])
            if p.sc in (2, 9, 12, 1, 0):
                return
            self._mem_put(p, val)
            dp = self._dep(ctx, a[1])
            if p.path:
                self.memdeps[p.var] = self.memdeps.get(p.var, set()) | dp
            else:
                self.memdeps[p.var] = set(dp)
            self.r.stores.append((p.root, p.path, (ctx, a[1]), ctx))
            return
        if op == "OpFunctionCall":
            args = [(V(x), self._dep(ctx, x)) for x in a[1:]]
            sub = ctx + (ins.id,)
            got = self._call(a[0], sub, args)
            if got is not None:
                self._set(ctx, ins, got[0], got[1])
            return
        if op in ("OpSampledImage", "OpImage", "OpCopyObject") and isinstance(V(a[0]), Handle):
            self.r.values[(ctx, ins.id)] = V(a[0])
            return
        if op.startswith("OpImageSample") or op in ("OpImageFetch", "OpImageGather", "OpImageRead",
                                                    "OpImageDrefGather"):
            self._source(ctx, ins, "texel", D(*a[1:2]))
            return
        if op.startswith("OpImageQuery") or op in ("OpArrayLength", "OpDPdx", "OpDPdy", "OpFwidth",
                                                   "OpDPdxFine", "OpDPdyFine", "OpFwidthFine",
                                                   "OpDPdxCoarse", "OpDPdyCoarse", "OpFwidthCoarse") \
                or op.startswith("OpAtomic") or op.startswith("OpGroup") or op.startswith("OpSubgroup"):
            self._source(ctx, ins, "query")
            return
        if op == "OpImageWrite" or op.startswith("OpEmit") or op.startswith("OpEnd") or op.startswith("OpDemote"):
            return
        if op == "OpExtInst":
            if a[0] != m.glsl_ext:
                return                                  # debug info etc.
            name = self.ext.get(a[1], "")
            ops = a[2:]
            val = self.glsl(name, [V(x) for x in ops], ins.type)
            self._set(ctx, ins, val, D(*ops))
            return
        val = self.arith(op, ins, ctx, V)
        if val is NotImplemented:
            raise Unsupported(op)
        ids = [x for x in a if not isinstance(x, int) or (ctx, x) in self.r.values or ((), x) in self.r.values]
        self._set(ctx, ins, val, D(*ids))

    def _source(self, ctx, ins, kind, deps=None):
        n = self.n
        key = (ctx, ins.id)
        v = self.source_value(key, ins.type, n, kind)
        if v is None:
            v = self.default_source(key, ins.type, n)
        self._set(ctx, ins, v, set(), source=kind)

    def default_source(self, key, t, n):
        """Fixed pseudo-random value of a source, constant across lanes."""
        rng = np.random.default_rng(abs(hash(key)) % (2 ** 32))
        k = self.tk(t)
        a = self.m.types.get(t)
        if k == "float":
            return np.full(n, rng.uniform(0.2, 0.8))
        if k == "int":
            return np.full(n, 1, dtype=np.int64)
        if k == "bool":
            return np.zeros(n, dtype=bool)
        if k == "vector":
            ct = self.tk(a.args[0])
            if ct == "float":
                return np.tile(rng.uniform(0.2, 0.8, a.args[1]), (n, 1))
            if ct == "bool":
                return np.zeros((n, a.args[1]), dtype=bool)
            return np.ones((n, a.args[1]), dtype=np.int64)
        if k == "matrix":
            rows = self.m.types[a.args[0]].args[1]
            return np.tile(np.eye(a.args[1], rows) + rng.uniform(-0.1, 0.1, (a.args[1], rows)), (n, 1, 1))
        if k == "struct":
            return [self.default_source(key + (i,), x, n) for i, x in enumerate(a.args)]
        if k == "array":
            ln = self.m.consts.get(a.args[1])
            return [self.default_source(key + (i,), a.args[0], n) for i in range(ln if isinstance(ln, int) else 1)]
        return self.zeros(t, n)

    # ------------------------------------------------------------ arithmetic
    @staticmethod
    def _bc(x, y):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            if x.ndim == 1 and y.ndim == 2:
                return x[:, None], y
            if y.ndim == 1 and x.ndim == 2:
                return x, y[:, None]
        return x, y

    def arith(self, op, ins, ctx, V):
        a = ins.args
        bin_f = {"OpFAdd": np.add, "OpFSub": np.subtract, "OpFMul": np.multiply,
                 "OpIAdd": np.add, "OpISub": np.subtract, "OpIMul": np.multiply}
        if op in bin_f:
            x, y = self._bc(V(a[0]), V(a[1]))
            return bin_f[op](x, y)
        if op == "OpFDiv":
            x, y = self._bc(V(a[0]), V(a[1]))
            with np.errstate(all="ignore"):
                return np.divide(x, y)
        if op in ("OpFMod", "OpFRem"):
            x, y = self._bc(V(a[0]), V(a[1]))
            with np.errstate(all="ignore"):
                return np.mod(x, y) if op == "OpFMod" else np.fmod(x, y)
        if op in ("OpSDiv", "OpUDiv"):
            x, y = V(a[0]), V(a[1])
            with np.errstate(all="ignore"):
                return np.where(y != 0, np.trunc(x / np.where(y == 0, 1, y)), 0).astype(np.int64)
        if op in ("OpSMod", "OpUMod", "OpSRem"):
            x, y = V(a[0]), V(a[1])
            return np.where(y != 0, np.mod(x, np.where(y == 0, 1, y)), 0).astype(np.int64)
        if op in ("OpFNegate", "OpSNegate"):
            return -V(a[0])
        if op == "OpNot":
            return ~V(a[0]).astype(np.int64)
        if op in ("OpShiftLeftLogical", "OpShiftRightLogical", "OpShiftRightArithmetic",
                  "OpBitwiseAnd", "OpBitwiseOr", "OpBitwiseXor"):
            x, y = V(a[0]).astype(np.int64), V(a[1]).astype(np.int64)
            return {"OpShiftLeftLogical": np.left_shift, "OpShiftRightLogical": np.right_shift,
                    "OpShiftRightArithmetic": np.right_shift, "OpBitwiseAnd": np.bitwise_and,
                    "OpBitwiseOr": np.bitwise_or, "OpBitwiseXor": np.bitwise_xor}[op](x, y)
        if op == "OpVectorTimesScalar":
            return V(a[0]) * V(a[1])[:, None]
        if op == "OpMatrixTimesScalar":
            return V(a[0]) * V(a[1])[:, None, None]
        if op == "OpMatrixTimesVector":
            return np.einsum("ncr,nc->nr", V(a[0]), V(a[1]))
        if op == "OpVectorTimesMatrix":
            return np.einsum("nr,ncr->nc", V(a[0]), V(a[1]))
        if op == "OpMatrixTimesMatrix":
            return np.einsum("nkr,nck->ncr", V(a[0]), V(a[1]))
        if op == "OpOuterProduct":
            return np.einsum("nr,nc->ncr", V(a[0]), V(a[1]))
        if op == "OpTranspose":
            return np.swapaxes(V(a[0]), 1, 2)
        if op == "OpDot":
            return np.sum(V(a[0]) * V(a[1]), axis=1)
        if op in ("OpConvertFToS", "OpConvertFToU"):
            return np.trunc(V(a[0])).astype(np.int64)
        if op in ("OpConvertSToF", "OpConvertUToF", "OpFConvert"):
            return V(a[0]).astype(np.float64)
        if op in ("OpUConvert", "OpSConvert"):
            return V(a[0]).astype(np.int64)
        if op == "OpBitcast":
            x = V(a[0])
            tk = self.tk(ins.type) if self.tk(ins.type) != "vector" else self.tk(self.m.types[ins.type].args[0])
            if tk == "float" and x.dtype != np.float64:
                return x.astype(np.int64).astype(np.uint32).view(np.float32).astype(np.float64) \
                    if x.dtype != np.uint32 else x.view(np.float32).astype(np.float64)
            if tk == "int" and x.dtype == np.float64:
                return x.astype(np.float32).view(np.int32).astype(np.int64)
            return x
        if op == "OpCopyObject":
            return V(a[0])
        if op == "OpCompositeConstruct":
            return self._construct(ins.type, [V(x) for x in a], self.n)
        if op == "OpCompositeExtract":
            v = V(a[0])
            for idx in a[1:]:
                v = self._index(v, idx)
            return v
        if op == "OpCompositeInsert":
            return self._update(V(a[1]), list(a[2:]), V(a[0]))
        if op == "OpVectorShuffle":
            x, y = V(a[0]), V(a[1])
            both = np.concatenate([x.reshape(self.n, -1), y.reshape(self.n, -1)], axis=1)
            cols = [both[:, c] if c != 0xFFFFFFFF else np.zeros(self.n) for c in a[2:]]
            return np.stack(cols, axis=1)
        if op == "OpVectorExtractDynamic":
            return self._index(V(a[0]), V(a[1]))
        if op == "OpVectorInsertDynamic":
            return self._update(V(a[0]), [V(a[2])], V(a[1]))
        if op == "OpSelect":
            c, x, y = V(a[0]), V(a[1]), V(a[2])
            if isinstance(x, list):
                if not (c == c.reshape(-1)[0]).all():
                    raise Divergent()
                return x if c.reshape(-1)[0] else y
            c = np.asarray(c)
            if c.ndim == 1 and x.ndim >= 2:
                c = c.reshape((-1,) + (1,) * (x.ndim - 1))
            return np.where(c, x, y)
        cmp = {"OpFOrdEqual": np.equal, "OpFUnordEqual": np.equal, "OpFOrdNotEqual": np.not_equal,
               "OpFUnordNotEqual": np.not_equal, "OpFOrdLessThan": np.less, "OpFUnordLessThan": np.less,
               "OpFOrdGreaterThan": np.greater, "OpFUnordGreaterThan": np.greater,
               "OpFOrdLessThanEqual": np.less_equal, "OpFUnordLessThanEqual": np.less_equal,
               "OpFOrdGreaterThanEqual": np.greater_equal, "OpFUnordGreaterThanEqual": np.greater_equal,
               "OpIEqual": np.equal, "OpINotEqual": np.not_equal, "OpULessThan": np.less,
               "OpSLessThan": np.less, "OpUGreaterThan": np.greater, "OpSGreaterThan": np.greater,
               "OpULessThanEqual": np.less_equal, "OpSLessThanEqual": np.less_equal,
               "OpUGreaterThanEqual": np.greater_equal, "OpSGreaterThanEqual": np.greater_equal,
               "OpLogicalEqual": np.equal, "OpLogicalNotEqual": np.not_equal,
               "OpLogicalAnd": np.logical_and, "OpLogicalOr": np.logical_or}
        if op in cmp:
            x, y = self._bc(V(a[0]), V(a[1]))
            return cmp[op](x, y)
        if op == "OpLogicalNot":
            return np.logical_not(V(a[0]))
        if op == "OpAny":
            return np.any(V(a[0]), axis=1)
        if op == "OpAll":
            return np.all(V(a[0]), axis=1)
        if op in ("OpIsNan", "OpIsInf"):
            return (np.isnan if op == "OpIsNan" else np.isinf)(V(a[0]))
        return NotImplemented

    def glsl(self, name, x, t):
        bc = self._bc
        with np.errstate(all="ignore"):
            one = {"Round": np.round, "RoundEven": np.round, "Trunc": np.trunc, "FAbs": np.abs,
                   "SAbs": np.abs, "FSign": np.sign, "SSign": np.sign, "Floor": np.floor, "Ceil": np.ceil,
                   "Fract": lambda v: v - np.floor(v), "Radians": np.radians, "Degrees": np.degrees,
                   "Sin": np.sin, "Cos": np.cos, "Tan": np.tan, "Asin": np.arcsin, "Acos": np.arccos,
                   "Atan": np.arctan, "Sinh": np.sinh, "Cosh": np.cosh, "Tanh": np.tanh,
                   "Asinh": np.arcsinh, "Acosh": np.arccosh, "Atanh": np.arctanh, "Exp": np.exp,
                   "Log": np.log, "Exp2": np.exp2, "Log2": np.log2, "Sqrt": np.sqrt,
                   "InverseSqrt": lambda v: 1 / np.sqrt(v)}
            if name in one:
                return one[name](x[0])
            if name in ("Pow", "Atan2", "FMin", "UMin", "SMin", "NMin", "FMax", "UMax", "SMax", "NMax",
                        "Step", "Ldexp"):
                p, q = bc(x[0], x[1])
                return {"Pow": np.power, "Atan2": np.arctan2, "FMin": np.minimum, "UMin": np.minimum,
                        "SMin": np.minimum, "NMin": np.fmin, "FMax": np.maximum, "UMax": np.maximum,
                        "SMax": np.maximum, "NMax": np.fmax,
                        "Step": lambda e, v: (v >= e).astype(np.float64),
                        "Ldexp": lambda v, e: v * np.exp2(e)}[name](p, q)
            if name in ("FClamp", "UClamp", "SClamp", "NClamp"):
                v, lo = bc(x[0], x[1])
                v, hi = bc(v, x[2])
                return np.minimum(np.maximum(v, lo), hi)
            if name in ("FMix", "IMix"):
                p, q = bc(x[0], x[1])
                p, s = bc(p, x[2])
                q, s = bc(q, s)
                return p * (1 - s) + q * s
            if name == "SmoothStep":
                e0, e1 = bc(x[0], x[1])
                e0, v = bc(e0, x[2])
                e1, v = bc(e1, v)
                tt = np.clip((v - e0) / (e1 - e0), 0, 1)
                return tt * tt * (3 - 2 * tt)
            if name == "Fma":
                p, q = bc(x[0], x[1])
                p, r = bc(p, x[2])
                return p * q + r
            if name == "Length":
                v = x[0]
                return np.abs(v) if v.ndim == 1 else np.sqrt(np.sum(v * v, axis=1))
            if name == "Distance":
                v = x[0] - x[1]
                return np.abs(v) if v.ndim == 1 else np.sqrt(np.sum(v * v, axis=1))
            if name == "Normalize":
                v = x[0]
                if v.ndim == 1:
                    return np.sign(v)
                return v / np.sqrt(np.sum(v * v, axis=1))[:, None]
            if name == "Cross":
                return np.cross(x[0], x[1])
            if name == "Reflect":
                i, nn = x[0], x[1]
                return i - 2 * np.sum(nn * i, axis=1)[:, None] * nn
            if name == "FaceForward":
                nn, i, nref = x
                return np.where((np.sum(nref * i, axis=1) < 0)[:, None], nn, -nn)
            if name == "Refract":
                i, nn, eta = x
                d = np.sum(nn * i, axis=1)
                k = 1 - eta * eta * (1 - d * d)
                return np.where((k < 0)[:, None], 0.0,
                                eta[:, None] * i - (eta * d + np.sqrt(np.maximum(k, 0)))[:, None] * nn)
            if name == "Determinant":
                return np.linalg.det(np.swapaxes(x[0], 1, 2))
            if name == "MatrixInverse":
                return np.swapaxes(np.linalg.inv(np.swapaxes(x[0], 1, 2)), 1, 2)
        raise Unsupported(f"GLSL.std.450 {name}")


# ---------------------------------------------------------------- running
def run_lanes(interp, fid, n, overrides=None):
    """Vectorised run; on a data-dependent branch, one run per lane (values
    present on every lane are stacked)."""
    try:
        return interp.run(fid, n, overrides)
    except Killed:
        return interp.r
    except Divergent:
        pass
    runs = []
    for i in range(n):
        ov = {k: (v[i:i + 1] if isinstance(v, np.ndarray) else v) for k, v in (overrides or {}).items()}
        sv = interp.source_value
        interp.source_value = (lambda key, t, nn, kind, _i=i, _sv=sv:
                               (lambda got: got[_i:_i + 1] if isinstance(got, np.ndarray) else got)(_sv(key, t, n, kind)))
        try:
            runs.append(interp.run(fid, 1, ov))
        except Killed:
            runs.append(interp.r)
        except Divergent:
            interp.source_value = sv
            raise Unsupported("divergence on one lane")
        finally:
            interp.source_value = sv
    out = Run()
    common = set(runs[0].values)
    for r in runs[1:]:
        common &= set(r.values)
    for k in runs[0].order:
        if k not in common:
            continue
        vals = [r.values[k] for r in runs]
        if all(isinstance(v, np.ndarray) for v in vals):
            out.values[k] = np.concatenate(vals, axis=0)
        else:
            out.values[k] = vals[0]
        out.order.append(k)
        out.types[k] = runs[0].types.get(k)
        out.deps[k] = set().union(*[r.deps.get(k, set()) for r in runs])
        if k in runs[0].sources:
            out.sources[k] = runs[0].sources[k]
    for r in runs:
        out.blocks |= r.blocks
        out.stores += r.stores
        out.loads.update(r.loads)
    return out


@dataclass
class Fingerprint:
    base: Run = None
    heads: set = field(default_factory=set)
    head_of: dict = field(default_factory=dict)      # node -> head
    idom: dict = field(default_factory=dict)
    edges: dict = field(default_factory=dict)        # node -> (dominator key, transfer.Fit)
    opaque: set = field(default_factory=set)         # region nodes with no unambiguous fit
    affine: dict = field(default_factory=dict)       # node -> (dominator key, a, b): exact affine
    constant: set = field(default_factory=set)       # nodes that do not depend on any source
    structural: set = field(default_factory=set)     # data movement (shuffles, inserts ...): no fit
    error: str = None

    def region(self, key):
        """True when key is a non-head node of some region (its transfer label
        comes only from the fingerprint)."""
        return key in self.head_of and key not in self.heads


ROOT = ("ROOT",)
MOVES = {"OpVectorShuffle", "OpCompositeConstruct", "OpCompositeExtract", "OpCompositeInsert",
         "OpCopyObject", "OpLoad", "OpPhi"}


def _float_kind(m, t):
    k = m.tkind(t)
    if k == "float":
        return 1
    if k == "vector" and m.tkind(m.types[t].args[0]) == "float":
        return m.types[t].args[1]
    return 0


def dominators(order, deps, sources):
    """Immediate dominators of the dependence graph with a virtual root that
    feeds every source (Cooper, Harvey and Kennedy's iterative algorithm)."""
    preds = {}
    nodes = [k for k in order if k in deps]
    reach = set(sources)
    for k in nodes:
        if k in sources:
            preds[k] = [ROOT]
            continue
        ps = [d for d in deps[k] if d in deps]
        preds[k] = ps
    # nodes reachable from the root (depend on some source)
    changed = True
    while changed:
        changed = False
        for k in nodes:
            if k not in reach and any(p in reach for p in preds[k]):
                reach.add(k)
                changed = True
    nodes = [k for k in nodes if k in reach]
    idx = {k: i + 1 for i, k in enumerate(nodes)}
    idx[ROOT] = 0
    idom = {ROOT: ROOT}

    def intersect(a, b):
        while a != b:
            while idx[a] > idx[b]:
                a = idom[a]
            while idx[b] > idx[a]:
                b = idom[b]
        return a

    changed = True
    rounds = 0
    while changed and rounds < 50:
        changed = False
        rounds += 1
        for k in nodes:
            ps = [p for p in preds[k] if p in idom and (p == ROOT or p in reach)]
            if not ps:
                continue
            new = ps[0]
            for p in ps[1:]:
                new = intersect(p, new)
            if idom.get(k) != new:
                idom[k] = new
                changed = True
    return idom, reach


def ramp_value(m, t, ramp):
    """Splat the ramp over a float scalar / vector; a 4-vector gets alpha 1."""
    nk = _float_kind(m, t)
    if nk == 1:
        return ramp.copy()
    if nk >= 2:
        v = np.tile(ramp[:, None], (1, nk))
        if nk == 4:
            v[:, 3] = 1.0
        return v
    return None


COLOR_SC = {2, 9, 12}            # Uniform, PushConstant, StorageBuffer


def _root_storage(module, ptr, depth=0):
    d = module.defs.get(ptr)
    while d is not None and d.op in ("OpAccessChain", "OpInBoundsAccessChain", "OpPtrAccessChain",
                                     "OpCopyObject") and depth < 16:
        d = module.defs.get(d.args[0])
        depth += 1
    if d is not None and d.op == "OpVariable" and d.args:
        return d.args[0]
    return None


def pseudo_sources(module, base):
    """Values the fingerprint treats as heads although they are not sources of
    the base run: (a) the float result of converting integer data (packed or
    UINT colours: float(c & 0xff), float4(uint4 colour)), whose slice would
    otherwise sit under an integer head that no ramp can drive; (b) float 3- and
    4-vectors loaded from uniform, push-constant or storage memory whose bytes
    the driver recorded (a tint or cursor colour): they are constants of the
    run, so nothing below them was ever fitted. Both only add slices to fit;
    they do not change any value of the base run."""
    out = set()
    for k in base.order:
        if not (isinstance(k, tuple) and len(k) == 2):
            continue
        d = module.defs.get(k[1])
        if d is None or not _float_kind(module, base.types.get(k)):
            continue
        if d.op in ("OpConvertUToF", "OpConvertSToF"):
            out.add(k)
        elif d.op == "OpLoad" and k not in base.sources and _float_kind(module, base.types.get(k)) >= 3 \
                and _root_storage(module, d.args[0]) in COLOR_SC:
            out.add(k)
    return out


CODE_RANGES = (255.0, 1023.0, 4095.0, 65535.0)


def _is_code_head(h, pseudo, base, module):
    """A head that may hold integer code values: a pseudo-source, or a merge
    of them (float4(r, g, b, a) built from four converted bytes)."""
    if h in pseudo:
        return True
    return any(d in pseudo for d in base.deps.get(h, ()))


def _code_range(run, h, rv, members, module):
    """k when some value below head h equals the head divided by k for a code
    range k (x / 255.0 normalising 8-bit codes), else None."""
    x = np.asarray(rv, dtype=np.float64)
    x0 = x[:, 0] if x.ndim == 2 else x
    for v in members:
        y = run.values.get(v)
        if not isinstance(y, np.ndarray) or y.dtype == bool or y.shape[0] != x0.shape[0]:
            continue
        y0 = y[:, 0] if y.ndim == 2 else y
        if y0.ndim != 1:
            continue
        for k in CODE_RANGES:
            if np.allclose(y0 * k, x0, rtol=1e-5, atol=1e-9):
                return k
    return None


def fingerprint(module, fid, memory=None, source_value=None, ramp=None, force=None):
    """Fingerprint every univariate slice of the entry point fid (module
    docstring). Never raises: on unsupported code returns a Fingerprint with
    .error set and no edges (callers then treat nonlinear ops as opaque).
    force: {key: value} applied to every run (e.g. a branch condition held to
    the side the base run did not take)."""
    ramp = transfer.RAMP if ramp is None else ramp
    n = len(ramp)
    fp = Fingerprint()
    interp = Interp(module, memory, source_value)
    force = dict(force or {})
    try:
        base = run_lanes(interp, fid, n, dict(force) or None)
    except (Unsupported, Divergent, KeyError, IndexError, ValueError, TypeError) as e:
        fp.error = f"{type(e).__name__}: {e}"
        return fp
    fp.base = base
    pseudo = pseudo_sources(module, base)
    sources = {k for k, kind in base.sources.items() if kind != "param"} | pseudo
    idom, reach = dominators(base.order, base.deps, sources)
    fp.idom = idom
    fp.constant = {k for k in base.order if k in base.deps and k not in reach}
    for k in base.order:
        if k not in idom or k == ROOT:
            continue
        if idom[k] == ROOT:
            fp.heads.add(k)
    # head of every node: walk up the dominator tree
    for k in base.order:
        if k not in idom or k == ROOT:
            continue
        h = k
        while idom.get(h, ROOT) != ROOT:
            h = idom[h]
        fp.head_of[k] = h
    pos = {k: i for i, k in enumerate(base.order)}
    by_head = {}
    for k, h in fp.head_of.items():
        if k != h:
            by_head.setdefault(h, []).append(k)
    for h, members in by_head.items():
        t = base.types.get(h)
        rv = ramp_value(module, t, ramp) if t is not None else None
        if rv is None:
            fp.opaque.update(k for k in members if _float_kind(module, base.types.get(k)))
            continue
        try:
            run = run_lanes(interp, fid, n, {**force, h: rv})
            k = _code_range(run, h, rv, members, module) if _is_code_head(h, pseudo, base, module) else None
            if k:
                # integer code values (0..255, 0..1023 ...) normalised further down:
                # ramp the head over the code range so that the normalised value
                # spans [0, 1] (a transfer curve is not scale-free)
                rv = rv * k
                if rv.ndim == 2 and rv.shape[1] == 4:
                    rv[:, 3] = k
                run = run_lanes(interp, fid, n, {**force, h: rv})
        except (Unsupported, Divergent, KeyError, IndexError, ValueError, TypeError):
            fp.opaque.update(members)
            continue
        members.sort(key=lambda k: pos.get(k, 0))
        labelled = {h}
        for v in members:
            if not _float_kind(module, base.types.get(v)):
                continue
            d = module.defs.get(v[1]) if isinstance(v, tuple) and len(v) == 2 else None
            if d is not None and d.op in MOVES:
                # data movement keeps each component's meaning: the caller
                # unifies components structurally; fitting it as a whole
                # would compare unrelated components
                fp.structural.add(v)
                if any(x in labelled for x in base.deps.get(v, ())):
                    labelled.add(v)
                _affine(module, fp, run, v, _chain(idom, v, h), base)
                continue
            chain = _chain(idom, v, h)
            # v is labelled relative to its nearest *labelled* dominator that
            # fits; fitting from the head alone would fit composites (e.g.
            # sRGB decode then PQ encode) against single library curves
            got = None
            for u in chain:
                if u not in labelled or not _float_kind(module, base.types.get(u)):
                    continue
                f = _fit_pair(run, u, v)
                if f is not None:
                    got = (u, f)
                    break
            if got is None:
                fp.opaque.add(v)
            else:
                fp.edges[v] = got
                labelled.add(v)
            _affine(module, fp, run, v, chain, base)
    return fp


def _chain(idom, v, h):
    """v's dominators up to its head h, nearest first."""
    chain, u = [], idom.get(v)
    while u is not None and u != ROOT:
        chain.append(u)
        if u == h:
            break
        u = idom.get(u)
    return chain


def _affine(module, fp, run, v, chain, base):
    """Exact affine relation of scalar v to its farthest scalar dominator
    (range remaps such as depth * 2 - 1 span several instructions, and may
    pass through a local merged by a phi)."""
    if _float_kind(module, base.types.get(v)) != 1 or v not in run.values:
        return
    for u in reversed(chain):
        if u in run.values and _float_kind(module, base.types.get(u)) == 1:
            af = transfer.fit_affine(run.values[u], run.values[v])
            if af is not None:
                fp.affine[v] = (u, af[0], af[1])
                return


def _comps(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return [x]
    return [x[:, i] for i in range(min(3, x.shape[1]))]


def _fit_pair(run, u, v):
    if u not in run.values or v not in run.values:
        return None
    xu, yv = run.values[u], run.values[v]
    if not isinstance(xu, np.ndarray) or not isinstance(yv, np.ndarray) or xu.dtype == bool:
        return None
    xs, ys = _comps(xu), _comps(yv)
    fits = []
    for c, y in enumerate(ys):
        if float(np.ptp(y)) < 1e-9:
            continue                     # constant component (alpha, a zeroed channel)
        x = xs[c] if c < len(xs) else xs[0]
        f = transfer.fit(x, y)
        if f is None:
            return None
        fits.append(f)
    if not fits or any(f.element != fits[0].element for f in fits):
        return None
    return fits[0]


# ---------------------------------------------------------------- witnesses
def affine_map(module, fid, head, n_in, memory=None, source_value=None):
    """Evaluate the entry point with source `head` (an n_in float vector) set
    to 0, e_0 .. e_{n-1} and a random point; returns {node: (A, b)} for every
    float vector node that is affine in head (value = A @ x + b), checked on
    the random point."""
    pts = np.vstack([np.zeros(n_in), np.eye(n_in), np.random.default_rng(7).uniform(-2, 2, n_in)])
    interp = Interp(module, memory, source_value)
    try:
        run = run_lanes(interp, fid, len(pts), {head: pts})
    except (Unsupported, Divergent, KeyError, IndexError, ValueError, TypeError):
        return {}, None
    out = {}
    for k, v in run.values.items():
        if not isinstance(v, np.ndarray) or v.dtype == bool or v.ndim != 2 or v.shape[0] != len(pts):
            continue
        b = v[0]
        A = (v[1:n_in + 1] - b).T
        pred = A @ pts[-1] + b
        if np.allclose(pred, v[-1], rtol=1e-6, atol=1e-6):
            out[k] = (A, b)
    return out, run
