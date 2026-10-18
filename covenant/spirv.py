"""SPIR-V front end: compile Slang/GLSL to SPIR-V and decode the binary.

Every shading language COVENANT supports goes through SPIR-V, so this one
decoder is the shader front end for Slang, GLSL, HLSL and RCGP's generated
GLSL. Meaning seeds come from Slang's reflection JSON (user attributes) or a
GLSL sidecar file (<shader>.cov.json).
"""
import json
import os
import struct
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).parent / "data"
SLANGC = os.environ.get("SLANGC", "/opt/slang/bin/slangc")
GLSLANG = os.environ.get("GLSLANG", "glslangValidator")


@lru_cache(None)
def _grammar():
    core = json.loads((DATA / "spirv.core.grammar.json").read_text())
    ops = {}
    for ins in core["instructions"]:
        kinds = [o["kind"] for o in ins.get("operands", [])]
        ops[ins["opcode"]] = (ins["opname"], "IdResultType" in kinds[:1],
                              "IdResult" in kinds[:2])
    glsl = json.loads((DATA / "extinst.glsl.std.450.grammar.json").read_text())
    ext = {i["opcode"]: i["opname"] for i in glsl["instructions"]}
    return ops, ext


def _string(words):
    raw = b"".join(struct.pack("<I", w) for w in words)
    s = raw.split(b"\0", 1)[0].decode("utf-8", "replace")
    n = len(s.encode()) // 4 + 1
    return s, n


@dataclass
class Ins:
    op: str
    type: int = None          # result type id
    id: int = None            # result id
    args: list = field(default_factory=list)   # remaining raw words
    line: tuple = None        # (file, line) from debug info


@dataclass
class Module:
    path: str
    generator: int = 0
    ins: list = field(default_factory=list)
    names: dict = field(default_factory=dict)          # id -> name
    member_names: dict = field(default_factory=dict)   # (type id, i) -> name
    decos: dict = field(default_factory=dict)          # id -> {deco: [args]}
    member_decos: dict = field(default_factory=dict)   # (type id, i) -> {deco: [args]}
    strings: dict = field(default_factory=dict)        # OpString id -> text
    types: dict = field(default_factory=dict)          # id -> Ins
    consts: dict = field(default_factory=dict)         # id -> python value
    defs: dict = field(default_factory=dict)           # id -> defining Ins
    functions: dict = field(default_factory=dict)      # id -> [Ins] (body)
    params: dict = field(default_factory=dict)         # func id -> [param ids]
    entries: list = field(default_factory=list)        # (model, func id, name, [iface ids])
    glsl_ext: int = None
    meta: dict = field(default_factory=dict)           # reflection / sidecar
    exec_modes: dict = field(default_factory=dict)     # func id -> {mode: [operands]} (e.g. LocalSize)

    # ---------------------------------------------------------- type helpers
    def tkind(self, tid):
        t = self.types.get(tid)
        return t.op[6:].lower() if t else None          # "OpTypeStruct" -> "struct"

    def pointee(self, tid):
        t = self.types[tid]
        return t.args[1] if t.op == "OpTypePointer" else tid

    def member_type(self, tid, i):
        t = self.types[tid]
        if t.op == "OpTypeStruct":
            return t.args[i]
        if t.op in ("OpTypeArray", "OpTypeRuntimeArray", "OpTypeVector", "OpTypeMatrix"):
            return t.args[0]
        return None

    def struct_name(self, tid):
        n = self.names.get(tid, f"struct{tid}")
        for suf in ("_std430_logical", "_std140_logical", "_natural_logical",
                    "_std430", "_std140", "_natural", "_logical"):
            if n.endswith(suf):
                return n[: -len(suf)]
        return n

    def is_matrix(self, tid):
        return self.tkind(tid) == "matrix"

    def is_vector_or_scalar_float(self, tid):
        return self.tkind(tid) in ("vector", "float")

    def name(self, i):
        return self.names.get(i, f"%{i}")

    def storage_class(self, var_id):
        d = self.defs.get(var_id)
        return d.args[0] if d is not None and d.op == "OpVariable" else None


STORAGE = {0: "UniformConstant", 1: "Input", 2: "Uniform", 3: "Output", 6: "Private",
           7: "Function", 9: "PushConstant", 12: "StorageBuffer"}
DECO = {30: "Location", 33: "Binding", 34: "DescriptorSet", 35: "Offset", 11: "BuiltIn",
        6: "ArrayStride", 7: "MatrixStride", 2: "Block", 3: "BufferBlock", 22: "UserSemantic"}


def parse(path):
    data = Path(path).read_bytes()
    words = struct.unpack(f"<{len(data) // 4}I", data)
    assert words[0] == 0x07230203, f"{path}: not SPIR-V"
    ops, ext = _grammar()
    m = Module(str(path), generator=words[2] >> 16)
    i, cur_fn, line = 5, None, None
    debug_set = None
    while i < len(words):
        n, opcode = words[i] >> 16, words[i] & 0xFFFF
        body = list(words[i + 1:i + n])
        i += n
        opname, has_t, has_r = ops.get(opcode, (f"Op{opcode}", False, False))
        ins = Ins(opname)
        k = 0
        if has_t:
            ins.type, k = body[0], 1
        if has_r:
            ins.id, k = body[k], k + 1
        ins.args = body[k:]
        if opname == "OpName":
            m.names[body[0]] = _string(body[1:])[0]
        elif opname == "OpMemberName":
            m.member_names[(body[0], body[1])] = _string(body[2:])[0]
        elif opname == "OpString":
            m.strings[ins.id] = _string(ins.args)[0]
        elif opname == "OpDecorate":
            m.decos.setdefault(body[0], {})[DECO.get(body[1], body[1])] = body[2:]
        elif opname == "OpDecorateString":
            m.decos.setdefault(body[0], {})[DECO.get(body[1], body[1])] = [_string(body[2:])[0]]
        elif opname in ("OpMemberDecorate", "OpMemberDecorateString"):
            v = body[3:] if opname == "OpMemberDecorate" else [_string(body[3:])[0]]
            m.member_decos.setdefault((body[0], body[1]), {})[DECO.get(body[2], body[2])] = v
        elif opname == "OpExtInstImport":
            name = _string(ins.args)[0]
            if name == "GLSL.std.450":
                m.glsl_ext = ins.id
            elif name.startswith("NonSemantic.Shader.DebugInfo"):
                debug_set = ins.id
        elif opname in ("OpExecutionMode", "OpExecutionModeId"):
            m.exec_modes.setdefault(body[0], {})[body[1]] = body[2:]
        elif opname == "OpEntryPoint":
            model, fid = body[0], body[1]
            name, ln = _string(body[2:])
            m.entries.append((model, fid, name, body[2 + ln:]))
        elif opname.startswith("OpType"):
            m.types[ins.id] = ins
        elif opname in ("OpConstant", "OpSpecConstant"):
            t = m.types.get(ins.type)
            if t is not None and t.op == "OpTypeFloat":
                m.consts[ins.id] = struct.unpack("<f", struct.pack("<I", ins.args[0]))[0]
            else:
                m.consts[ins.id] = ins.args[0] if ins.args else 0
        elif opname in ("OpConstantComposite", "OpConstantNull", "OpConstantTrue",
                        "OpConstantFalse", "OpUndef"):
            m.consts[ins.id] = None
        elif opname == "OpFunction":
            cur_fn = ins.id
            m.functions[cur_fn] = []
            m.params[cur_fn] = []
        elif opname == "OpFunctionEnd":
            cur_fn = None
        elif opname == "OpLine":
            line = (m.strings.get(body[0], "?"), body[1])
        elif opname == "OpNoLine":
            line = None
        if opname == "OpExtInst" and debug_set is not None and ins.args[0] == debug_set:
            # NonSemantic DebugLine: Source, LineStart, LineEnd, ColStart, ColEnd
            if ins.args[1] == 103:          # DebugLine
                src = m.defs.get(ins.args[2])
                fname = m.strings.get(src.args[2], "?") if src is not None else "?"
                line = (fname, m.consts.get(ins.args[3], 0))
            elif ins.args[1] == 104:        # DebugNoLine
                line = None
            if ins.id is not None:
                m.defs[ins.id] = ins
            continue
        if ins.id is not None:
            m.defs[ins.id] = ins
        if cur_fn is not None and opname != "OpFunction":
            if opname == "OpFunctionParameter":
                m.params[cur_fn].append(ins.id)
            ins.line = line
            m.functions[cur_fn].append(ins)
    return m


# ------------------------------------------------------------------ compile
_PTR_OPS = {"OpAccessChain", "OpInBoundsAccessChain", "OpPtrAccessChain",
            "OpInBoundsPtrAccessChain", "OpCopyObject", "OpBitcast", "OpLoad",
            "OpImageTexelPointer"}      # image atomics go through a texel pointer
_WRITE_PTR = {"OpStore", "OpAtomicStore", "OpAtomicExchange", "OpAtomicCompareExchange",
              "OpAtomicIIncrement", "OpAtomicIDecrement", "OpAtomicIAdd", "OpAtomicISub",
              "OpAtomicSMin", "OpAtomicUMin", "OpAtomicSMax", "OpAtomicUMax", "OpAtomicAnd",
              "OpAtomicOr", "OpAtomicXor", "OpAtomicFAddEXT", "OpAtomicFMinEXT",
              "OpAtomicFMaxEXT", "OpCopyMemory"}


def entry_bindings(m, entry):
    """{(set, binding): written} for the descriptor resources the entry point
    can reach (through its call graph). Resources it never touches are absent.
    "written" means some reachable store, atomic or image write targets it."""
    fid = next((f for _, f, n, _ in m.entries if n == entry), None)
    if fid is None:
        return None
    var_key = {v: (d.get("DescriptorSet", [0])[0], d["Binding"][0])
               for v, d in m.decos.items() if "Binding" in d}
    reach, todo = set(), [fid]
    while todo:
        f = todo.pop()
        if f in reach or f not in m.functions:
            continue
        reach.add(f)
        todo += [i.args[0] for i in m.functions[f] if i.op == "OpFunctionCall"]
    root = {v: v for v in var_key}            # id -> descriptor variable it derives from
    out = {}
    for _ in range(4):                        # params bound through calls need a few rounds
        for f in reach:
            for i in m.functions[f]:
                if i.op in _PTR_OPS and i.args and i.args[0] in root:
                    root[i.id] = root[i.args[0]]
                elif i.op == "OpFunctionCall":
                    for prm, arg in zip(m.params.get(i.args[0], []), i.args[1:]):
                        if arg in root:
                            root[prm] = root[arg]
    for f in reach:
        for i in m.functions[f]:
            if i.op in _NO_ACCESS:
                continue                    # literal operands (OpLine line numbers) are no ids
            for a in i.args:
                if a in root:
                    out.setdefault(var_key[root[a]], False)
            tgt = i.args[0] if i.args and (i.op in _WRITE_PTR or i.op == "OpImageWrite") else None
            if tgt in root:
                out[var_key[root[tgt]]] = True
    return out


_NO_ACCESS = {"OpLine", "OpNoLine", "OpSelectionMerge", "OpLoopMerge", "OpBranch", "OpBranchConditional",
              "OpSwitch", "OpLabel", "OpName", "OpMemberName", "OpDecorate", "OpMemberDecorate", "OpReturn",
              "OpReturnValue", "OpKill", "OpUnreachable", "OpModuleProcessed", "OpSource",
              "OpSourceContinued", "OpString"}
_CHAIN_OPS = ("OpAccessChain", "OpInBoundsAccessChain", "OpPtrAccessChain", "OpInBoundsPtrAccessChain")


def entry_binding_modes(m, entry):
    """{(set, binding): frozenset((class, member))} for written descriptor
    resources whose every access in the entry point is a plain load ("read")
    or

    * "atomic": an atomic instruction (OpAtomic*, image atomics through
      OpImageTexelPointer included). Two atomic operations never form a data
      race (Vulkan memory model), so atomic accesses of two commands on one
      queue need no barrier between them;
    * "slot": a load/store through an access chain with an index
      computed from an atomic's result (a slot allocated with atomicAdd, e.g. an
      order-independent-transparency node list). Two commands that only take
      fresh slots touch disjoint elements. This is a heuristic (flow-insensitive:
      a function variable once assigned an atomic's result counts as one).

    member: the constant index of the top-level block member accessed (None:
    unknown, or an image). Bindings with any other access (plain store, image
    read/write, ...) are absent. See modes_compatible."""
    fid = next((f for _, f, n, _ in m.entries if n == entry), None)
    if fid is None:
        return {}
    var_key = {v: (d.get("DescriptorSet", [0])[0], d["Binding"][0])
               for v, d in m.decos.items() if "Binding" in d}
    reach, todo = set(), [fid]
    while todo:
        f = todo.pop()
        if f in reach or f not in m.functions:
            continue
        reach.add(f)
        todo += [i.args[0] for i in m.functions[f] if i.op == "OpFunctionCall"]
    body = [i for f in reach for i in m.functions[f]]
    root = {v: v for v in var_key}
    handle = ("image", "sampledimage", "sampler", "accelerationstructurekhr", "pointer")
    for _ in range(4):
        for i in body:
            if i.op in _PTR_OPS and i.args and i.args[0] in root and \
                    (i.op != "OpLoad" or m.tkind(i.type) in handle):
                root[i.id] = root[i.args[0]]
            elif i.op == "OpFunctionCall":
                for prm, arg in zip(m.params.get(i.args[0], []), i.args[1:]):
                    if arg in root:
                        root[prm] = root[arg]
    # values computed from an atomic's result (through function / private variables too)
    derived, dvars = set(), set()
    for i in body:
        if i.op.startswith("OpAtomic") and i.id is not None:
            derived.add(i.id)
    for _ in range(8):
        n = len(derived) + len(dvars)
        for i in body:
            if i.op == "OpStore" and len(i.args) > 1 and i.args[1] in derived and i.args[0] not in root:
                dvars.add(i.args[0])
            elif i.op == "OpLoad" and i.args and i.args[0] in dvars:
                derived.add(i.id)
            elif i.op in _CHAIN_OPS and i.args and i.args[0] in dvars:
                dvars.add(i.id)
            elif i.id is not None and i.op != "OpLoad" and any(a in derived for a in i.args):
                derived.add(i.id)
        if len(derived) + len(dvars) == n:
            break

    def slot(ptr):
        for _ in range(64):
            d = m.defs.get(ptr)
            if d is None or d.op not in _CHAIN_OPS:
                return False
            if any(a in derived for a in d.args[1:]):
                return True
            ptr = d.args[0]
        return False

    def member(ptr):
        """First member index below the descriptor variable (None: unknown / whole)."""
        top = None
        for _ in range(64):
            d = m.defs.get(ptr)
            if d is None or d.op not in _CHAIN_OPS:
                break
            top, ptr = d, d.args[0]
        if top is None or len(top.args) < 2 or top.args[0] not in var_key:
            return None
        c = m.consts.get(top.args[1])
        return c if isinstance(c, int) else None

    modes, written = {}, set()
    for i in body:
        if i.op in _NO_ACCESS or (i.op == "OpExtInst" and i.args and i.args[0] != m.glsl_ext):
            continue                        # literal operands (line numbers, ...) are no ids
        refs = [a for a in i.args if a in root]
        if not refs or i.op == "OpFunctionCall":
            continue                        # passes the pointer on: its uses are in the callee
        if i.op in _PTR_OPS and i.id in root:
            continue                        # derives a pointer / loads a descriptor handle
        key = var_key[root[refs[0]]]
        ptr = i.args[0]
        if i.op.startswith("OpAtomic") and ptr in root:
            cls = "atomic"
            if i.op != "OpAtomicLoad":
                written.add(key)
        elif i.op in ("OpStore", "OpLoad") and ptr in root and slot(ptr):
            cls = "slot"
            if i.op == "OpStore":
                written.add(key)
        elif i.op == "OpLoad" and ptr in root:
            cls = "read"
        elif i.op == "OpArrayLength":
            continue
        else:
            cls = None                      # plain store, image access, ...
        if cls is None or modes.get(key, ()) is None:
            modes[key] = None
            continue
        d = m.defs.get(ptr)
        mem = None if d is not None and d.op == "OpImageTexelPointer" else member(ptr)
        modes[key] = modes.get(key, frozenset()) | {(cls, mem)}
    return {k: v for k, v in modes.items() if v and k in written}


def modes_compatible(a, b):
    """Two accesses with entry_binding_modes a and b (sets of (class, member))
    do not race: on every member both touch, both are atomics, both take
    atomically allocated slots, or both only read (member None overlaps all)."""
    if not a or not b:
        return False
    for c1, m1 in a:
        for c2, m2 in b:
            if m1 is not None and m2 is not None and m1 != m2:
                continue
            if c1 != c2:
                return False
    return True


def used_members(m):
    """{struct name: set of member names read or written through access chains},
    with None for a struct that is loaded or stored whole (every member used)."""
    out = {}
    for f in m.functions.values():
        for i in f:
            if i.op in ("OpAccessChain", "OpInBoundsAccessChain") and i.args:
                base = m.defs.get(i.args[0])
                if base is None or base.type not in m.types:
                    continue
                tid = m.pointee(base.type)
                for k in i.args[1:]:
                    while m.tkind(tid) in ("array", "runtimearray"):
                        tid = m.types[tid].args[0]
                        break
                    if m.tkind(tid) != "struct":
                        break
                    idx = m.consts.get(k)
                    if not isinstance(idx, int):
                        break
                    name = m.struct_name(tid)
                    if out.get(name, set()) is not None:
                        out.setdefault(name, set()).add(m.member_names.get((tid, idx), str(idx)))
                    tid = m.types[tid].args[idx]
            elif i.op == "OpLoad" and i.type in m.types and m.tkind(i.type) == "struct":
                out[m.struct_name(i.type)] = None
    return out


def compile_slang(src, out_dir, extra=()):
    """Compile every [shader] entry point in a Slang file into one SPIR-V
    module (+ reflection JSON). Returns the parsed Module."""
    src, out_dir = Path(src), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spv = out_dir / (src.stem + ".spv")
    refl = out_dir / (src.stem + ".refl.json")
    inc = Path(__file__).parent / "slang"
    cmd = [SLANGC, str(src), "-target", "spirv", "-g2", "-fvk-use-entrypoint-name", "-I", str(inc),
           "-o", str(spv), "-reflection-json", str(refl), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"slangc failed on {src}:\n{r.stderr}")
    m = parse(spv)
    m.meta = seeds_from_reflection(json.loads(refl.read_text()))
    m.meta["source"] = str(src)
    return m


def compile_glsl(src, out_dir):
    src, out_dir = Path(src), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spv = out_dir / (src.name + ".spv")
    r = subprocess.run([GLSLANG, "-V", "-g", str(src), "-o", str(spv)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"glslang failed on {src}:\n{r.stdout}{r.stderr}")
    m = parse(spv)
    side = Path(str(src) + ".cov.json")
    m.meta = seeds_from_sidecar(json.loads(side.read_text())) if side.exists() else _empty_seeds()
    m.meta["source"] = str(src)
    return m


# ------------------------------------------------------------------- seeds
def _empty_seeds():
    return {"fields": {}, "globals": {}, "varyings": {}}


def _attrs(lst):
    out = {}
    for a in lst or ():
        n, args = a["name"].lower(), a.get("arguments", [])
        if n in ("frame", "unit", "color"):
            out[n] = args[0]
        elif n == "maps":
            out["from"], out["to"] = args[0], args[1]
        elif n == "convention":
            k, _, v = args[0].partition("=")
            out[f"conv.{k}"] = v
    return out


def seeds_from_reflection(refl):
    """{"fields": {(struct, member): tag}, "globals": {name: tag}}"""
    seeds = _empty_seeds()

    def walk_type(t):
        if not isinstance(t, dict):
            return
        if t.get("kind") == "struct":
            for f in t.get("fields", []):
                tag = _attrs(f.get("userAttribs"))
                if tag:
                    seeds["fields"][(t["name"], f["name"])] = tag
                walk_type(f.get("type"))
        for k in ("elementType", "resultType", "type"):
            walk_type(t.get(k))
        walk_type(t.get("elementVarLayout", {}).get("type"))

    for p in refl.get("parameters", []):
        tag = _attrs(p.get("userAttribs"))
        if tag:
            seeds["globals"][p["name"]] = tag
        walk_type(p.get("type"))
    for e in refl.get("entryPoints", []):
        for p in e.get("parameters", []):
            walk_type(p.get("type"))
            t = p.get("type", {})
            if t.get("kind") == "struct":
                for f in t.get("fields", []):
                    seeds["varyings"][f"{p['name']}.{f['name']}"] = (t["name"], f["name"])
            tag = _attrs(p.get("userAttribs"))
            if tag:
                seeds["varyings_tags"] = seeds.get("varyings_tags", {})
                seeds["varyings_tags"][p["name"]] = tag
        res = e.get("result", {}).get("type", {})
        walk_type(res)
        if res.get("kind") == "struct":
            for f in res.get("fields", []):
                seeds["varyings"][f"entryPointParam_{e['name']}.{f['name']}"] = (res["name"], f["name"])
    return seeds


def seeds_from_sidecar(d):
    seeds = _empty_seeds()
    for k, tag in d.items():
        if "." in k:
            s, f = k.split(".", 1)
            seeds["fields"][(s, f)] = tag
        else:
            seeds["globals"][k] = tag
    return seeds
