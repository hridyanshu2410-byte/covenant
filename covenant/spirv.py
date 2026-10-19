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
def _operand_kinds():
    """opname -> [(kind, quantifier)] of the operands after the result type and
    id; and the kinds that are literals or enumerants without id parameters."""
    core = json.loads((DATA / "spirv.core.grammar.json").read_text())
    ops = {}
    for ins in core["instructions"]:
        ops[ins["opname"]] = [(o["kind"], o.get("quantifier")) for o in ins.get("operands", [])
                              if o["kind"] not in ("IdResultType", "IdResult")]
    plain = {}
    for k in core["operand_kinds"]:
        if k["category"] == "Literal":
            plain[k["kind"]] = "string" if k["kind"] == "LiteralString" else 1
        elif k["category"] in ("ValueEnum", "BitEnum"):
            params = any(e.get("parameters") for e in k.get("enumerants", []))
            plain[k["kind"]] = "enum+" if params else 1
    return ops, plain


def _id_args(i):
    """The operands of i that can be ids: literal and enumerant words are left
    out where the grammar fixes their position; after an operand of unknown
    length (an enumerant with parameters, an unknown opcode) every remaining
    word counts."""
    ops, plain = _operand_kinds()
    kinds = ops.get(i.op)
    if kinds is None:
        return list(i.args)
    out, k = [], 0
    for kind, q in kinds:
        if k >= len(i.args):
            break
        how = plain.get(kind)
        if how is None:                     # an id (IdRef, IdScope, ...) or a pair / composite
            if kind.startswith("Id") and q != "*":
                out.append(i.args[k])
                k += 1
                continue
            return out + list(i.args[k:])   # a list of ids, pairs, composites: every word may be one
        if how == "string":
            k += _string(i.args[k:])[1]
        elif how == "enum+":
            return out + list(i.args[k + 1:])
        else:
            k += len(i.args) - k if q == "*" else 1
    return out


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
    nonsemantic: set = field(default_factory=set)      # NonSemantic.* extended instruction sets
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
            if name.startswith("NonSemantic."):
                m.nonsemantic.add(ins.id)
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
            "OpImageTexelPointer",      # image atomics go through a texel pointer
            "OpSampledImage", "OpImage"}
# The memory-effect inventory. Every instruction that takes a descriptor
# pointer or handle as an operand is in exactly one of these classes; any other
# instruction on one (an extension's load or store, untyped pointers, a pointer
# converted to an integer, ...) makes the analysis incomplete, so the caller
# fails closed instead of reading an unknown effect as a read.
_WRITE_PTR = {"OpStore", "OpAtomicStore", "OpAtomicExchange", "OpAtomicCompareExchange",
              "OpAtomicCompareExchangeWeak", "OpAtomicIIncrement", "OpAtomicIDecrement",
              "OpAtomicIAdd", "OpAtomicISub", "OpAtomicSMin", "OpAtomicUMin", "OpAtomicSMax",
              "OpAtomicUMax", "OpAtomicAnd", "OpAtomicOr", "OpAtomicXor", "OpAtomicFAddEXT",
              "OpAtomicFMinEXT", "OpAtomicFMaxEXT", "OpAtomicFlagTestAndSet", "OpAtomicFlagClear",
              "OpCopyMemory", "OpCopyMemorySized", "OpImageWrite",
              "OpCooperativeMatrixStoreKHR", "OpCooperativeMatrixStoreNV",
              "OpCooperativeMatrixStoreTensorNV", "OpCooperativeVectorStoreNV",
              "OpCooperativeVectorReduceSumAccumulateNV",
              "OpCooperativeVectorOuterProductAccumulateNV"}    # first operand: the target written
                                                                # (a copy's second, its source, is read)
_READ_OPS = {"OpLoad", "OpAtomicLoad", "OpArrayLength", "OpImageRead", "OpImageFetch",
             "OpImageGather", "OpImageDrefGather", "OpCooperativeMatrixLoadKHR",
             "OpCooperativeMatrixLoadNV", "OpCooperativeMatrixLoadTensorNV",
             "OpCooperativeVectorLoadNV", "OpCooperativeVectorMatrixMulNV",
             "OpCooperativeVectorMatrixMulAddNV", "OpTraceRayKHR", "OpTraceNV",
             "OpTraceRayMotionNV", "OpRayQueryInitializeKHR", "OpHitObjectTraceRayNV",
             "OpHitObjectTraceRayMotionNV", "OpFragmentFetchAMD", "OpFragmentMaskFetchAMD"}
_READ_PREFIX = ("OpImageSample", "OpImageSparse", "OpImageQuery", "OpImageBlockMatch",
                "OpImageBoxFilter")
_FLOW_OPS = _PTR_OPS | {"OpPhi", "OpSelect", "OpCopyLogical", "OpFunctionCall", "OpReturnValue",
                        "OpPtrEqual", "OpPtrNotEqual", "OpPtrDiff"}    # pass a pointer on
_HANDLE_KINDS = ("pointer", "untypedpointerkhr", "image", "sampledimage", "sampler",
                 "accelerationstructurekhr")


class AnalysisIncomplete(ValueError):
    """The entry point accesses descriptor memory through a value the provenance
    analysis does not attribute to a descriptor; callers fail closed."""


_DESC_CLASSES = (0, 2, 11, 12)      # UniformConstant, Uniform, Image, StorageBuffer


def _reach(m, fid):
    reach, todo = set(), [fid]
    while todo:
        f = todo.pop()
        if f in reach or f not in m.functions:
            continue
        reach.add(f)
        todo += [i.args[0] for i in m.functions[f] if i.op == "OpFunctionCall"]
    return reach


def _descriptor_roots(m, reach, var_key, through_load):
    """{id: frozenset of descriptor variables it may derive from}, to a fixpoint.
    A value keeps every root it can have: a helper parameter called with A at one
    call site and with B at another derives from both. Follows access chains,
    copies, OpPhi / OpSelect, function parameters and return values, and
    pointers stored to and loaded from variables. through_load(i): whether an
    OpLoad result derives from its pointer's roots."""
    root = {v: frozenset((v,)) for v in var_key}
    rets = {}                                   # function id -> roots of its return values
    body = [(f, i) for f in reach for i in m.functions[f]]

    def add(k, r):
        if r and not r <= root.get(k, frozenset()):
            root[k] = root.get(k, frozenset()) | r
            return True
        return False

    changed = True
    while changed:
        changed = False
        for f, i in body:
            a = i.args
            if i.op in _PTR_OPS and a and a[0] in root and (i.op != "OpLoad" or through_load(i)):
                changed |= add(i.id, root[a[0]])
            elif i.op == "OpPhi":
                changed |= add(i.id, frozenset().union(*(root.get(v, frozenset()) for v in a[0::2])))
            elif i.op == "OpSelect" and len(a) > 2:
                changed |= add(i.id, root.get(a[1], frozenset()) | root.get(a[2], frozenset()))
            elif i.op == "OpCopyLogical" and a:
                changed |= add(i.id, root.get(a[0], frozenset()))
            elif i.op == "OpFunctionCall":
                for prm, arg in zip(m.params.get(a[0], []), a[1:]):
                    changed |= add(prm, root.get(arg, frozenset()))
                changed |= add(i.id, rets.get(a[0], frozenset()))
            elif i.op == "OpReturnValue" and a and a[0] in root:
                if not root[a[0]] <= rets.get(f, frozenset()):
                    rets[f] = rets.get(f, frozenset()) | root[a[0]]
                    changed = True
            elif i.op == "OpStore" and len(a) > 1 and a[1] in root and \
                    m.tkind(m.defs[a[1]].type if a[1] in m.defs else None) == "pointer":
                changed |= add(a[0], root[a[1]])        # a pointer kept in a variable
    return root


def _effect(m, i):
    """The memory effect of instruction i on its pointer and handle operands:
    "write" (the first operand is written; a copy also reads its second),
    "read", "flow" (passes a pointer or handle on), "none" (debug information)
    or None (not in the inventory)."""
    op = i.op
    if op == "OpExtInst":
        if i.args and i.args[0] == m.glsl_ext:
            return "write"                  # Modf / Frexp store through a pointer operand
        return "none" if i.args and i.args[0] in m.nonsemantic else None
    if op in _WRITE_PTR or (op.startswith("OpAtomic") and op != "OpAtomicLoad"):
        return "write"
    if op in _READ_OPS or op.startswith(_READ_PREFIX):
        return "read"
    if op in _FLOW_OPS:
        return "flow"
    if op in _NO_ACCESS or op == "OpVariable":
        return "none"                       # OpVariable: a storage-class literal, an initializer
    return None


def _is_handle(m, a):
    d = m.defs.get(a)
    return d is not None and m.tkind(d.type) in _HANDLE_KINDS


def _is_descriptor_memory(m, a):
    """a is a pointer into descriptor memory or a descriptor handle (image,
    sampled image, acceleration structure)."""
    d = m.defs.get(a)
    t = m.types.get(d.type) if d is not None else None
    if t is None:
        return False
    if t.op == "OpTypePointer":
        return t.args[0] in _DESC_CLASSES
    return m.tkind(d.type) in ("image", "sampledimage", "accelerationstructurekhr", "untypedpointerkhr")


def _check_attributed(m, reach, root):
    """Raise AnalysisIncomplete if an instruction reaches descriptor memory
    through a pointer or handle the analysis did not attribute, or applies an
    effect outside the inventory to a descriptor pointer or handle."""
    for f in reach:
        for i in m.functions[f]:
            eff = _effect(m, i)
            if eff in ("flow", "none"):
                continue
            for a in _id_args(i):
                if a in root:
                    if eff is None and _is_handle(m, a):
                        raise AnalysisIncomplete(f"{i.op} on descriptor memory (%{a}) is not modelled")
                elif _is_descriptor_memory(m, a):
                    raise AnalysisIncomplete(f"{i.op} through %{a} is not attributed to a descriptor")


def entry_bindings(m, entry):
    """{(set, binding): written} for the descriptor resources the entry point
    can reach (through its call graph). Resources it never touches are absent.
    "written" means some reachable instruction of the inventory's write class
    (store, atomic, copy target, image write, cooperative store, ...) targets
    it. Raises AnalysisIncomplete when an access cannot be attributed or an
    instruction outside the inventory touches descriptor memory."""
    fid = next((f for _, f, n, _ in m.entries if n == entry), None)
    if fid is None:
        return None
    var_key = {v: (d.get("DescriptorSet", [0])[0], d["Binding"][0])
               for v, d in m.decos.items() if "Binding" in d}
    reach = _reach(m, fid)
    root = _descriptor_roots(m, reach, var_key, lambda i: True)
    _check_attributed(m, reach, root)
    out = {}
    for f in reach:
        for i in m.functions[f]:
            if i.op in _NO_ACCESS:
                continue                    # literal operands (OpLine line numbers) are no ids
            for a in i.args:
                for v in root.get(a, ()):
                    out.setdefault(var_key[v], False)
            tgt = i.args[0] if i.args and _effect(m, i) == "write" else None
            if i.op == "OpExtInst":         # GLSL.std.450: every pointer operand may be stored to
                tgts = [a for a in i.args[2:] if a in root and _is_handle(m, a)]
            elif i.op == "OpStore" and len(i.args) > 1 and i.args[1] in root and _is_handle(m, i.args[1]) \
                    and i.args[0] not in root:
                tgts = []                   # a pointer kept in a variable: no memory effect
            else:
                tgts = [tgt]
            for t in tgts:
                for v in root.get(t, ()):
                    out[var_key[v]] = True
    return out


_NO_ACCESS = {"OpLine", "OpNoLine", "OpSelectionMerge", "OpLoopMerge", "OpBranch", "OpBranchConditional",
              "OpSwitch", "OpLabel", "OpName", "OpMemberName", "OpDecorate", "OpMemberDecorate", "OpReturn",
              "OpReturnValue", "OpKill", "OpUnreachable", "OpModuleProcessed", "OpSource",
              "OpSourceContinued", "OpString"}
_CHAIN_OPS = ("OpAccessChain", "OpInBoundsAccessChain", "OpPtrAccessChain", "OpInBoundsPtrAccessChain")
_PTR_CHAIN_OPS = ("OpPtrAccessChain", "OpInBoundsPtrAccessChain")   # Base, Element, Indexes...
_DEVICE_SCOPES = (0, 1, 5)     # Scope CrossDevice, Device, QueueFamily: every invocation of every command


def entry_binding_modes(m, entry, spans=False):
    """{(set, binding): frozenset((class, member))} for written descriptor
    resources whose every access in the entry point is a plain load ("read")
    or

    * "atomic": an atomic instruction (OpAtomic*, image atomics through
      OpImageTexelPointer included) whose memory scope is a constant that
      includes every invocation of the device (CrossDevice, Device or
      QueueFamily). Two such atomic operations never form a data race (Vulkan
      memory model), so atomic accesses of two commands on one queue need no
      barrier between them. An atomic with a narrower or unknown scope is an
      ordinary access;
    * "slot": a load/store through an access chain with an index
      computed from an atomic's result (a slot allocated with atomicAdd, e.g. an
      order-independent-transparency node list). Distinct slots are assumed, not
      established (atomicAdd(c, 0) hands every caller the same index), so slot
      accesses are exempt only under the opt-in heuristic (modes_compatible
      slots=True, COVENANT_HEURISTIC_SLOTS=1); flow-insensitive: a function
      variable once assigned an atomic's result counts as one.

    member: the constant index of the top-level block member accessed, or with
    spans=True its byte span (lo, hi) within the block, from the Offset
    decorations (hi None: to the end of the range). None: unknown, a block
    array, an image, or a pointer that reaches the access through a helper
    parameter. The loader turns spans into buffer bytes by adding the
    descriptor's offset (physical_modes). Bindings with any other access (plain
    store, image read/write, ...) are absent. See modes_compatible."""
    fid = next((f for _, f, n, _ in m.entries if n == entry), None)
    if fid is None:
        return {}
    var_key = {v: (d.get("DescriptorSet", [0])[0], d["Binding"][0])
               for v, d in m.decos.items() if "Binding" in d}
    reach = _reach(m, fid)
    body = [i for f in reach for i in m.functions[f]]
    handle = ("image", "sampledimage", "sampler", "accelerationstructurekhr", "pointer")
    root = _descriptor_roots(m, reach, var_key, lambda i: m.tkind(i.type) in handle)
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
        """First member below the descriptor variable: its index, or with spans
        its byte span (lo, hi) in the block (hi None: to the end of the range).
        None: unknown, the whole block, or a block array. OpPtrAccessChain and
        OpInBoundsPtrAccessChain carry an Element operand before their indexes,
        which moves the address by whole elements of their base type: only a
        constant Element 0 keeps the access inside the member."""
        top = None
        for _ in range(64):
            d = m.defs.get(ptr)
            if d is None or d.op not in _CHAIN_OPS:
                break
            if d.op in _PTR_CHAIN_OPS and m.consts.get(d.args[1]) != 0:
                return None                 # an Element that may step out of the member
            top, ptr = d, d.args[0]
        first = 2 if top is not None and top.op in _PTR_CHAIN_OPS else 1
        if top is None or len(top.args) <= first or top.args[0] not in var_key:
            return None
        c = m.consts.get(top.args[first])
        if not isinstance(c, int):
            return None
        if not spans:
            return c
        var = m.defs.get(top.args[0])
        st = m.pointee(var.type) if var is not None and var.type in m.types else None
        if m.tkind(st) != "struct":
            return None                     # a descriptor array: the index is no member
        offs = [m.member_decos.get((st, k), {}).get("Offset", [None])[0]
                for k in range(len(m.types[st].args))]
        if c >= len(offs) or offs[c] is None:
            return None
        hi = min((o for o in offs if o is not None and o > offs[c]), default=None)
        return (offs[c], hi)

    modes, written = {}, set()
    for i in body:
        if i.op in _NO_ACCESS or (i.op == "OpExtInst" and i.args and i.args[0] != m.glsl_ext):
            continue                        # literal operands (line numbers, ...) are no ids
        refs = [a for a in i.args if a in root]
        if not refs or i.op == "OpFunctionCall":
            continue                        # passes the pointer on: its uses are in the callee
        if i.op in _PTR_OPS and i.id in root:
            continue                        # derives a pointer / loads a descriptor handle
        keys = {var_key[v] for a in refs for v in root[a]}
        ptr = i.args[0]
        if i.op.startswith("OpAtomic") and ptr in root and \
                len(i.args) > 1 and m.consts.get(i.args[1]) in _DEVICE_SCOPES:
            cls = "atomic"
            if i.op != "OpAtomicLoad":
                written |= keys
        elif i.op in ("OpStore", "OpLoad") and ptr in root and slot(ptr):
            cls = "slot"
            if i.op == "OpStore":
                written |= keys
        elif i.op == "OpLoad" and ptr in root:
            cls = "read"
        elif i.op == "OpArrayLength":
            continue
        else:
            cls = None                      # plain store, image access, ...
        d = m.defs.get(ptr)
        mem = None if d is not None and d.op == "OpImageTexelPointer" else member(ptr)
        for key in keys:
            if cls is None or modes.get(key, ()) is None:
                modes[key] = None
            else:
                modes[key] = modes.get(key, frozenset()) | {(cls, mem)}
    return {k: v for k, v in modes.items() if v and k in written}


def physical_modes(modes, buf, base):
    """entry_binding_modes(spans=True) of one descriptor as buffer locations:
    (class, (buf, base + lo, base + hi or None)); a member None stays None
    (the whole range). base None (offset unknown): every member becomes None."""
    if modes is None:
        return None
    out = set()
    for cls, sp in modes:
        if sp is None or base is None or not isinstance(sp, tuple):
            out.add((cls, None))
        else:
            out.add((cls, (buf, base + sp[0], None if sp[1] is None else base + sp[1])))
    return frozenset(out)


def _overlap(m1, m2):
    if m1 is None or m2 is None:
        return True
    if isinstance(m1, tuple) and isinstance(m2, tuple):
        if m1[0] != m2[0]:
            return True                     # different buffers: memory may still alias
        lo, hi = max(m1[1], m2[1]), min(x for x in (m1[2], m2[2], float("inf")) if x is not None)
        return lo < hi
    if isinstance(m1, tuple) or isinstance(m2, tuple):
        return True
    return m1 == m2


def modes_compatible(a, b, slots=False):
    """Two accesses with modes a and b (sets of (class, location)) do not race:
    wherever their locations may overlap, both are atomics or both only read;
    both taking atomically allocated slots counts only with slots=True (an
    opt-in heuristic, outside the verified configuration). A location is None
    (anywhere), a buffer location (buf, lo, hi) from physical_modes, compared by
    byte overlap, or a member index of one shader's block (compared by equality:
    meaningful only within one module and one descriptor offset)."""
    if not a or not b:
        return False
    for c1, m1 in a:
        for c2, m2 in b:
            if not _overlap(m1, m2):
                continue
            if c1 != c2 or (c1 == "slot" and not slots):
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
