"""Meaning inference over real shaders (SPIR-V), linked to host facts.

Each value or memory cell gets one unknown per dimension (frame, unit, color;
matrices get from/to). Seeds come from interface annotations (Slang user
attributes, GLSL sidecar, C++ COV annotations) and from host facts (view and
attachment formats). Operation rules (spec section 4) add equalities, and a
union-find reports any conflict with the origin of both sides.

The analysis is flow-insensitive and context-sensitive (each call site
analyses its callee afresh; shaders have no recursion).

Fixed-function state as a source of meaning
-------------------------------------------
Vulkan state the shader never sees still fixes what its values must mean.
These rules read it from the recorded pipeline and draw (PipelineUse):

* Swapchain color space (PipelineUse.presented). A presented image is seeded
  with the encoding its VkColorSpaceKHR asks the display for:
  SRGB_NONLINEAR, EXTENDED_SRGB_NONLINEAR, DISPLAY_P3_NONLINEAR and
  DCI_P3_NONLINEAR -> 'srgb' (an *_SRGB format reaches that by encoding on
  write; a UNORM format needs the shader to encode), HDR10_ST2084 -> 'pq'
  (SMPTE ST 2084 over BT.2020 primaries), HDR10_HLG -> 'hlg',
  EXTENDED_SRGB_LINEAR / BT709_LINEAR / BT2020_LINEAR / DISPLAY_P3_LINEAR ->
  'linear'; PASS_THROUGH and unknown spaces seed nothing. The fragment output
  written there unites with that encoding, so e.g. sRGB code values written
  to an HDR10 swapchain, or PQ-encoding values that are still sRGB-encoded,
  are color mismatches.
* PQ transfer heuristic (like the sRGB pow heuristic, and flagged as
  inferred): pow(x, m1) with m1 = 0.1593017578125 takes linear light (the
  first step of the PQ inverse EOTF), pow(x, m2) with m2 = 78.84375 yields a
  PQ-encoded value; pow(x, 1/m2) takes a PQ value and pow(x, 1/m1) yields
  linear light (the EOTF). The other ST 2084 constants (c1 = 0.8359375,
  c2 = 18.8515625, c3 = 18.6875) only appear in the rational step between
  the two pows, which the ordinary arithmetic rules already join.
* Blend state vs alpha convention (dimension 'conv.alpha', values 'straight'
  and 'premultiplied'; conflicts are meaning.convention-mismatch). For each
  blended color attachment with an ADD color op: color factors
  (SRC_ALPHA, ONE_MINUS_SRC_ALPHA) require a straight-alpha fragment output,
  (ONE, ONE_MINUS_SRC_ALPHA) a premultiplied one; other factor pairs say
  nothing. When the pass cleared the attachment to a transparent color
  (PipelineUse.dst_alpha == 'transparent') and the alpha factors are an
  'over' pair, the attachment then holds premultiplied color (rgb * a), and
  that convention travels with the image's resource cell to every later pass
  that samples it. Unblended outputs pass their convention to the attachment
  unchanged. Alpha squared: alpha factors (SRC_ALPHA, ONE_MINUS_SRC_ALPHA)
  leave dst.a = a * a + (1 - a) * dst.a, which is reported when a later pass
  samples that attachment (its alpha is consumed). Convention seeds also come
  from [Convention("alpha=...")] declarations and from in-shader evidence:
  rgb multiplied by its own alpha (inferred) makes a premultiplied value from
  a straight one. Alpha travels through copies, sums, mixes, selects and
  component-wise products of two 4-vectors, but not through scaling by a
  scalar or division (that is how conversions are written).
* Clip-space Y convention. Each pre-rasterization stage that writes
  BuiltIn Position gets the list of Y flips applied to the clip position:
  y negated (-y, y * -1, a constant scale with a negative y component), a
  constant matrix whose [1][1] is negative, or a matrix declared
  [Convention("clip_y=down")] (a projection that already produces Vulkan's
  +y-down clip space). The draw's first viewport (dynamic or static) adds a
  flip when its height is negative (VK_KHR_maintenance1). Two or more flips
  from at least two different sources (shader code, projection, viewport)
  cancel or double up, and are reported as meaning.convention-mismatch;
  a single flip, or flips seen on one side only, report nothing.
"""
"""Meaning inference over real shaders (SPIR-V), linked to host facts.

Every value or memory cell has one node per meaning dimension; the solver
(meaning_solver.MeaningSolver, over labels.LabelledUF) keeps for each node a
group label relative to its class. Dataflow adds edges y = g(x):

* ordinary operations (copies, sums, component-wise products, selects of
  values of one kind, stores, loads, varyings, attachments) are identity edges;
* library and fixed-function semantics are *signatures* (covenant/signatures):
  an *_SRGB view decodes (texel = sRGB_dec(image)), an *_SRGB attachment
  encodes, a swapchain color space fixes the encoding the display expects,
  blend factors fix the alpha convention of the source and add one
  premultiplication (and one power of coverage to stored alpha), a negative
  viewport height flips clip Y and asserts +Y-up input, a depth texel is in
  [0, 1];
* nonlinear shader code gets its label from *probe fingerprinting*
  (probe.fingerprint): the stage is executed by a small SPIR-V interpreter on
  a ramp and every univariate slice is fitted against the transfer library
  (transfer.fit); an unambiguous monotone fit labels the edge (sRGB / PQ / HLG
  encode or decode, identity for affine maps and tone curves), anything else
  is opaque: no constraint, no alarm. Labels that come only from a fit (or
  another inference) are weak: their conflicts are warnings.

A conflict is two paths between the same nodes with different labels (or two
named frames meeting); domain constraints say that a value is never encoded
or premultiplied twice (Porter-Duff applies alpha once; a transfer function
is applied once). See meaning_solver for the diagnostics.
"""
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from . import probe, witness
from .diagnostics import Report
from .meaning_solver import HEURISTIC, MeaningSolver
from .signatures import SRGB_DEC, SRGB_ENC, glsl_std450 as g450, vulkan as vk

DIMS = ("frame", "unit", "color", "conv.alpha")
# dimensions a value that only supplies alpha (or a partial overwrite) does not carry
COLORISH = ("color", "frame", "conv.alpha")
MDIMS = ("from", "to")
# annotation dimensions the solver tracks on struct members and globals
TAG_DIMS = DIMS + MDIMS

COLOR_SPACE_ENCODING = vk.COLOR_SPACE_ENCODING
BF = vk.BF
OVER_STRAIGHT, OVER_PREMUL = vk.OVER_STRAIGHT, vk.OVER_PREMUL

ARITH_SAME = {"OpFAdd", "OpFSub", "OpFNegate", "OpFMod", "OpFRem", "OpCopyObject",
              # a numeric conversion keeps what a value means (float4(uint4 colour))
              "OpConvertUToF", "OpConvertSToF", "OpFConvert", "OpUConvert", "OpSConvert"}
# extracting a bit field of a packed word with a constant mask or shift keeps the
# word's meaning (the channels of a packed uchar4 colour are that colour)
FIELD_EXTRACT = {"OpBitwiseAnd", "OpShiftRightLogical", "OpShiftRightArithmetic",
                 "OpBitFieldUExtract", "OpBitFieldSExtract"}
GLSL_SAME = g450.COLOR_SAME
SAMPLE_OPS = {"OpImageSampleImplicitLod", "OpImageSampleExplicitLod",
              "OpImageSampleProjImplicitLod", "OpImageSampleProjExplicitLod",
              "OpImageFetch", "OpImageGather"}
# without a fingerprint, these never join their operands' color
NONLINEAR_OPS = {"OpSelect", "OpPhi"}


def _observable(node):
    """Values that are stored or exchanged (resources, varyings, globals,
    declarations), not intermediate SSA values of one computation."""
    root = node[0]
    return not (isinstance(root, tuple) and root and root[0] in ("ssa", "ver", "ret"))


def _valid_encoding(word):
    """A color label relative to linear light is linear or one encode."""
    return word == () or (len(word) == 1 and word[0][1] == 1)


@dataclass
class PipelineUse:
    """One pipeline as the host uses it.

    stages: [(Module, entry name)] in pipeline order
    bindings: {(set, binding): (resource id, view format or None)}
    attachments: {location: (resource id, format)}
    presented: resource ids that are shown on a display: a set (an sRGB
        display) or {resource id: (encoding, origin)} from the swapchain's
        color space (see COLOR_SPACE_ENCODING)
    blend: {location: {"enable", "src_color", "dst_color", "color_op",
        "src_alpha", "dst_alpha", "alpha_op"}} (VkBlendFactor/VkBlendOp names)
    dst_alpha: {location: "transparent" | "opaque"} what the attachment held
        before the draw, when known (cleared by the pass, or no alpha channel)
    viewport: (x, y, width, height) of the draw's first viewport, if known
    values: concrete bytes the draw's shaders read: {"push": bytes,
        (set, binding): bytes} (push constants and uniform buffers recorded by
        the model driver); they make the probe's parameters concrete
    """
    name: str
    stages: list
    bindings: dict = field(default_factory=dict)
    attachments: dict = field(default_factory=dict)
    presented: set = field(default_factory=set)
    blend: dict = field(default_factory=dict)
    dst_alpha: dict = field(default_factory=dict)
    viewport: tuple = None
    values: dict = field(default_factory=dict)
    color_spaces: dict = field(default_factory=dict)   # presented resource id -> VkColorSpaceKHR
    vertex_input: dict = None      # {"bindings": [...], "attributes": [...]} from the pipeline


@dataclass
class HostSeed:
    """A meaning fact from host code, attached to a shader struct member (by
    type) or to a host resource."""
    tag: dict
    origin: str
    struct: str = None
    member: str = None
    resource: str = None
    where: str = None


TAG_DIM = {"conv.alpha": "conv.alpha", "frame": "frame", "unit": "unit", "color": "color",
           "from": "from", "to": "to"}


def check(pipelines, host_seeds=(), report=None, host_values=None, host_flows=()):
    """host_values: {(struct, member): (number, host name)} for uniform fields the
    host always sets to one known number; they count as constants in shaders.
    host_flows: [(image, transfer.Fit, sink encoding, origin, where)] from
    witness.host_pixel_flows: the host's path from a readback image to an
    image file (an edge labelled by the fit, into a node the sink's signature
    fixes)."""
    report = report or Report()
    s = MeaningSolver(report)
    s.host_values = host_values or {}
    s.require_domain("color", _valid_encoding,
                     "a stored value is linear light or encoded once", _observable)
    s.require_domain("conv.alpha", {0, 1},
                     "Porter-Duff compositing multiplies color by alpha once (straight or "
                     "premultiplied)", _observable)
    seeded = {}
    for p in pipelines:
        for m, _ in p.stages:
            for (st, mem), tag in m.meta.get("fields", {}).items():
                seeded.setdefault((st, mem), []).append(
                    (tag, f"the declaration of {st}.{mem} in {Path(m.meta.get('source', m.path)).name}",
                     f"{Path(m.meta.get('source', m.path)).name}: {st}.{mem}"))
    for h in host_seeds:
        if h.struct:
            seeded.setdefault((h.struct, h.member), []).append((h.tag, h.origin, h.where or h.origin))
    s.depth_ranges = {}             # (struct, member) -> (clip depth range, origin)
    for (st, mem), facts in seeded.items():
        for tag, origin, where in facts:
            if "depth_range" in tag:
                s.depth_ranges[(st, mem)] = (tag["depth_range"], origin)
            elif tag.get("conv.depth") == "reverse_z" and (st, mem) not in s.depth_ranges:
                s.depth_ranges[(st, mem)] = ("zero_to_one", f"{origin}, declared reverse-Z "
                                             "(reverse-Z needs the [0,1] clip depth range)")
            for d, v in tag.items():
                if d in TAG_DIM:
                    s.set(("t", st, mem, d), TAG_DIM[d], v, origin, where)
    for h in host_seeds:
        if h.resource:
            for d, v in h.tag.items():
                if d in TAG_DIM:
                    s.set((("res", h.resource), (), d), TAG_DIM[d], v, h.origin, h.where or h.origin)

    # images some pass reads: their alpha channel is consumed later
    s.sampled = {b[0] for p in pipelines for b in p.bindings.values() if b and b[0]}
    for pi, p in enumerate(pipelines):
        stage_an = []
        for si, (m, entry) in enumerate(p.stages):
            a = _StageAnalysis(s, m, entry, p, (pi, si), seeded)
            a.run()
            stage_an.append(a)
        for up, down in zip(stage_an, stage_an[1:]):
            _link_varyings(s, up, down, report)
        _check_clip_y(s, p, stage_an, pi)
        for rid in p.presented:
            if isinstance(p.presented, dict):
                enc, origin = p.presented[rid]
            else:
                enc, origin = "srgb", f"'{rid}' being shown on an sRGB display"
            if enc:
                s.set((("res", rid), (), "color"), "color", enc, origin, p.name)
    for img, fit, sink, origin, where in host_flows:
        node = ("file", where, "color")
        s.edge(((("res", img), (), "color")), node, "color", fit.element, origin, where, weak=True)
        s.set(node, "color", sink, f"{where}, whose pixels must be {sink} (image-I/O signature: "
              "8-bit image files hold display-referred code values)" if sink == "srgb" else
              f"{where}, whose pixels must be {sink} (image-I/O signature)", where)
    s.finish()
    return report


def _check_clip_y(s, p, stage_an, pi):
    """Clip-space Y (dimension clip_y, Z2: 0 = +Y up, 1 = +Y down). The
    flips applied to the clip position form a chain of edges labelled 1; each
    flip idiom (y negated, a negative [1][1], a negative viewport height)
    exists to turn +Y-up coordinates into Vulkan's +Y down, so it asserts that
    its input is +Y up. A projection declared clip_y=down asserts its output
    is +Y down. Two flips in a row, or a flip of a +Y-down projection, close a
    cycle with a non-identity residual."""
    pos = [a for a in stage_an if a.clip_flips is not None]
    if not pos:
        return
    a = pos[-1]                      # the last pre-rasterization stage wins
    cur = ("clip_y", pi, 0)
    k = 0
    for kind, text, where in a.clip_flips:
        if kind == "projection":
            s.set(cur, "clip_y", "down", text, where)
            continue
        s.set(cur, "clip_y", "up", f"{text}, which assumes +Y-up input", where)
        k += 1
        nxt = ("clip_y", pi, k)
        s.edge(cur, nxt, "clip_y", 1, "", where)
        cur = nxt
    vp = p.viewport
    if vk.viewport_flips(vp):
        s.set(cur, "clip_y", "up",
              f"the viewport of '{p.name}' has a negative height (y = {vp[1]:g}, height = "
              f"{vp[3]:g}), which flips Y and so expects +Y-up clip coordinates", p.name)


def _link_varyings(s, up, down, report):
    """Vulkan pairs varyings by location (and component): the value written
    at a location is the value read there, whatever either side calls it.
    Names are not compared (engines rename systematically: o_uv -> in_uv)."""
    outs = {loc: v for v, loc in up.io("Output").items()}
    ins = {loc: v for v, loc in down.io("Input").items()}
    for loc, vin in ins.items():
        vout = outs.get(loc)
        if vout is None:
            continue
        a, b = up.var_cell(vout), down.var_cell(vin)
        for d in DIMS:
            s.union(a + (d,), b + (d,), d, f"{up.label} -> {down.label} location {loc}")


class _StageAnalysis:
    def __init__(self, solver, module, entry, pipe, sid, seeded):
        self.s, self.m, self.entry, self.pipe, self.sid = solver, module, entry, pipe, sid
        self.seeded = seeded
        self.cells = {}            # (ctx, id) -> (root, path)
        self.ins_cover = {}        # (ctx, id) -> (base id, replaced components)
        self.ins_pending = {}      # (ctx, id) -> last insert of a chain, not yet united
        module = _ssa(module)      # flow-sensitive locals: color = pow(color, ...) is two values
        self.m = module
        self.fp = None             # probe.Fingerprint of this entry point
        self._nocolor = False      # visiting a node whose color label comes from the probe
        self.clip_flips = None     # Y flips of the clip position, if this stage writes it
        self.fid = None
        self.iface = []
        self.exec_model = None
        for model, fid, name, iface in module.entries:
            if name == entry:
                self.fid, self.iface, self.exec_model = fid, iface, model
        if self.fid is None:
            raise KeyError(f"{module.path}: no entry point {entry}")
        self.slang = module.generator == 40
        self.label = f"{Path(module.meta.get('source', module.path)).name}:{entry}"

    # ------------------------------------------------------------ helpers
    def io(self, sc_name):
        out = {}
        for v in self.iface:
            if self._sc(v) == sc_name and "Location" in self.m.decos.get(v, {}):
                out[v] = self.m.decos[v]["Location"][0]
        return out

    def _sc(self, var):
        from .spirv import STORAGE
        return STORAGE.get(self.m.storage_class(var))

    def var_cell(self, var):
        m = self.m
        sc = self._sc(var)
        d = m.decos.get(var, {})
        if sc in ("Uniform", "StorageBuffer", "UniformConstant") and "Binding" in d:
            key = (d.get("DescriptorSet", [0])[0], d["Binding"][0])
            bound = self.pipe.bindings.get(key)
            if bound:
                return (("res", bound[0]), ())
            return (("glob", m.path, m.name(var)), ())
        if sc in ("Input", "Output"):
            return (("io", self.sid, var), ())
        if sc in ("PushConstant", "Private"):
            return (("glob", m.path, m.name(var)), ())
        return None

    def loc(self, ins):
        if ins is not None and ins.line:
            return f"{Path(ins.line[0]).name}:{ins.line[1]}"
        return f"{Path(self.m.meta.get('source', self.m.path)).name} ({self.entry})"

    def const(self, i):
        return i in self.m.consts

    def cell(self, ctx, i):
        c = self.cells.get((ctx, i))
        if c is None:
            c = self.cells.get(((), i))           # globals
        if c is None:
            c = (("ssa", ctx, i), ())
        return c

    def key(self, c, d):
        return (c[0], c[1], d)

    def type_of(self, i):
        d = self.m.defs.get(i)
        return d.type if d is not None else None

    def dims_for(self, tid):
        if tid is not None and self.m.is_matrix(self.m.pointee(tid) if self.m.tkind(tid) == "pointer" else tid):
            return MDIMS + DIMS
        return DIMS

    def unite(self, a, b, dims, where):
        for d in dims:
            if d == "color" and self._nocolor:
                continue
            self.s.union(self.key(a, d), self.key(b, d), d, where)

    def unite_deep(self, a, b, tid, where, depth=0):
        """Union two cells of type tid, member by member."""
        m = self.m
        if tid is None or depth > 4:
            return
        k = m.tkind(tid)
        if k == "pointer":
            return self.unite_deep(a, b, m.pointee(tid), where, depth)
        if k == "struct":
            for i, mt in enumerate(m.types[tid].args):
                self.unite_deep((a[0], a[1] + (i,)), (b[0], b[1] + (i,)), mt, where, depth + 1)
                self.link_member(a, tid, i)
                self.link_member(b, tid, i)
        elif k in ("array", "runtimearray"):
            self.unite_deep((a[0], a[1] + ("*",)), (b[0], b[1] + ("*",)),
                            m.types[tid].args[0], where, depth + 1)
        else:
            self.unite(a, b, self.dims_for(tid), where)

    def link_member(self, base_cell, struct_tid, i):
        sname = self.m.struct_name(struct_tid)
        mname = self.m.member_names.get((struct_tid, i), str(i))
        if (sname, mname) in self.seeded:
            c = (base_cell[0], base_cell[1] + (i,))
            mt = self.m.types[struct_tid].args[i]
            for d in self.dims_for(mt):
                self.s.union(self.key(c, d), ("t", sname, mname, d), d,
                             f"{sname}.{mname}")
            self._link_struct_tree(c, mt)

    def _link_struct_tree(self, c, tid, depth=0):
        m = self.m
        if depth > 4 or tid is None:
            return
        k = m.tkind(tid)
        if k == "struct":
            for i in range(len(m.types[tid].args)):
                self.link_member(c, tid, i)
        elif k in ("array", "runtimearray"):
            self._link_struct_tree((c[0], c[1] + ("*",)), m.types[tid].args[0], depth + 1)

    def step(self, base_cell, tid, idx_id, const_known):
        """Index one level into a composite of type tid."""
        m = self.m
        k = m.tkind(tid)
        if k == "struct":
            i = m.consts[idx_id] if const_known else idx_id
            self.link_member(base_cell, tid, i)
            return (base_cell[0], base_cell[1] + (i,)), m.types[tid].args[i]
        if k in ("array", "runtimearray"):
            return (base_cell[0], base_cell[1] + ("*",)), m.types[tid].args[0]
        # vector component / matrix column: same meaning cell
        return base_cell, m.member_type(tid, 0)

    # --------------------------------------------------------------- run
    def run(self):
        m = self.m
        # SPIR-V < 1.4 lists only Input / Output variables in the entry
        # point's interface: resources and push constants are module globals
        # (statically used ones only: those named by a function the entry
        # point reaches through its call graph)
        reach, todo, used = set(), [self.fid], set()
        while todo:
            f = todo.pop()
            if f in reach or f not in m.functions:
                continue
            reach.add(f)
            for i in m.functions[f]:
                used.update(x for x in i.args if isinstance(x, int))
                if i.op == "OpFunctionCall":
                    todo.append(i.args[0])
        extra = [d.id for d in m.defs.values()
                 if d.op == "OpVariable" and d.id not in self.iface and d.id in used
                 and self._sc(d.id) in ("Uniform", "StorageBuffer", "UniformConstant",
                                        "PushConstant", "Private")]
        for v in list(self.iface) + extra:
            c = self.var_cell(v)
            if c is None:
                continue
            self.cells[((), v)] = c
            self._link_struct_tree(c, m.pointee(m.defs[v].type))
            sf = m.meta.get("varyings", {}).get(m.name(v))
            if sf and tuple(sf) in self.seeded:
                for d in DIMS:
                    self.s.union(self.key(c, d), ("t", sf[0], sf[1], d), d,
                                 f"{sf[0]}.{sf[1]}")
            tag = m.meta.get("globals", {}).get(m.name(v))
            if tag:
                for d, val in tag.items():
                    self.s.set(self.key(c, d), d, val,
                               f"the declaration of '{m.name(v)}'", self.loc(None))
            if m.decos.get(v, {}).get("BuiltIn") == [0] and self._sc(v) == "Output":
                self.s.set(self.key(c, "frame"), "frame", "clip",
                           "the vertex position output, which must be in clip space",
                           self.loc(None))
        self.fp = _fingerprint(m, self.fid, self._memory())
        self.analyse(self.fid, (), [])
        if self.exec_model == 4:
            for where, msg, notes in witness.color_matrix(m, self.fp, self.loc):
                self.s.report.add("meaning", "meaning.convention-mismatch", msg, where, notes)

    def _memory(self):
        """Concrete block bytes for the probe: push constants and uniform
        buffers the driver recorded for this draw (PipelineUse.values)."""
        m, vals = self.m, self.pipe.values or {}
        blocks = {}
        for vid, d in m.defs.items():
            if d.op != "OpVariable":
                continue
            sc = d.args[0] if d.args else None
            data = None
            if sc == 9:
                data = vals.get("push")
            elif sc in (2, 12):
                dd = m.decos.get(vid, {})
                if "Binding" in dd:
                    data = vals.get((dd.get("DescriptorSet", [0])[0], dd["Binding"][0]))
            if data is not None:
                if isinstance(data, tuple):
                    blocks[vid] = data
                else:
                    blocks[vid] = (bytes(data), b"\x01" * len(data))
        return probe.Memory(m, blocks, getattr(self.s, "host_values", {}))

    def _fp_kind(self, key):
        """How the color dimension of SSA value `key` is decided:
        'edge' (a fitted transfer label), 'opaque' (no constraint),
        'structural' / None (ordinary rules)."""
        fp = self.fp
        if fp is None or fp.error:
            return None
        if key in fp.structural:
            return "structural"
        if key in fp.edges:
            return "edge"
        if key in fp.opaque:
            return "opaque"
        return None

    def analyse(self, fid, ctx, args):
        m = self.m
        for p, a in zip(m.params[fid], args):
            self.cells[(ctx, p)] = a
        ret = (("ret", ctx), ())
        body = self._reachable(fid, ctx)
        straight = not any(i.op in ("OpBranchConditional", "OpSwitch") for i in body)
        version = {}
        for ins in body:
            if straight and ins.op == "OpStore" and self._is_local(ins.args[0]):
                # straight-line code: each whole store to a local starts a new
                # value, so c = decode(c) does not merge c's old and new meaning
                v = ins.args[0]
                version[v] = version.get(v, 0) + 1
                self.cells[(ctx, v)] = (("ver", ctx, v, version[v]), ())
            self.visit(ins, ctx, ret)
        self._premultiplied(body, ctx)
        self._depth_range(body, ctx)
        self._clip_y(body)
        for (cx, _), (b, res, rgb, w) in list(self.ins_pending.items()):
            if cx == ctx:
                if not rgb:
                    self.unite(b, res, COLORISH, w)
                del self.ins_pending[(cx, _)]
        return ret

    def _premultiplied(self, body, ctx=()):
        """rgb multiplied by its own alpha is a Porter-Duff premultiply: an
        edge in conv.alpha (Z, the number of premultiplications) from the
        4-vector to the product. A texture declared premultiplied already
        counts one, so a second multiply leaves the domain {0, 1}. The edge
        is weak (recognised from the shape of the code)."""
        m = self.m
        val = {}                         # id or local -> ("vec" | "rgb" | "alpha", source id)

        def vec4(i):
            t = self.type_of(i)
            return t in m.types and m.tkind(t) == "vector" and m.types[t].args[1] == 4

        for ins in body:
            op, a, i = ins.op, ins.args, ins.id
            if op in SAMPLE_OPS:
                val[i] = ("vec", i)
            elif op == "OpLoad" and a[0] not in val and vec4(i) and not self._is_local(a[0]):
                val[i] = ("vec", i)
            elif op == "OpStore" and self._is_local(a[0]):
                if a[1] in val:
                    val[a[0]] = val[a[1]]
                else:
                    val.pop(a[0], None)
            elif op in ("OpLoad", "OpCopyObject") and a[0] in val:
                val[i] = val[a[0]]
            elif op == "OpVectorShuffle" and val.get(a[0], ("",))[0] == "vec" \
                    and all(c < 3 for c in a[2:]) and a[0] == a[1]:
                val[i] = ("rgb", val[a[0]][1])
            elif op == "OpCompositeExtract" and val.get(a[0], ("",))[0] == "vec" and a[1:] == [3]:
                val[i] = ("alpha", val[a[0]][1])
            elif op == "OpCompositeConstruct" and a and all(val.get(x, ("",))[0] == "alpha" for x in a):
                val[i] = val[a[0]]       # float3(fg.a)
            elif op in ("OpFMul", "OpVectorTimesScalar"):
                kinds = [val.get(x) for x in a[:2]]
                rgb = [k for k in kinds if k and k[0] == "rgb"]
                alpha = [k for k in kinds if k and k[0] == "alpha"]
                if rgb and alpha and rgb[0][1] == alpha[0][1]:
                    src = rgb[0][1]
                    self.s.edge(self.key(self.cell(ctx, src), "conv.alpha"),
                                self.key(self.cell(ctx, i), "conv.alpha"), "conv.alpha", 1,
                                "rgb multiplied by its own alpha (a Porter-Duff premultiply) at "
                                + self.loc(ins), self.loc(ins), weak=True)
                elif rgb or alpha:
                    val[i] = rgb[0] if rgb else alpha[0]   # scaling by opacity keeps the kind

    def _depth_range(self, body, ctx=()):
        """Clip depth range (dimension depth.range, Z: 0 = [0, 1], 1 = [-1, 1]).
        A sampled depth value is in [0, 1] (vulkan signature); exact affine
        slices found by the probe are edges (2x - 1: +1, 0.5x + 0.5: -1,
        x: identity); the z of a 4-vector is its own node; an inverse
        projection with a known clip depth range asserts the range of the z
        it is handed."""
        m = self.m
        ranges = getattr(self.s, "depth_ranges", {})
        if not ranges:
            return
        fp = self.fp
        D = "depth.range"
        k = lambda c: self.key(c, D)
        kz = lambda c: (c[0], c[1] + ("z",), D)
        if fp is not None and not fp.error:
            for v, (u, a, b) in fp.affine.items():
                if v[0] != ctx:
                    continue
                g = None
                if abs(a - 2) < 1e-4 and abs(b + 1) < 1e-4:
                    g, what = 1, "z * 2 - 1 maps [0, 1] to [-1, 1]"
                elif abs(a - 0.5) < 1e-4 and abs(b - 0.5) < 1e-4:
                    g, what = -1, "z * 0.5 + 0.5 maps [-1, 1] to [0, 1]"
                elif abs(a - 1) < 1e-6 and abs(b) < 1e-6:
                    g, what = 0, ""
                if g is not None:
                    ins = m.defs.get(v[1])
                    self.s.edge(k(self.cell(*u)), k(self.cell(*v)), D, g,
                                f"{what} at {self.loc(ins)}" if what else "", self.loc(ins))

        def depth_tex(i, depth=0):
            d = m.defs.get(i)
            if d is None or depth > 8:
                return False
            if d.op == "OpVariable":
                rid, fmt = self._view_format(self.cell((), i))
                return vk.is_depth_format(fmt)
            if d.op in ("OpSampledImage", "OpLoad", "OpImage", "OpAccessChain"):
                return depth_tex(d.args[0], depth + 1)
            return False

        def width(i):
            t = self.type_of(i)
            return m.types[t].args[1] if t in m.types and m.tkind(t) == "vector" else 1

        pretty = {"zero_to_one": "[0,1]", "neg_one_to_one": "[-1,1]"}
        for ins in body:
            op, a, i = ins.op, ins.args, ins.id
            if op in SAMPLE_OPS and depth_tex(a[0]):
                self.s.edge(self.s.uf(D).anchor("[0,1]"), k(self.cell(ctx, i)), D, vk.DEPTH_TEXEL_RANGE,
                            f"a depth value sampled at {self.loc(ins)} (Vulkan depth is in [0, 1])",
                            self.loc(ins))
            elif op == "OpCompositeExtract" and len(a) == 2 and a[1] == 0:
                self.s.union(k(self.cell(ctx, a[0])), k(self.cell(ctx, i)), D, self.loc(ins))
            elif op == "OpCompositeConstruct" and width(i) == 4:
                pos = 0
                for c in a:
                    if pos == 2 and width(c) == 1 and c not in m.consts:
                        self.s.union(k(self.cell(ctx, c)), kz(self.cell(ctx, i)), D, self.loc(ins))
                    pos += width(c)
            elif op in ("OpVectorTimesMatrix", "OpMatrixTimesVector"):
                v, mt = (a[0], a[1]) if op == "OpVectorTimesMatrix" else (a[1], a[0])
                key = self._member_key(self._load_ptr(mt))
                if key in ranges:
                    rng, origin = ranges[key]
                    self.s.set(kz(self.cell(ctx, v)), D, rng,
                               f"'{key[0]}.{key[1]}', which inverts a projection whose clip depth "
                               f"is {pretty.get(rng, rng)} (the range of {key[1]} comes from {origin})",
                               self.loc(ins))

    def _load_ptr(self, i):
        d = self.m.defs.get(i)
        return d.args[0] if d is not None and d.op == "OpLoad" else i

    def _dead_local(self, ptr):
        """ptr is (a member of) a Function variable that no load, copy or
        call reads."""
        m = self.m
        root = ptr
        while root in m.defs and m.defs[root].op in ("OpAccessChain", "OpInBoundsAccessChain"):
            root = m.defs[root].args[0]
        if not self._is_local(root):
            return False
        live = getattr(self, "_live_locals", None)
        if live is None:
            live = set()

            def root_of(x):
                while x in m.defs and m.defs[x].op in ("OpAccessChain", "OpInBoundsAccessChain",
                                                       "OpPtrAccessChain", "OpCopyObject"):
                    x = m.defs[x].args[0]
                return x
            for body in m.functions.values():
                for d in body:
                    if d.op in ("OpLoad", "OpCopyMemory", "OpCopyMemorySized") \
                            or d.op.startswith("OpAtomic"):
                        srcs = d.args[1:2] if d.op.startswith("OpCopyMemory") else d.args[:1]
                    elif d.op in ("OpFunctionCall", "OpExtInst"):
                        srcs = d.args
                    else:
                        continue
                    live.update(root_of(x) for x in srcs)
            self._live_locals = live
        return root not in live

    def _is_local(self, ptr):
        d = self.m.defs.get(ptr)
        return d is not None and d.op == "OpVariable" and d.args and d.args[0] == 7   # Function

    _CMP = {"OpIEqual": lambda a, b: a == b, "OpINotEqual": lambda a, b: a != b,
            "OpFOrdEqual": lambda a, b: a == b, "OpFOrdNotEqual": lambda a, b: a != b,
            "OpULessThan": lambda a, b: a < b, "OpSLessThan": lambda a, b: a < b,
            "OpUGreaterThan": lambda a, b: a > b, "OpSGreaterThan": lambda a, b: a > b,
            "OpFOrdLessThan": lambda a, b: a < b, "OpFOrdGreaterThan": lambda a, b: a > b,
            "OpLogicalEqual": lambda a, b: a == b, "OpLogicalNotEqual": lambda a, b: a != b,
            "OpLogicalAnd": lambda a, b: bool(a and b), "OpLogicalOr": lambda a, b: bool(a or b)}

    def _eval(self, i, depth=0):
        """Value of a scalar SSA id from constants and host-known uniform fields."""
        m = self.m
        if i in m.consts and isinstance(m.consts[i], (int, float, bool)):
            return m.consts[i]
        d = m.defs.get(i)
        if d is None or depth > 8:
            return None
        if d.op == "OpConstantTrue":
            return True
        if d.op == "OpConstantFalse":
            return False
        if d.op == "OpLoad":
            got = self._host_value(d.args[0])
            return got[0] if got else None
        if d.op in ("OpUConvert", "OpSConvert", "OpConvertUToF", "OpConvertSToF", "OpBitcast",
                    "OpCopyObject"):
            return self._eval(d.args[0], depth + 1)
        if d.op == "OpLogicalNot":
            v = self._eval(d.args[0], depth + 1)
            return None if v is None else (not v)
        if d.op in self._CMP:
            a, b = (self._eval(x, depth + 1) for x in d.args[:2])
            return None if a is None or b is None else self._CMP[d.op](a, b)
        return None

    def _param_dependent(self, cid, depth=0, seen=None):
        """The condition's SPIR-V slice reaches a function parameter (its value
        may differ between call sites)."""
        seen = set() if seen is None else seen
        if cid in seen or depth > 32:
            return False
        seen.add(cid)
        d = self.m.defs.get(cid)
        if d is None:
            return False
        if d.op == "OpFunctionParameter":
            return True
        return any(self._param_dependent(x, depth + 1, seen) for x in d.args
                   if isinstance(x, int) and x in self.m.defs and x not in self.m.consts)

    def _concrete_condition(self, cid, ctx=None):
        """The value of a branch condition in the probe's base run, when it
        depends only on constants and on uniform / push-constant bytes the model
        driver recorded for this draw (no varying, texel, or unknown uniform):
        the draw then takes one side of the branch on every invocation. The
        calling context counts: a condition on a parameter (load_color(c,
        linear)) is decided per call site, from the run that executed that call
        (the base run, or a run with a varying branch held to its other side)."""
        fp = self.fp
        if fp is None or fp.error or fp.base is None:
            return None
        run, keys = fp.base, []
        if ctx is not None:
            for r in [fp.base] + list(getattr(fp, "flip_bases", ())):
                keys = [k for k in r.values if k[1] == cid and k[0] == ctx]
                if keys:
                    run = r
                    break
            if not keys and self._param_dependent(cid):
                return None                      # this call site never ran
        if not keys:
            keys = [k for k in run.values if k[1] == cid]
        if not keys:
            return None
        import numpy as np
        for k in keys:                           # held to the other side in a
            for r in [fp.base] + list(getattr(fp, "flip_bases", ())):   # path run
                if r is not run and k in r.values and not np.array_equal(
                        np.asarray(r.values[k]), np.asarray(run.values[k])):
                    return None
        got = None
        for k in keys:
            seen, todo = set(), [k]
            while todo:
                x = todo.pop()
                if x in seen:
                    continue
                seen.add(x)
                if x in run.sources:
                    return None                  # a varying, texel or unknown uniform
                todo.extend(run.deps.get(x, ()))
            v = run.values[k]
            try:
                import numpy as np
                a = np.asarray(v).reshape(-1)
            except Exception:
                return None
            if a.dtype != bool or not (a == a[0]).all():
                return None
            if got is not None and bool(a[0]) != got:
                return None
            got = bool(a[0])
        return got

    def _reachable(self, fid, ctx=None):
        """The function's instructions in blocks reachable from its entry, where a
        conditional branch whose condition the host fixes takes one side only
        (e.g. if (material.flag == 0) with flag always 1 on the host)."""
        body = self.m.functions[fid]
        blocks, order, cur = {}, [], None
        head = []
        for ins in body:
            if ins.op == "OpLabel":
                cur = ins.id
                order.append(cur)
                blocks[cur] = []
            if cur is None:
                head.append(ins)
            else:
                blocks[cur].append(ins)
        if not order:
            return body
        pruned = False
        seen, todo = set(), [order[0]]
        while todo:
            b = todo.pop()
            if b in seen or b not in blocks:
                continue
            seen.add(b)
            term = blocks[b][-1] if blocks[b] else None
            if term is None:
                continue
            if term.op == "OpBranch":
                todo.append(term.args[0])
            elif term.op == "OpBranchConditional":
                v = self._eval(term.args[0])
                if v is None:
                    v = self._concrete_condition(term.args[0], ctx)
                if v is None:
                    todo += term.args[1:3]
                else:
                    pruned = True
                    tgt = term.args[1] if v else term.args[2]
                    todo.append(tgt)
                    # the host fixed this branch, so it is an unconditional jump
                    blocks[b][-1] = type(term)("OpBranch", term.type, term.id, [tgt], term.line)
            elif term.op == "OpSwitch":
                todo += [term.args[1]] + term.args[3::2]
        if not pruned:
            return body
        out = list(head)
        for b in order:
            if b in seen:
                out += [i for i in blocks[b]
                        if not (i.op == "OpPhi")] + []
        # phis: keep only incoming values along edges that remain (from a
        # reachable predecessor whose possibly host-fixed branch still goes here)
        succ = {}
        for b in seen:
            term = blocks[b][-1] if blocks[b] else None
            if term is None:
                continue
            if term.op == "OpBranch":
                succ[b] = {term.args[0]}
            elif term.op == "OpBranchConditional":
                succ[b] = set(term.args[1:3])
            elif term.op == "OpSwitch":
                succ[b] = {term.args[1]} | set(term.args[3::2])
        for b in order:
            if b not in seen:
                continue
            for i in blocks[b]:
                if i.op == "OpPhi":
                    pairs = [(i.args[k], i.args[k + 1]) for k in range(0, len(i.args), 2)]
                    keep = [x for pair in pairs if b in succ.get(pair[1], ()) for x in pair]
                    out.append(type(i)(i.op, i.type, i.id, keep, i.line))
        return out

    def visit(self, ins, ctx, ret):
        """Ordinary rules, except that the color of a value inside a probed
        slice comes from its fingerprint (an edge from its fitted dominator,
        or nothing when opaque)."""
        key = (ctx, ins.id)
        kind = self._fp_kind(key) if ins.id is not None else None
        nonlin = kind in ("edge", "opaque") or (kind is None and self._nonlinear(ins))
        self._nocolor = nonlin
        try:
            self._visit_ops(ins, ctx, ret)
        finally:
            self._nocolor = False
        if kind in ("edge", "opaque"):
            self._outside_operands(ins, ctx)
        if kind == "edge":
            u, fit = self.fp.edges[key]
            R = (("ssa", ctx, ins.id), ())
            if ins.op == "OpFunctionCall":
                R = self.cells.get(key, R)
            w = self.loc(ins)
            origin = f"{fit.describe()} (probe fit) at {w}" if fit.element else ""
            self.s.edge(self.key(self.cell(*u), "color"), self.key(R, "color"), "color",
                        fit.element, origin, w, weak=bool(fit.element))

    def _outside_operands(self, ins, ctx):
        """A probed value still has the color of operands from outside its
        slice (a uniform tint or fog color mixed in): those do not depend on
        any per-invocation source, and modulating or mixing with them keeps
        one encoding."""
        m, fp, op = self.m, self.fp, ins.op
        if op in ARITH_SAME or op in ("OpFMul", "OpFDiv", "OpVectorTimesScalar"):
            ops = ins.args
        elif op == "OpExtInst" and ins.args[0] == m.glsl_ext:
            from .spirv import _grammar
            if _grammar()[1].get(ins.args[1], "") not in GLSL_SAME:
                return
            ops = ins.args[2:]
        else:
            return
        R = (("ssa", ctx, ins.id), ())
        for x in ops:
            if not isinstance(x, int) or x in m.consts:
                continue
            k = (ctx, x) if (ctx, x) in fp.base.values else ((), x)
            if k in fp.head_of or k not in fp.base.values:
                continue
            t = self.type_of(x)
            if t in m.types and m.tkind(t) == "vector" and m.types[t].args[1] >= 3:
                self.s.union(self.key(self.cell(ctx, x), "color"), self.key(R, "color"), "color",
                             self.loc(ins))

    def _nonlinear(self, ins):
        """Without a fingerprint, transfer-capable operations join nothing."""
        if ins.op in NONLINEAR_OPS:
            return True
        if ins.op == "OpExtInst" and ins.args and ins.args[0] == self.m.glsl_ext:
            from .spirv import _grammar
            name = _grammar()[1].get(ins.args[1], "")
            return name in g450.NONLINEAR or name == "FMix"
        return False

    def _visit_ops(self, ins, ctx, ret):
        m, op, a = self.m, ins.op, ins.args
        w = self.loc(ins)
        r = (ctx, ins.id)
        C = lambda i: self.cell(ctx, i)

        if op == "OpVariable":
            return
        if op in ("OpAccessChain", "OpInBoundsAccessChain"):
            base = C(a[0])
            tid = m.pointee(self.type_of(a[0]) or m.defs[a[0]].type)
            cur = base
            for idx in a[1:]:
                cur, tid = self.step(cur, tid, idx, idx in m.consts)
            self.cells[r] = cur
        elif op == "OpLoad":
            self.cells[r] = C(a[0])
        elif op == "OpStore":
            ptr_t = self.type_of(a[0])
            if a[1] in m.consts:
                return
            if self._dead_local(a[0]):
                # a local nothing reads (its loads became SSA values in
                # _ssa): the stores carry no meaning of their own, and uniting
                # them would merge every value the variable held in turn
                return
            self.unite_deep(C(a[0]), C(a[1]), m.pointee(ptr_t) if ptr_t else None, w)
            self._output_rules(a[0], a[1], ctx, w)
        elif op == "OpCompositeExtract":
            cur, tid = C(a[0]), self.type_of(a[0])
            for idx in a[1:]:
                k = m.tkind(tid)
                if k == "struct":
                    self.link_member(cur, tid, idx)
                    cur, tid = (cur[0], cur[1] + (idx,)), m.types[tid].args[idx]
                elif k in ("array", "runtimearray"):
                    cur, tid = (cur[0], cur[1] + ("*",)), m.types[tid].args[0]
                else:
                    tid = m.member_type(tid, 0)
            self.cells[r] = cur
        elif op == "OpCompositeInsert" and m.tkind(ins.type) == "vector" and len(a) == 3:
            # chains of single-component inserts (c.rgb = f(c.rgb) lowers to three)
            # are tracked against their first base: once r, g and b are replaced,
            # the base supplies only alpha, which has no colour encoding or frame
            comp, obj, idx = a          # Object (the new part), Composite, index
            base, cover = self.ins_cover.get((ctx, obj), (obj, frozenset()))
            cover = cover | {idx}
            self.ins_cover[(ctx, ins.id)] = (base, cover)
            res = (("ssa", ctx, ins.id), ())
            if comp not in m.consts:
                self.unite(C(comp), res, DIMS, w)
            plain = tuple(d for d in DIMS if d not in COLORISH)
            if base not in m.consts:
                self.unite(C(base), res, plain, w)
                # colour and frame are decided at the end of the chain (end of analyse)
                self.ins_pending.pop((ctx, obj), None)
                self.ins_pending[(ctx, ins.id)] = (C(base), res, m.types[ins.type].args[1] == 4
                                                   and {0, 1, 2} <= cover, w)
            if obj != base and obj not in m.consts:
                self.unite(C(obj), res, plain, w)
        elif op == "OpCompositeInsert":
            obj, comp = a[0], a[1]
            self.cells[r] = C(comp)
            tid = ins.type
            cur = C(comp)
            for idx in a[2:]:
                cur, tid = self.step(cur, tid, None, False) if m.tkind(tid) != "struct" else \
                    ((cur[0], cur[1] + (idx,)), m.types[tid].args[idx])
            if obj not in m.consts:
                self.unite_deep(cur, C(obj), tid, w)
        elif op == "OpCompositeConstruct":
            tid = ins.type
            if m.tkind(tid) == "struct":
                for i, c in enumerate(a):
                    if c not in m.consts:
                        self.unite_deep((("ssa", ctx, ins.id), (i,)), C(c),
                                        m.types[tid].args[i], w)
            elif m.tkind(tid) in ("vector",):
                for c in a:
                    ct = self.type_of(c)
                    if c not in m.consts and m.tkind(ct) == "vector":
                        # a 2-vector is not a point in a 3D frame: float3(uv, 0)
                        # and float4(uv * 2 - 1, z, 1) share no frame
                        dims = DIMS if m.types[ct].args[1] >= 3 else \
                            tuple(d for d in DIMS if d != "frame")
                        self.unite(C(c), (("ssa", ctx, ins.id), ()), dims, w)
                # scalars extracted from one vector and put back together
                # (glslang lowers vec4(c.rgb, 1) to three extracts): three or
                # more components of one source keep that source's meaning
                srcs = {}
                for c in a:
                    d = m.defs.get(c)
                    if d is not None and d.op == "OpCompositeExtract" and len(d.args) == 2 \
                            and m.tkind(self.type_of(c)) != "vector":
                        srcs.setdefault(d.args[0], set()).add(d.args[1])
                for src, idx in srcs.items():
                    if len(idx) >= 3 and src not in m.consts:
                        self.unite(C(src), (("ssa", ctx, ins.id), ()), DIMS, w)
                # a packed word unpacked by hand (float((c >> 8) & 0xff), ...,
                # the manual form of unpackUnorm4x8 / unpackUint4x8, GLSL.std.450
                # 8.4): three or more components that are bit fields of one
                # scalar keep that scalar's meaning
                words = {}
                for c in a:
                    root = self._packed_root(c)
                    if root is not None:
                        words.setdefault(root, set()).add(c)
                for root, cs in words.items():
                    if len(cs) >= 3:
                        self.unite(C(root), (("ssa", ctx, ins.id), ()), DIMS, w)
        elif op == "OpVectorShuffle":
            n1 = m.types[self.type_of(a[0])].args[1] if self.type_of(a[0]) in m.types else 4
            comps = a[2:]
            for k, c in enumerate(a[:2]):
                if c in m.consts:
                    continue
                pos = [j for j, x in enumerate(comps) if x != 0xFFFFFFFF and (x < n1) == (k == 0)]
                if not pos:
                    continue
                # a source that only supplies the alpha of a 4-vector carries no
                # colour encoding or frame: c.rgb = decode(c.rgb) keeps old c's alpha
                dims = DIMS if not (len(comps) == 4 and pos == [3]) else \
                    tuple(d for d in DIMS if d not in COLORISH)
                self.unite(C(c), (("ssa", ctx, ins.id), ()), dims, w)
        elif op in ARITH_SAME:
            for c in a:
                if c not in m.consts:
                    self.unite(C(c), (("ssa", ctx, ins.id), ()), self.dims_for(ins.type), w)
        elif op in FIELD_EXTRACT:
            nonconst = [c for c in a if c not in m.consts]
            if len(nonconst) == 1:
                self.unite(C(nonconst[0]), (("ssa", ctx, ins.id), ()), DIMS, w)
        elif op in ("OpFMul", "OpFDiv"):
            vec = m.tkind(ins.type) == "vector"
            nonconst = [c for c in a if c not in m.consts]
            if vec:
                # a component-wise product of two 4-vectors (texel * tint) keeps the
                # alpha convention; scaling by a scalar or a splat does not
                dims = ("frame", "color")
                if op == "OpFMul" and len(nonconst) == 2 and m.types[ins.type].args[1] == 4 \
                        and not any(self._splat(c) for c in nonconst):
                    dims += ("conv.alpha",)
                if op == "OpFMul":     # sign(v) * f(|v|): the sign carries no meaning
                    nonconst = [c for c in nonconst if not self._is_sign(c)]
                for c in nonconst:
                    self.unite(C(c), (("ssa", ctx, ins.id), ()), dims, w)
        elif op == "OpVectorTimesScalar":
            if a[0] not in m.consts:
                self.unite(C(a[0]), (("ssa", ctx, ins.id), ()), ("frame", "color"), w)
        elif op == "OpDot":
            if a[0] not in m.consts and a[1] not in m.consts:
                self.unite(C(a[0]), C(a[1]), ("frame", "color"), w)
        elif op in ("OpMatrixTimesVector", "OpVectorTimesMatrix"):
            M, v = (a[0], a[1]) if op == "OpMatrixTimesVector" else (a[1], a[0])
            self._transform(C(M), C(v), (("ssa", ctx, ins.id), ()), w)
        elif op == "OpMatrixTimesMatrix":
            A, B = (a[1], a[0]) if self.slang else (a[0], a[1])   # Slang emits mul(A,B) as B*A
            A, B, R = C(A), C(B), (("ssa", ctx, ins.id), ())
            self.s.union(self.key(A, "from"), self.key(B, "to"), "from", w)
            self.s.union(self.key(R, "from"), self.key(B, "from"), "from", w)
            self.s.union(self.key(R, "to"), self.key(A, "to"), "to", w)
        elif op == "OpTranspose":
            self.unite(C(a[0]), (("ssa", ctx, ins.id), ()), MDIMS, w)
        elif op in ("OpSelect",):
            arms = [c for c in a[1:] if c not in m.consts]
            for c in arms:
                self.unite_deep(C(c), (("ssa", ctx, ins.id), ()), ins.type, w)
        elif op == "OpPhi":
            for c in a[0::2]:
                if c not in m.consts:
                    self.unite_deep(C(c), (("ssa", ctx, ins.id), ()), ins.type, w)
        elif op == "OpExtInst" and a[0] == m.glsl_ext:
            self._glsl(ins, ctx, w)
        elif op in SAMPLE_OPS:
            self._sample(ins, ctx, w)
        elif op == "OpImageWrite":
            img = self._image_root(a[0], ctx)
            if img is not None and a[2] not in m.consts:
                self.unite(img, C(a[2]), DIMS, w)
        elif op == "OpFunctionCall":
            callee = a[0]
            res = self.analyse(callee, ctx + (ins.id,), [C(x) for x in a[1:]])
            self.cells[r] = res
        elif op == "OpReturnValue":
            if a[0] not in m.consts:
                self.unite_deep(ret, C(a[0]), self.type_of(a[0]), w)

    def _is_sign(self, c, depth=4):
        """c is sign(x) (GLSL.std.450 FSign / SSign: -1, 0 or 1 whatever x
        means), possibly through numeric conversions."""
        m = self.m
        from .spirv import _grammar
        for _ in range(depth):
            d = m.defs.get(c)
            if d is None:
                return False
            if d.op == "OpExtInst" and d.args and d.args[0] == m.glsl_ext:
                return _grammar()[1].get(d.args[1], "") in ("FSign", "SSign")
            if d.op in ("OpConvertSToF", "OpConvertFToS", "OpFConvert", "OpSConvert", "OpCopyObject") and d.args:
                c = d.args[0]
                continue
            return False
        return False

    def _packed_root(self, c, depth=6):
        """The integer scalar c is a bit field of (through conversions,
        shifts and masks by constants), or None."""
        m = self.m
        seen_field = False
        for _ in range(depth):
            d = m.defs.get(c)
            if d is None or c in m.consts:
                return None
            if d.op in FIELD_EXTRACT or d.op in ("OpConvertUToF", "OpConvertSToF", "OpUConvert",
                                                  "OpSConvert", "OpBitcast"):
                var = [x for x in d.args if x not in m.consts]
                if len(var) != 1:
                    return None
                seen_field |= d.op in FIELD_EXTRACT
                c = var[0]
                continue
            break
        if not seen_field or m.tkind(self.type_of(c)) != "int":
            return None
        return c

    def _transform(self, M, v, R, w):
        self.s.union(self.key(v, "frame"), self.key(M, "from"), "from", w)
        self.s.union(self.key(R, "frame"), self.key(M, "to"), "to", w)
        self.s.union(self.key(R, "unit"), self.key(v, "unit"), "unit", w)

    def _glsl(self, ins, ctx, w):
        from .spirv import _grammar
        m = self.m
        name = _grammar()[1].get(ins.args[1], "")
        ops = ins.args[2:]
        C = lambda i: self.cell(ctx, i)
        R = (("ssa", ctx, ins.id), ())
        nc = [o for o in ops if o not in m.consts]
        if name in GLSL_SAME:
            for o in nc:
                self.unite(C(o), R, DIMS, w)
        elif name in ("Normalize", "Cross", "Reflect", "Refract", "FaceForward"):
            for o in nc[:2]:
                if m.tkind(self.type_of(o)) == "vector":
                    self.unite(C(o), R, ("frame", "color"), w)
        elif name in ("Length",):
            self.unite(C(ops[0]), R, ("unit",), w)
        elif name == "Distance":
            if len(nc) == 2:
                self.unite(C(nc[0]), C(nc[1]), ("frame", "unit"), w)
            self.unite(C(ops[0]), R, ("unit",), w)
        # Pow, Exp, Log, ... (g450.NONLINEAR): their color label comes from the
        # probe fingerprint only (visit), never from constants in the code
        elif name == "MatrixInverse":
            self.s.union(self.key(R, "from"), self.key(C(ops[0]), "to"), "from", w)
            self.s.union(self.key(R, "to"), self.key(C(ops[0]), "from"), "to", w)

    def _host_value(self, ptr):
        """A uniform field load whose value the host always sets to one number."""
        m, hv = self.m, getattr(self.s, "host_values", {})
        d = m.defs.get(ptr)
        if not hv or d is None or d.op not in ("OpAccessChain", "OpInBoundsAccessChain"):
            return None
        base, idx = d.args[0], d.args[1:]
        bd = m.defs.get(base)
        if bd is None or bd.op != "OpVariable":
            return None
        tid = m.pointee(bd.type)
        for k in idx:
            if m.tkind(tid) != "struct" or not isinstance(m.consts.get(k), int):
                return None
            mem = m.member_names.get((tid, m.consts[k]))
            key = (m.struct_name(tid), mem)
            if key in hv:
                return hv[key]
            tid = m.types[tid].args[m.consts[k]]
        return None

    def _image_root(self, i, ctx, depth=0):
        """Follow loads / OpSampledImage / access chains to the image variable's cell."""
        m = self.m
        d = m.defs.get(i)
        if d is None or depth > 8:
            return None
        if d.op == "OpSampledImage":
            return self._image_root(d.args[0], ctx, depth + 1)
        if d.op in ("OpLoad", "OpImage", "OpCopyObject"):
            return self._image_root(d.args[0], ctx, depth + 1) or self.cell(ctx, i)
        if d.op in ("OpVariable", "OpAccessChain", "OpInBoundsAccessChain",
                    "OpFunctionParameter"):
            return self.cell(ctx, i)
        return self.cell(ctx, i)

    def _view_format(self, img_cell):
        root = img_cell[0]
        if root[0] == "res":
            for (rid, fmt) in self.pipe.bindings.values():
                if rid == root[1]:
                    return rid, fmt
        return None, None

    def _sample(self, ins, ctx, w):
        img = self._image_root(ins.args[0], ctx)
        if img is None:
            return
        R = (("ssa", ctx, ins.id), ())
        rid, fmt = self._view_format(img)
        if vk.format_transfer(fmt) == "srgb":
            # Vulkan: an *_SRGB format holds sRGB-encoded texels and sampling
            # decodes them (texel = sRGB_dec(image))
            self.s.set(self.key(img, "color"), "color", "srgb",
                       f"'{rid}' being sampled through an {fmt} view (the format says its "
                       "texels are sRGB-encoded)", w)
            self.s.edge(self.key(img, "color"), self.key(R, "color"), "color", SRGB_DEC,
                        f"the hardware sRGB decode of '{rid}' ({fmt})", w)
            self.unite(img, R, ("frame", "unit", "conv.alpha"), w)
        else:
            self.unite(img, R, DIMS, w)
        if rid is not None and _has_alpha(fmt):
            # a sampled alpha channel is read as coverage
            self.s.set(self.key(img, "alpha.power"), "alpha.power", "coverage",
                       f"'{rid}' being sampled by '{self.pipe.name}', which reads its alpha as "
                       "coverage", w)

    def _output_rules(self, ptr, val, ctx, w):
        """Fragment outputs feed attachments; an sRGB attachment encodes on write."""
        m = self.m
        if self.exec_model not in (None, 4):       # only Fragment outputs are attachments
            return
        root = self.cell(ctx, ptr)[0]
        if root[0] != "io":
            return
        var = root[2]
        if self._sc(var) != "Output":
            return
        locd = m.decos.get(var, {}).get("Location")
        if locd is None:
            return
        att = self.pipe.attachments.get(locd[0])
        if not att:
            return
        rid, fmt = att
        v = self.cell(ctx, val)
        res = (("res", rid), ())
        bl = self.pipe.blend.get(locd[0]) or {}
        blended = bool(bl.get("enable"))
        if blended:
            self._blend_rules(v, rid, fmt, bl, locd[0], w)
        if vk.format_transfer(fmt) == "srgb":
            self.s.set(self.key(res, "color"), "color", "srgb",
                       f"'{rid}' having format {fmt} (its texels are sRGB-encoded)", w)
            self.s.edge(self.key(v, "color"), self.key(res, "color"), "color", SRGB_ENC,
                        f"the sRGB encode on write to '{rid}' ({fmt})", w)
            if not blended:
                self.unite(v, res, ("conv.alpha",), w)
        else:
            self.unite(v, res, tuple(d for d in DIMS if d != "conv.alpha") if blended else DIMS, w)

    def _blend_rules(self, v, rid, fmt, bl, loc, w):
        """Blend state as signatures (signatures.vulkan): the color factors
        fix the source convention and apply alpha once more; over a
        transparent destination the attachment then holds the source's
        convention plus that premultiplication. The alpha factors add powers
        of coverage to the stored alpha (SRC_ALPHA on alpha stores a * a)."""
        short = lambda f: str(f).replace(BF, "")
        cb = vk.color_blend(bl)
        has_alpha = _has_alpha(fmt)
        desc = f"the color blend ({short(bl.get('src_color'))}, {short(bl.get('dst_color'))}) " \
               f"of '{self.pipe.name}' into '{rid}'"
        ab = vk.alpha_blend(bl)
        if ab is not None and has_alpha:
            adesc = f"the alpha blend ({short(bl.get('src_alpha'))}, {short(bl.get('dst_alpha'))}) " \
                    f"of '{self.pipe.name}' into '{rid}'"
            self.s.set(self.key(v, "alpha.power"), "alpha.power", "coverage",
                       f"the fragment output of '{self.pipe.name}', whose alpha is coverage", w)
            self.s.edge(self.key(v, "alpha.power"), self.key((("res", rid), ()), "alpha.power"),
                        "alpha.power", ab,
                        f"{adesc}, which stores a * a + (1 - a) * dst.a (alpha squared)" if ab
                        else f"{adesc}, which stores coverage", w)
        if cb is None:
            return
        need, applied = cb
        name = "premultiplied" if need else "straight"
        self.s.set(self.key(v, "conv.alpha"), "conv.alpha", name,
                   f"{desc}, which expects {name}-alpha source color", w)
        if self.pipe.dst_alpha.get(loc) == "transparent" and has_alpha and ab is not None:
            self.s.edge(self.key(v, "conv.alpha"), self.key((("res", rid), ()), "conv.alpha"),
                        "conv.alpha", applied,
                        f"{desc}: over a transparent clear it stores "
                        f"{'rgb * a (premultiplied color)' if applied else 'the color as given'}", w)

    def _splat(self, i):
        d = self.m.defs.get(i)
        return d is not None and d.op == "OpCompositeConstruct" and len(set(d.args)) == 1 \
            and len(d.args) > 1

    # ------------------------------------------------------- clip-space Y
    def _position_targets(self):
        """(variable ids that are BuiltIn Position, {block var: member index})"""
        m = self.m
        whole, block = set(), {}
        for v in self.iface:
            if self._sc(v) != "Output":
                continue
            if m.decos.get(v, {}).get("BuiltIn") == [0]:
                whole.add(v)
                continue
            tid = m.pointee(m.defs[v].type)
            if m.tkind(tid) == "struct":
                for i in range(len(m.types[tid].args)):
                    if m.member_decos.get((tid, i), {}).get("BuiltIn") == [0]:
                        block[v] = i
        return whole, block

    def _clip_y(self, body):
        """Collect the Y flips applied to the value stored to BuiltIn Position:
        each is (source kind, explanation, location)."""
        m = self.m
        whole, block = self._position_targets()
        if not whole and not block:
            return
        cv = lambda i: m.consts.get(i) if isinstance(m.consts.get(i), (int, float)) else None
        fl = {}                  # id or local -> [flip]
        ptr = {}                 # access-chain id -> "pos" | "y" (into the position)
        state = {"pos": None}

        def width(i):
            t = self.type_of(i)
            return m.types[t].args[1] if t in m.types and m.tkind(t) == "vector" else 1

        def const_vec(i):
            d = m.defs.get(i)
            if d is not None and d.op in ("OpConstantComposite", "OpCompositeConstruct"):
                vals = [cv(x) for x in d.args]
                return vals if all(x is not None for x in vals) else None
            return None

        def mat_flip(i, depth=0):
            d = m.defs.get(i)
            if d is None or depth > 8:
                return []
            if d.op in ("OpLoad", "OpCopyObject", "OpTranspose"):
                if d.op == "OpLoad":
                    key = self._member_key(d.args[0])
                    for tag, origin, _ in self.seeded.get(key, ()) if key else ():
                        if tag.get("conv.clip_y") == "down":
                            return [("projection", f"'{key[0]}.{key[1]}' is declared "
                                     f"clip_y=down ({origin}): it already produces Vulkan's "
                                     "+y-down clip space", self.loc(d))]
                    if d.args[0] in fl:
                        return list(fl[d.args[0]])
                return mat_flip(d.args[0], depth + 1)
            if d.op == "OpMatrixTimesMatrix":
                return mat_flip(d.args[0], depth + 1) + mat_flip(d.args[1], depth + 1)
            if d.op == "OpConstantComposite":
                cols = [const_vec(c) for c in d.args]
                if len(cols) >= 2 and all(cols) and len(cols[1]) >= 2 and cols[1][1] < 0 \
                        and cols[0][0] > 0:
                    return [("shader", f"a constant matrix with a negative [1][1] "
                             f"({cols[1][1]:g}) flips Y", self.loc(d))]
            return []

        def neg(i):
            """Flips carried by a scalar placed in the y slot."""
            d = m.defs.get(i)
            if d is None:
                return fl.get(i, [])
            if d.op == "OpFNegate":
                return fl.get(d.args[0], []) + [("shader", "the vertex shader negates the clip "
                                                 "position's y", self.loc(d))]
            if d.op == "OpFMul" and width(i) == 1:
                k = [cv(x) for x in d.args[:2]]
                other = [x for x, c in zip(d.args[:2], k) if c is None]
                if any(c is not None and c < 0 for c in k) and len(other) == 1:
                    return fl.get(other[0], []) + [("shader", "the vertex shader multiplies the "
                                                    "clip position's y by a negative constant",
                                                    self.loc(d))]
            return fl.get(i, [])

        for ins in body:
            op, a, i = ins.op, ins.args, ins.id
            if op in ("OpAccessChain", "OpInBoundsAccessChain"):
                base, idx = a[0], [cv(x) for x in a[1:]]
                if base in whole:
                    ptr[i] = "pos" if not idx else ("y" if idx == [1] else None)
                elif base in block and idx and idx[0] == block[base]:
                    ptr[i] = "pos" if len(idx) == 1 else ("y" if idx[1:] == [1] else None)
            elif op == "OpLoad":
                if a[0] in whole or ptr.get(a[0]) == "pos":
                    if state["pos"] is not None:
                        fl[i] = list(state["pos"])
                elif ptr.get(a[0]) == "y":
                    if state["pos"] is not None:
                        fl[i] = list(state["pos"])
                elif a[0] in fl:
                    fl[i] = list(fl[a[0]])
            elif op == "OpStore":
                tgt, v = a[0], a[1]
                if tgt in whole or ptr.get(tgt) == "pos":
                    state["pos"] = list(fl.get(v, []))
                elif ptr.get(tgt) == "y":
                    state["pos"] = list(neg(v))
                elif self._is_local(tgt):
                    if v in fl:
                        fl[tgt] = list(fl[v])
                    else:
                        fl.pop(tgt, None)
            elif op == "OpCopyObject" and a[0] in fl:
                fl[i] = list(fl[a[0]])
            elif op == "OpCompositeExtract" and len(a) == 2 and a[1] == 1 and a[0] in fl:
                fl[i] = list(fl[a[0]])
            elif op == "OpCompositeInsert" and len(a) == 3:
                obj, comp, idx = a
                if idx == 1:
                    got = neg(obj)
                    if got:
                        fl[i] = got
                elif comp in fl:
                    fl[i] = list(fl[comp])
            elif op == "OpCompositeConstruct" and m.tkind(ins.type) == "vector":
                pos = 0
                for c in a:
                    w_ = width(c)
                    if pos <= 1 < pos + w_:
                        got = neg(c) if w_ == 1 else (fl.get(c, []) if pos == 0 else [])
                        if got:
                            fl[i] = list(got)
                    pos += w_
            elif op == "OpVectorShuffle":
                n1 = width(a[0])
                comps = a[2:]
                if len(comps) >= 2:
                    src, k = (a[0], comps[1]) if comps[1] < n1 else (a[1], comps[1] - n1)
                    if k == 1 and src in fl:
                        fl[i] = list(fl[src])
            elif op in ("OpFAdd", "OpFSub"):
                nc = [x for x in a[:2] if x not in m.consts and const_vec(x) is None]
                if len(nc) == 1 and nc[0] in fl:
                    fl[i] = list(fl[nc[0]])
            elif op == "OpFMul" and width(i) >= 2:
                kv = [const_vec(x) for x in a[:2]]
                other = [x for x, k in zip(a[:2], kv) if k is None]
                k = next((k for k in kv if k), None)
                if k and len(other) == 1:
                    got = list(fl.get(other[0], []))
                    if len(k) >= 2 and k[1] < 0 and k[0] > 0:
                        got.append(("shader", f"the vertex shader scales the clip position by "
                                    f"{tuple(k)}, flipping y", self.loc(ins)))
                    if got:
                        fl[i] = got
            elif op == "OpVectorTimesScalar" and a[0] in fl:
                fl[i] = list(fl[a[0]])
            elif op in ("OpMatrixTimesVector", "OpVectorTimesMatrix"):
                M, v = (a[0], a[1]) if op == "OpMatrixTimesVector" else (a[1], a[0])
                got = fl.get(v, []) + mat_flip(M)
                if got:
                    fl[i] = got
            elif op == "OpMatrixTimesMatrix":
                got = mat_flip(i)
                if got:
                    fl[i] = got
        if state["pos"] is not None:
            self.clip_flips = state["pos"]
        elif self.clip_flips is None and (whole or block):
            self.clip_flips = None

    def _member_key(self, ptr):
        """(struct, member) of a pointer that indexes a struct variable by constants."""
        m = self.m
        d = m.defs.get(ptr)
        if d is None or d.op not in ("OpAccessChain", "OpInBoundsAccessChain"):
            return None
        bd = m.defs.get(d.args[0])
        if bd is None or bd.op != "OpVariable":
            return None
        tid = m.pointee(bd.type)
        key = None
        for k in d.args[1:]:
            if m.tkind(tid) in ("array", "runtimearray"):
                tid = m.types[tid].args[0]
                continue
            if m.tkind(tid) != "struct" or not isinstance(m.consts.get(k), int):
                return key if m.tkind(tid) != "struct" else None
            key = (m.struct_name(tid), m.member_names.get((tid, m.consts[k])))
            tid = m.types[tid].args[m.consts[k]]
        return key


def _has_alpha(fmt):
    """Whether a VkFormat stores alpha (unknown formats count as having it)."""
    if not fmt:
        return True
    comps = str(fmt).replace("VK_FORMAT_", "").split("_")[0]
    return "A" in comps


_FP_CACHE = {}


def _fingerprint(module, fid, memory):
    """probe.fingerprint, cached per (module, entry, concrete bytes)."""
    try:
        ident = hashlib.sha1(Path(module.path).read_bytes()).hexdigest()
    except OSError:
        ident = id(module)
    key = (ident, id(module), fid,
           tuple(sorted((k, hashlib.sha1(bytes(v[0])).hexdigest()) for k, v in memory.blocks.items())),
           tuple(sorted(memory.host_values.items())) if memory.host_values else ())
    try:
        hash(key)
    except TypeError:
        key = None
    if key is not None and key in _FP_CACHE:
        return _FP_CACHE[key]
    try:
        fp = probe.fingerprint(module, fid, memory)
    except Exception as e:                      # the probe is best effort
        fp = probe.Fingerprint(error=f"{type(e).__name__}: {e}")
    try:
        _cover_varying_branches(module, fid, memory, fp)
    except Exception:
        pass
    if key is not None:
        _FP_CACHE[key] = fp
    return fp


MAX_FLIPS = 3


def _varying_branches(module, fp):
    """[(condition key, value)] of conditional branches the base run executed
    whose condition depends on a per-invocation source (a varying, vertex input
    or texel) and whose other side it never entered: every invocation of the
    probe took one side, a real draw takes both."""
    import numpy as np
    run = fp.base
    out = []
    for body in module.functions.values():
        blocks, cur = {}, None
        for ins in body:
            if ins.op == "OpLabel":
                cur = ins.id
                blocks[cur] = []
            elif cur is not None:
                blocks[cur].append(ins)

        def unseen_work(b0):
            """Float-valued instructions in the blocks reachable from b0 that the
            base run never executed (the slice a flip would add)."""
            seen_b, todo, cnt = set(), [b0], 0
            while todo:
                b = todo.pop()
                if b in seen_b or b in run.blocks or b not in blocks:
                    continue
                seen_b.add(b)
                for i in blocks[b]:
                    if i.type is not None and module.tkind(i.type) in ("float", "vector"):
                        cnt += 1
                    if i.op == "OpBranch":
                        todo.append(i.args[0])
                    elif i.op == "OpBranchConditional":
                        todo += i.args[1:3]
                    elif i.op == "OpSwitch":
                        todo += [a for j, a in enumerate(i.args[1:]) if j % 2 == 0]
            return cnt
        # control dependence of phis: a short circuit (a || b) merges a
        # constant flag with b in a phi, and the probe records no data
        # dependence on a; the phi depends on the conditions of the branches
        # that select its incoming edge.
        guard, phi_block = {}, {}
        for b, inss in blocks.items():
            for i in inss:
                if i.op == "OpBranchConditional":
                    for t in i.args[1:3]:
                        guard.setdefault(t, set()).add(i.args[0])
                        guard.setdefault(b, set())
                elif i.op == "OpPhi":
                    phi_block[i.id] = b

        def ctrl(x):
            ins_x = module.defs.get(x[1])
            if ins_x is None or ins_x.op != "OpPhi":
                return ()
            preds = [ins_x.args[j] for j in range(1, len(ins_x.args), 2)]
            cs = set()
            for pb in preds + [phi_block.get(x[1])]:
                cs |= guard.get(pb, set())
            return [(x[0], c) for c in cs]
        for ins in body:
            if ins.op != "OpBranchConditional":
                continue
            cid = ins.args[0]
            cd = module.defs.get(cid)
            if cd is not None and (cd.op.startswith("OpFOrd") or cd.op.startswith("OpFUnord")) \
                    and any(x in module.consts for x in cd.args):
                # x <= c selects a piece of a piecewise function of x (the sRGB
                # toe); the probe's ramp over x already runs both pieces, and
                # fitting one piece alone would split the curve
                continue
            for k in [k for k in run.values if k[1] == cid]:
                v = np.asarray(run.values[k]).reshape(-1)
                if v.dtype != bool or not (v == v[0]).all():
                    continue
                other = ins.args[2] if bool(v[0]) else ins.args[1]
                if other in run.blocks:
                    continue
                seen, todo, varying = set(), [k], False
                while todo and not varying:
                    x = todo.pop()
                    if x in seen:
                        continue
                    seen.add(x)
                    if run.sources.get(x) in ("input", "texel"):
                        varying = True
                    todo.extend(run.deps.get(x, ()))
                    todo.extend(ctrl(x))
                if varying:
                    work = unseen_work(other)
                    if work:
                        out.append((work, k, ~v.astype(bool)))
    out.sort(key=lambda t: -t[0])
    return [(k, v) for _, k, v in out]


def _cover_varying_branches(module, fid, memory, fp):
    """Path coverage for the fingerprint: for up to MAX_FLIPS varying branches
    the base run never took, fingerprint again with the condition held to the
    other side, and add the slices only those runs executed (their keys are
    new, so nothing already fitted changes). A merge of the two sides is then
    an ordinary phi: both arms carry labels, and a variable whose arms disagree
    closes a cycle (one value, two encodings)."""
    if fp is None or fp.error or fp.base is None:
        return
    fp.flip_bases = []
    for key, val in _varying_branches(module, fp)[:MAX_FLIPS]:
        fp2 = probe.fingerprint(module, fid, memory, force={key: val})
        if fp2.error or fp2.base is None:
            continue
        known = set(fp.base.values)
        for b in getattr(fp, "flip_bases", ()):
            known |= set(b.values)
        fresh = set(fp2.base.values) - known
        if not fresh:
            continue
        fp.flip_bases.append(fp2.base)
        for k in fresh:
            if k in fp2.edges:
                fp.edges[k] = fp2.edges[k]
            if k in fp2.opaque:
                fp.opaque.add(k)
            if k in fp2.structural:
                fp.structural.add(k)
            if k in fp2.affine:
                fp.affine[k] = fp2.affine[k]
            if k in fp2.head_of:
                fp.head_of[k] = fp2.head_of[k]
            if k in fp2.heads:
                fp.heads.add(k)


_SSA_CACHE = {}


def _ssa(module):
    """The module after spirv-opt --ssa-rewrite (mem2reg) when it keeps values
    in Function variables: each store becomes its own SSA value and merges
    become phis, so an in-place update (c = encode(c)) does not give c two
    meanings at once. Falls back to the module itself."""
    import shutil
    import subprocess
    import tempfile
    from . import spirv
    if not any(d.op == "OpVariable" and d.args and d.args[0] == 7 for d in module.defs.values()):
        return module
    key = (module.path, id(module))
    if key in _SSA_CACHE:
        return _SSA_CACHE[key]
    out = module
    tool = shutil.which("spirv-opt")
    if tool:
        try:
            d = Path(tempfile.mkdtemp(prefix="cov-ssa-"))
            dst = d / Path(module.path).name
            r = subprocess.run([tool, "--ssa-rewrite", str(module.path), "-o", str(dst)],
                               capture_output=True, text=True, timeout=60)
            if r.returncode == 0:
                out = spirv.parse(dst)
                out.meta = module.meta
                out.generator = module.generator
        except (OSError, subprocess.SubprocessError, ValueError, KeyError):
            out = module
    _SSA_CACHE[key] = out
    return out
