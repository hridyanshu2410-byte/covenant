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
from dataclasses import dataclass, field
from pathlib import Path

from ..diagnostics import Report
from .meaning import _UF, HEURISTIC

DIMS = ("frame", "unit", "color", "conv.alpha")
# dimensions a value that only supplies alpha (or a partial overwrite) does not carry
COLORISH = ("color", "frame", "conv.alpha")
MDIMS = ("from", "to")
MSG_DIM = {"from": "frame", "to": "frame"}

SRGB_POW = (2.2, 2.4)
INV_SRGB_POW = (1 / 2.2, 1 / 2.4)
# SMPTE ST 2084 (PQ) constants
PQ_M1, PQ_M2 = 0.1593017578125, 78.84375
PQ_C1, PQ_C2, PQ_C3 = 0.8359375, 18.8515625, 18.6875

# swapchain color space -> encoding the display expects in the image
COLOR_SPACE_ENCODING = {
    "VK_COLOR_SPACE_SRGB_NONLINEAR_KHR": "srgb",
    "VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT": "srgb",
    "VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT": "srgb",
    "VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT": "srgb",
    "VK_COLOR_SPACE_HDR10_ST2084_EXT": "pq",
    "VK_COLOR_SPACE_HDR10_HLG_EXT": "hlg",
    "VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT": "linear",
    "VK_COLOR_SPACE_BT709_LINEAR_EXT": "linear",
    "VK_COLOR_SPACE_BT2020_LINEAR_EXT": "linear",
    "VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT": "linear",
}

BF = "VK_BLEND_FACTOR_"
OVER_STRAIGHT = (BF + "SRC_ALPHA", BF + "ONE_MINUS_SRC_ALPHA")
OVER_PREMUL = (BF + "ONE", BF + "ONE_MINUS_SRC_ALPHA")

ARITH_SAME = {"OpFAdd", "OpFSub", "OpFNegate", "OpFMod", "OpFRem", "OpCopyObject"}
GLSL_SAME = {"FAbs", "Floor", "Ceil", "Fract", "Round", "RoundEven", "Trunc", "FSign",
             "FMin", "FMax", "NMin", "NMax", "FClamp", "NClamp", "FMix"}
SAMPLE_OPS = {"OpImageSampleImplicitLod", "OpImageSampleExplicitLod",
              "OpImageSampleProjImplicitLod", "OpImageSampleProjExplicitLod",
              "OpImageFetch", "OpImageGather"}


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
    """
    name: str
    stages: list
    bindings: dict = field(default_factory=dict)
    attachments: dict = field(default_factory=dict)
    presented: set = field(default_factory=set)
    blend: dict = field(default_factory=dict)
    dst_alpha: dict = field(default_factory=dict)
    viewport: tuple = None


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


class _Solver:
    def __init__(self, report):
        self.uf = _UF(report)
        self.report = report

    def union(self, a, b, dim, where):
        self.uf.union(a, b, MSG_DIM.get(dim, dim), where)

    def set(self, k, dim, value, origin, where):
        self.uf.set(k, MSG_DIM.get(dim, dim), value, origin, where)


def check(pipelines, host_seeds=(), report=None, host_values=None):
    """host_values: {(struct, member): (number, host name)} for uniform fields the
    host always sets to one known number; they count as constants in shaders."""
    report = report or Report()
    s = _Solver(report)
    s.host_values = host_values or {}
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
            for d, v, w in ((d, v, w) for d, v in tag.items() for w in [where]):
                s.set(("t", st, mem, d), d, v, origin, w)
    for h in host_seeds:
        if h.resource:
            for d, v in h.tag.items():
                s.set((("res", h.resource), (), d), d, v, h.origin, h.where or h.origin)

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
        _check_clip_y(s, p, stage_an)
        for rid in p.presented:
            if isinstance(p.presented, dict):
                enc, origin = p.presented[rid]
            else:
                enc, origin = "srgb", f"'{rid}' being shown on an sRGB display"
            if enc:
                s.set((("res", rid), (), "color"), "color", enc, origin, p.name)
    return report


def _check_clip_y(s, p, stage_an):
    """Y flips of the clip position (shader code, projection) and of the
    viewport compose; see the module docstring."""
    pos = [a for a in stage_an if a.clip_flips is not None]
    if not pos:
        return
    a = pos[-1]                      # the last pre-rasterization stage wins
    flips = list(a.clip_flips)
    vp = p.viewport
    if vp is not None and len(vp) >= 4 and vp[3] < 0:
        flips.append(("viewport", f"the viewport of '{p.name}' has a negative height "
                      f"(y = {vp[1]:g}, height = {vp[3]:g}), which flips Y", p.name))
    if len(flips) < 2 or len({f[0] for f in flips}) < 2:
        return
    s.report.add("meaning", "meaning.convention-mismatch",
                 f"clip-space Y is flipped {len(flips)} times for '{p.name}': the flips "
                 "cancel (or flip a projection that already targets Vulkan's +y-down clip "
                 "space), so the image is upside down.", flips[0][2],
                 [f[1] for f in flips] + ["keep exactly one Y flip: in the projection, in the "
                                          "vertex shader, or in the viewport"])


def _norm_name(n):
    n = n.split(".")[-1].lower()
    for pre in ("out_", "in_", "out", "in", "v_", "vs_", "fs_", "frag_", "vert_"):
        if n.startswith(pre) and len(n) > len(pre):
            n = n[len(pre):]
            break
    return n


def _link_varyings(s, up, down, report):
    outs = {loc: v for v, loc in up.io("Output").items()}
    ins = {loc: v for v, loc in down.io("Input").items()}
    for loc, vin in ins.items():
        vout = outs.get(loc)
        if vout is None:
            continue
        no, ni = up.m.name(vout), down.m.name(vin)
        if _norm_name(no) != _norm_name(ni):
            report.add("structure", "structure.varying-mismatch",
                       f"location {loc} is written as '{no.split('.')[-1]}' by "
                       f"{up.label} but read as '{ni.split('.')[-1]}' by {down.label}; "
                       "Vulkan pairs varyings by location, so the second stage reads "
                       "the wrong value.", f"{down.label} input '{ni}'",
                       ["give both the same location, or reorder the declarations"])
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
        self.transfer_ids = set()  # ids holding the result of an sRGB transfer curve
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
        from ..spirv import STORAGE
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
        for v in self.iface:
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
        self.analyse(self.fid, (), [])

    def analyse(self, fid, ctx, args):
        m = self.m
        for p, a in zip(m.params[fid], args):
            self.cells[(ctx, p)] = a
        ret = (("ret", ctx), ())
        body = self._reachable(fid)
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
        self._depth_unproject(body)
        self._clip_y(body)
        for (cx, _), (b, res, rgb, w) in list(self.ins_pending.items()):
            if cx == ctx:
                if not rgb:
                    self.unite(b, res, COLORISH, w)
                del self.ins_pending[(cx, _)]
        return ret

    def _premultiplied(self, body, ctx=()):
        """A texture declared alpha=premultiplied already has rgb * a; multiplying
        its rgb by its own alpha again darkens every partly covered pixel.
        For any other 4-vector, rgb * its own alpha is evidence (inferred) that
        the vector holds straight alpha and the product premultiplied color."""
        m = self.m
        val = {}                         # id or local -> ("vec" | "rgb" | "alpha", texture or source id)

        def vec4(i):
            t = self.type_of(i)
            return t in m.types and m.tkind(t) == "vector" and m.types[t].args[1] == 4

        def tex_of(i, depth=0):
            d = m.defs.get(i)
            if d is None or depth > 8:
                return None
            if d.op == "OpVariable":
                conv = m.meta.get("globals", {}).get(m.name(i), {})
                return m.name(i) if conv.get("conv.alpha") == "premultiplied" else None
            if d.op in ("OpSampledImage", "OpLoad", "OpImage", "OpCopyObject",
                        "OpAccessChain", "OpInBoundsAccessChain"):
                return tex_of(d.args[0], depth + 1)
            return None

        for ins in body:
            op, a, i = ins.op, ins.args, ins.id
            if op in SAMPLE_OPS:
                t = tex_of(a[0])
                val[i] = ("vec", t) if t else ("vec", ("src", i))
            elif op == "OpLoad" and a[0] not in val and vec4(i) and not self._is_local(a[0]):
                val[i] = ("vec", ("src", i))
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
                if rgb and alpha and rgb[0][1] == alpha[0][1] and isinstance(rgb[0][1], tuple):
                    src = rgb[0][1][1]
                    why = HEURISTIC + "rgb multiplied by its own alpha (premultiplying)"
                    self.s.set(self.key(self.cell(ctx, src), "conv.alpha"), "conv.alpha", "straight",
                               why + ", so the source holds straight alpha", self.loc(ins))
                    self.s.set(self.key(self.cell(ctx, i), "conv.alpha"), "conv.alpha",
                               "premultiplied", why, self.loc(ins))
                elif rgb and alpha and rgb[0][1] == alpha[0][1]:
                    t = rgb[0][1]
                    self.s.report.add(
                        "meaning", "meaning.convention-mismatch",
                        f"'{t}' is declared premultiplied (its rgb already holds rgb * a), but "
                        f"its rgb is multiplied by its own alpha again.", self.loc(ins),
                        [f"'{t}' has [Convention(\"alpha=premultiplied\")]",
                         "drop the extra * a, or store straight (non-premultiplied) alpha"])
                elif rgb or alpha:
                    val[i] = rgb[0] if rgb else alpha[0]   # scaling by opacity keeps the kind

    def _depth_unproject(self, body):
        """Rebuilding a position from a depth texture: the clip z handed to an
        inverse projection must use that projection's depth range. Vulkan depth
        is already [0,1]; depth * 2 - 1 is the OpenGL [-1,1] mapping."""
        m = self.m
        ranges = getattr(self.s, "depth_ranges", {})
        if not ranges:
            return
        val = {}          # id or local -> "zero_to_one" | "neg_one_to_one" | "x2" | ("clip", range, ins)

        def depth_tex(i, depth=0):
            d = m.defs.get(i)
            if d is None or depth > 8:
                return False
            if d.op == "OpVariable":
                rid, fmt = self._view_format(self.cell((), i))
                return bool(fmt) and fmt.upper().startswith(("VK_FORMAT_D16", "VK_FORMAT_D24",
                                                             "VK_FORMAT_D32", "D16", "D24", "D32"))
            if d.op in ("OpSampledImage", "OpLoad", "OpImage", "OpAccessChain"):
                return depth_tex(d.args[0], depth + 1)
            return False

        def member(i):
            d = m.defs.get(i)
            if d is not None and d.op == "OpLoad":
                d = m.defs.get(d.args[0])
            if d is None or d.op not in ("OpAccessChain", "OpInBoundsAccessChain"):
                return None
            bd = m.defs.get(d.args[0])
            if bd is None or bd.op != "OpVariable":
                return None
            tid = m.pointee(bd.type)
            key = None
            for k in d.args[1:]:
                if m.tkind(tid) != "struct" or not isinstance(m.consts.get(k), int):
                    return None
                key = (m.struct_name(tid), m.member_names.get((tid, m.consts[k])))
                tid = m.types[tid].args[m.consts[k]]
            return key

        def width(i):
            t = self.type_of(i)
            return m.types[t].args[1] if t in m.types and m.tkind(t) == "vector" else 1

        cv = lambda i: m.consts.get(i) if isinstance(m.consts.get(i), (int, float)) else None
        for ins in body:
            op, a, i = ins.op, ins.args, ins.id
            if op in SAMPLE_OPS and depth_tex(a[0]):
                val[i] = "vec"
            elif op == "OpCompositeExtract" and val.get(a[0]) == "vec" and a[1:] == [0]:
                val[i] = "zero_to_one"
            elif op == "OpStore" and self._is_local(a[0]):
                if a[1] in val:
                    val[a[0]] = val[a[1]]
                else:
                    val.pop(a[0], None)
            elif op in ("OpLoad", "OpCopyObject") and a[0] in val:
                val[i] = val[a[0]]
            elif op == "OpFMul" and len(a) == 2 and "zero_to_one" in (val.get(a[0]), val.get(a[1])) \
                    and 2.0 in (cv(a[0]), cv(a[1])):
                val[i] = "x2"
            elif op == "OpFSub" and val.get(a[0]) == "x2" and cv(a[1]) == 1.0:
                val[i] = "neg_one_to_one"
            elif op == "OpCompositeConstruct" and width(i) == 4:
                pos = 0
                for c in a:
                    if pos == 2 and val.get(c) in ("zero_to_one", "neg_one_to_one"):
                        val[i] = ("clip", val[c], ins)
                    pos += width(c)
            elif op in ("OpVectorTimesMatrix", "OpMatrixTimesVector"):
                v, mt = (a[0], a[1]) if op == "OpVectorTimesMatrix" else (a[1], a[0])
                got, key = val.get(v), member(mt)
                if isinstance(got, tuple) and key in ranges and got[1] != ranges[key][0]:
                    pretty = {"zero_to_one": "[0,1]", "neg_one_to_one": "[-1,1]"}
                    self.s.report.add(
                        "meaning", "meaning.convention-mismatch",
                        f"clip depth range mismatch: the clip z built from the depth texture is in "
                        f"{pretty[got[1]]}, but '{key[0]}.{key[1]}' inverts a projection whose clip "
                        f"depth is {pretty[ranges[key][0]]}.", self.loc(got[2]),
                        [f"the range of {key[1]} comes from {ranges[key][1]}",
                         "Vulkan depth is already [0,1]; use the sampled depth as clip z"
                         if got[1] == "neg_one_to_one" else "map depth with depth * 2 - 1"])

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

    def _reachable(self, fid):
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
        # phis: keep only incoming values from reachable predecessors
        for b in order:
            if b not in seen:
                continue
            for i in blocks[b]:
                if i.op == "OpPhi":
                    pairs = [(i.args[k], i.args[k + 1]) for k in range(0, len(i.args), 2)]
                    keep = [x for pair in pairs if pair[1] in seen for x in pair]
                    out.append(type(i)(i.op, i.type, i.id, keep, i.line))
        return out

    def visit(self, ins, ctx, ret):
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
            xfer = [c for c in arms if c in self.transfer_ids]
            if xfer:
                # the piecewise sRGB curve: select(c <= 0.04045, c / 12.92, pow(...)).
                # The result has the transfer function's encoding, not the linear arm's.
                arms = xfer
                self.transfer_ids.add(ins.id)
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

    def _transform(self, M, v, R, w):
        self.s.union(self.key(v, "frame"), self.key(M, "from"), "from", w)
        self.s.union(self.key(R, "frame"), self.key(M, "to"), "to", w)
        self.s.union(self.key(R, "unit"), self.key(v, "unit"), "unit", w)

    def _glsl(self, ins, ctx, w):
        from ..spirv import _grammar
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
        elif name == "Pow":
            self.host_const_used = None
            e = self._const_value(ops[1])
            pw = f"pow(x, {e:.3g})" if e is not None else ""
            if e is not None and self.host_const_used:
                pw += f" (exponent computed from host field {self.host_const_used}, always set to a known value)"
            if e is not None and any(abs(e - p) < 0.05 for p in SRGB_POW + INV_SRGB_POW):
                self.transfer_ids.add(ins.id)
            pq = None
            if e is not None:
                for k, (x, role) in enumerate(((PQ_M1, "m1"), (PQ_M2, "m2"),
                                               (1 / PQ_M2, "1/m2"), (1 / PQ_M1, "1/m1"))):
                    if abs(e - x) <= 1e-3 * x:
                        pq = (k, role)
            if pq is not None:
                k, role = pq
                enc = k < 2              # m1, m2: PQ inverse EOTF (encode); 1/m2, 1/m1: EOTF
                what = (f"{pw} (ST 2084 exponent {role}), which looks like a PQ "
                        f"{'encode' if enc else 'decode'}")
                if k == 0:
                    self.s.set(self.key(C(ops[0]), "color"), "color", "linear", HEURISTIC + what, w)
                elif k == 1:
                    self.s.set(self.key(R, "color"), "color", "pq", HEURISTIC + "the output of " + what, w)
                elif k == 2:
                    self.s.set(self.key(C(ops[0]), "color"), "color", "pq", HEURISTIC + what, w)
                else:
                    self.s.set(self.key(R, "color"), "color", "linear", HEURISTIC + "the output of " + what, w)
            elif e is not None:
                if any(abs(e - p) < 0.05 for p in SRGB_POW):
                    self.s.set(self.key(C(ops[0]), "color"), "color", "srgb",
                               HEURISTIC + f"{pw}, which looks like an sRGB decode", w)
                    self.s.set(self.key(R, "color"), "color", "linear",
                               HEURISTIC + f"{pw}, which looks like an sRGB decode", w)
                elif any(abs(e - p) < 0.01 for p in INV_SRGB_POW):
                    self.s.set(self.key(C(ops[0]), "color"), "color", "linear",
                               HEURISTIC + f"{pw}, which looks like an sRGB encode", w)
                    self.s.set(self.key(R, "color"), "color", "srgb",
                               HEURISTIC + f"the output of {pw}, which looks like an sRGB encode", w)
        elif name == "MatrixInverse":
            self.s.union(self.key(R, "from"), self.key(C(ops[0]), "to"), "from", w)
            self.s.union(self.key(R, "to"), self.key(C(ops[0]), "from"), "to", w)

    def _const_value(self, i, depth=0):
        m = self.m
        v = m.consts.get(i)
        if isinstance(v, float):
            return v
        d = m.defs.get(i)
        if d is None or depth > 8:
            return None
        if d.op == "OpConstantComposite":
            vals = [m.consts.get(x) for x in d.args]
            if vals and all(isinstance(x, float) and abs(x - vals[0]) < 1e-6 for x in vals):
                return vals[0]
        elif d.op == "OpCompositeConstruct" and d.args and len(set(d.args)) == 1:
            return self._const_value(d.args[0], depth + 1)      # splat
        elif d.op in ("OpFDiv", "OpFMul"):
            a, b = (self._const_value(x, depth + 1) for x in d.args[:2])
            if a is not None and b is not None and (d.op == "OpFMul" or b != 0):
                return a * b if d.op == "OpFMul" else a / b
        elif d.op == "OpLoad":
            got = self._host_value(d.args[0])
            if got is not None:
                self.host_const_used = got[1]
                return got[0]
        return None

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
        if fmt and "SRGB" in fmt.upper():
            self.s.set(self.key(img, "color"), "color", "srgb",
                       f"'{rid}' being sampled through an {fmt} view, which decodes sRGB", w)
            self.s.set(self.key(R, "color"), "color", "linear",
                       f"the hardware sRGB decode of '{rid}' ({fmt})", w)
            self.unite(img, R, ("frame", "unit", "conv.alpha"), w)
        else:
            self.unite(img, R, DIMS, w)

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
        bl = self.pipe.blend.get(locd[0]) or {}
        blended = bool(bl.get("enable"))
        if blended:
            self._blend_rules(v, rid, fmt, bl, locd[0], w)
        if fmt and "SRGB" in fmt.upper():
            self.s.set(self.key(v, "color"), "color", "linear",
                       f"'{rid}' using format {fmt}, which encodes sRGB on write", w)
            self.s.set(self.key((("res", rid), ()), "color"), "color", "srgb",
                       f"the sRGB encode on write to '{rid}' ({fmt})", w)
            if not blended:
                self.unite(v, (("res", rid), ()), ("conv.alpha",), w)
        else:
            self.unite(v, (("res", rid), ()),
                       tuple(d for d in DIMS if d != "conv.alpha") if blended else DIMS, w)

    def _blend_rules(self, v, rid, fmt, bl, loc, w):
        """Blend factors fix the alpha convention of the fragment output and of
        what the attachment holds afterwards (module docstring)."""
        short = lambda f: str(f).replace(BF, "")
        add = "VK_BLEND_OP_ADD"
        col = (bl.get("src_color"), bl.get("dst_color")) if bl.get("color_op", add) == add else None
        alp = (bl.get("src_alpha"), bl.get("dst_alpha")) if bl.get("alpha_op", add) == add else None
        need = {OVER_STRAIGHT: "straight", OVER_PREMUL: "premultiplied"}.get(col)
        if need is None:
            return
        desc = f"the color blend ({short(col[0])}, {short(col[1])}) of '{self.pipe.name}' into '{rid}'"
        self.s.set(self.key(v, "conv.alpha"), "conv.alpha", need,
                   f"{desc}, which expects {need}-alpha source color", w)
        has_alpha = _has_alpha(fmt)
        if alp == OVER_STRAIGHT and has_alpha and rid in getattr(self.s, "sampled", ()):
            self.s.report.add(
                "meaning", "meaning.convention-mismatch",
                f"'{self.pipe.name}' blends the alpha channel of '{rid}' with (SRC_ALPHA, "
                f"ONE_MINUS_SRC_ALPHA), which stores a * a + (1 - a) * dst.a (alpha squared); "
                f"a later pass samples '{rid}' and reads that alpha.", w,
                [f"{desc} writes rgb * a, i.e. premultiplied color",
                 "use (ONE, ONE_MINUS_SRC_ALPHA) for alpha (blendFuncSeparate), then composite "
                 "the target as premultiplied with (ONE, ONE_MINUS_SRC_ALPHA)"])
        if self.pipe.dst_alpha.get(loc) == "transparent" and has_alpha \
                and alp in (OVER_STRAIGHT, OVER_PREMUL):
            self.s.set(self.key((("res", rid), ()), "conv.alpha"), "conv.alpha", "premultiplied",
                       f"{desc}: over a transparent clear, it stores premultiplied color (rgb * a)", w)

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
