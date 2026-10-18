"""Gauge-equivariance probes: meaning checks measured by execution.

A convention (premultiplied alpha, the luminance a display assigns to 1.0,
...) is a choice of gauge for a physical quantity; changing the quantity by a
group element g (scaling coverage by s, scaling light by k) must change what
the program delivers to its sink by the element the sink's specification
prescribes: f(g . x) = rho(g) . f(x). The static solver (shader_meaning)
infers the labels on every edge and looks for cycles whose product is not the
identity. These probes *measure* the product instead: they execute the draw's
fragment shader (probe.Interp, with the uniform bytes the model driver
recorded) on a gauge orbit of one anchored input and read the transformation
at the sink. Every report therefore carries its own witness: two or three
concrete executions and the numbers they produced.

Relations (each anchored at both ends by a specification):

* COVERAGE (Porter-Duff). Blend (ONE, ONE_MINUS_SRC_ALPHA) adds the source as
  premultiplied color, so the source must be homogeneous of degree 1 in the
  fragment's coverage: scaling coverage by s scales rgb and alpha by s. Blend
  (SRC_ALPHA, ONE_MINUS_SRC_ALPHA) multiplies by alpha itself, so the source
  color must not depend on coverage (degree 0). The measured degree is the
  premultiplication count of the static dimension conv.alpha, but it is a
  real number: an encode applied to premultiplied color gives 1/2.2, a double
  premultiply 2. Coverage is the value the output alpha is a copy of.
* SDR WHITE in HDR output. Code values of VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR
  _EXT (scRGB: 1.0 = 80 cd/m2) and HDR10_ST2084 (PQ: absolute, 1.0 = 10000
  cd/m2) have absolute luminance. Display-referred SDR content (texels of an
  *_SRGB view) composited into HDR output carries SDR reference white, which
  ITU-R BT.2408 puts at 203 cd/m2 (platforms let the user move it; we accept
  100-1000 cd/m2). When a pass is a pure display mapping of SDR texels (the
  emitted light is homogeneous of degree 1 in the texels' light) and SDR white
  lands outside that band (e.g. scRGB 1.0 = 80 cd/m2), the luminance gauge is
  wrong.
* ORIENTATION (Z2). A UI library whose vertex positions are y-down display
  coordinates (signatures/ui_libs) must reach the framebuffer y-down: moving a
  vertex down must move its fragment down (vertex stage + viewport executed).

Probes run only where both anchors exist and the measured map has the shape
the relation needs (alpha linear in the chosen coverage, rgb a power law); any
other shape is opaque: no report.
"""
import math

import numpy as np

from . import probe, transfer
from .signatures import vulkan as vk

SCALES = (1.0, 0.5, 0.25)
SCRGB = "VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT"
SCRGB_NITS = 80.0
SDR_WHITE_NITS = 203.0           # ITU-R BT.2408 reference (graphics) white
PQ = "VK_COLOR_SPACE_HDR10_ST2084_EXT"
# luminance (cd/m2) of linear-light-coded display values per color space
DISPLAY_NITS = {SCRGB: lambda c: np.asarray(c, dtype=np.float64) * SCRGB_NITS,
                PQ: lambda c: transfer.pq_dec(np.asarray(c, dtype=np.float64)) * 10000.0}
# SDR white levels HDR platforms use for SDR content (BT.2408 nominal 203)
SDR_WHITE_BAND = (100.0, 1000.0)


def _entry(module, entry):
    for model, fid, name, iface in module.entries:
        if name == entry:
            return model, fid, iface
    return None, None, None


def _memory(module, values):
    blocks = {}
    for vid, d in module.defs.items():
        if d.op != "OpVariable":
            continue
        sc = d.args[0] if d.args else None
        data = None
        if sc == 9:
            data = (values or {}).get("push")
        elif sc in (2, 12):
            dd = module.decos.get(vid, {})
            if "Binding" in dd:
                data = (values or {}).get((dd.get("DescriptorSet", [0])[0], dd["Binding"][0]))
        if data is not None:
            blocks[vid] = data if isinstance(data, tuple) else (bytes(data), b"\x01" * len(data))
    return probe.Memory(module, blocks)


def _lane_sources(seed=11):
    """Source values that differ per lane: random in (0.2, 0.8)."""
    def sv(key, t, n, kind, _cache={}):
        return None
    return sv


def _output_value(run, var):
    """Value key last stored to the output variable `var` (whole vec4)."""
    got = None
    for root, path, vkey, ctx in run.stores:
        if root == var and not path:
            got = vkey
    return got


def _outputs(module, iface):
    out = {}
    from .spirv import STORAGE
    for v in iface:
        if STORAGE.get(module.storage_class(v)) == "Output" and "Location" in module.decos.get(v, {}):
            out[module.decos[v]["Location"][0]] = v
    return out


def _run(interp, fid, n, over=None):
    try:
        return probe.run_lanes(interp, fid, n, over)
    except (probe.Unsupported, probe.Divergent, KeyError, IndexError, ValueError, TypeError,
            ZeroDivisionError, FloatingPointError):
        return None


def _vec(v):
    return isinstance(v, np.ndarray) and v.dtype != bool


def _coverage_root(run, out_key, n):
    """The upstream value the output alpha is a copy of: the earliest value in
    the output's backward slice equal to alpha on every lane (a float scalar,
    or the fourth component of a 4-vector). Returns (key, component or None)."""
    out = run.values.get(out_key)
    if not _vec(out) or out.ndim != 2 or out.shape[1] < 4:
        return None
    a = out[:, 3]
    if np.allclose(a, a[0]) and (abs(a[0] - 1.0) < 1e-6 or abs(a[0]) < 1e-6):
        return None
    slice_, todo = set(), [out_key]
    while todo:
        k = todo.pop()
        if k in slice_:
            continue
        slice_.add(k)
        todo.extend(run.deps.get(k, ()))
    for k in run.order:
        if k not in slice_ or k == out_key:
            continue
        v = run.values.get(k)
        if not _vec(v):
            continue
        if v.ndim == 1 and v.shape[0] == n and np.allclose(v, a, rtol=1e-7, atol=1e-9):
            return k, None
        if v.ndim == 2 and v.shape[0] == n and v.shape[1] == 4 and \
                np.allclose(v[:, 3], a, rtol=1e-7, atol=1e-9):
            return k, 3
    return None


def _vec_width(module, tid):
    t = module.types.get(tid)
    return t.args[1] if t is not None and t.op == "OpTypeVector" and len(t.args) > 1 else 1


def _scales_rgba(module, rid):
    """True when scalar `rid` multiplies a whole 4-vector (all of RGBA, alpha
    included): a fade or tint applied uniformly, which straight and
    premultiplied blending both accept, so it is no evidence of either."""
    body = [i for f in module.functions.values() for i in f]
    splats = {i.id for i in body if i.op == "OpCompositeConstruct" and i.args
              and all(a == rid for a in i.args) and _vec_width(module, i.type) == 4}
    for i in body:
        if _vec_width(module, i.type) != 4 or len(i.args) < 2:
            continue
        if i.op == "OpVectorTimesScalar" and i.args[1] == rid:
            return True
        if i.op == "OpFMul" and (i.args[0] in splats or i.args[1] in splats):
            return True
    return False


def _power_degree(base, scaled, scales):
    """Degree d with scaled_i = s_i^d * base, when that holds (power law)."""
    ds = []
    for y, s in zip(scaled, scales):
        if s == 1.0:
            continue
        ok = np.abs(base) > 1e-4
        if not ok.any():
            return None
        ratio = y[ok] / base[ok]
        if (ratio <= 0).any():
            return None
        d = np.log(ratio) / math.log(s)
        if np.ptp(d) > 0.03:
            return None
        ds.append(float(np.mean(d)))
    if not ds or max(ds) - min(ds) > 0.03:
        return None
    return float(np.mean(ds))


def coverage_probe(use, module, entry, loc, blend):
    """COVERAGE relation for color attachment `loc` of this pipeline use.
    Returns a finding (message, notes) or None."""
    cb = vk.color_blend(blend)
    if cb is None:
        return None
    need = cb[0]                              # 1: premultiplied source, 0: straight
    model, fid, iface = _entry(module, entry)
    if fid is None or model != 4:
        return None
    var = _outputs(module, iface).get(loc)
    if var is None:
        return None
    mem = _memory(module, use.values)
    interp = probe.Interp(module, mem)
    n = 3
    base = _run(interp, fid, n)
    if base is None:
        return None
    ok = _output_value(base, var)
    if ok is None:
        return None
    root = _coverage_root(base, ok, n)
    if root is None:
        return None
    rk, comp = root
    if comp is None and _scales_rgba(module, rk[1]):
        return None                            # an opacity fade of the whole RGBA, not a premultiply
    v0 = base.values[rk]
    over = v0.copy().astype(np.float64)
    for i, s in enumerate(SCALES):
        if comp is None:
            over[i] = v0[i] * s
        else:
            over[i, comp] = v0[i, comp] * s
    run = _run(interp, fid, n, {rk: over})
    if run is None:
        return None
    out = run.values.get(_output_value(run, var))
    if not _vec(out) or out.ndim != 2:
        return None
    a = out[:, 3]
    if abs(a[0]) < 1e-4 or not np.allclose(a[1:], a[0] * np.array(SCALES[1:]), rtol=1e-3, atol=1e-6):
        return None                            # alpha is not the coverage we scaled
    rgb = out[:, :3]
    d = _power_degree(rgb[0], [rgb[1], rgb[2]], SCALES[1:])
    if d is None or abs(d - need) <= 0.1:
        return None
    if need == 1 and abs(d) <= 0.1:
        # Straight-looking color into a premultiplied blend. The source's alpha convention is not
        # anchored: it may be stored premultiplied (a premultiplied texture). Report only when that
        # hypothesis fails too (no convention of the source makes the draw Porter-Duff correct).
        return _premultiplied_hypothesis(use, interp, fid, var, base, rk, comp, n, blend, loc)
    short = lambda f: str(f).replace(vk.BF, "")
    blend_s = f"({short(blend.get('src_color'))}, {short(blend.get('dst_color'))})"
    want = "premultiplied color, which scales with coverage (degree 1)" if need else \
        "straight color, which must not change with coverage (degree 0)"
    what = {0: "straight", 1: "premultiplied", 2: "premultiplied twice"}.get(round(d)) \
        if abs(d - round(d)) < 0.05 else f"coverage^{d:.3g}"
    msg = (f"the color that '{use.name}' blends into attachment {loc} is {what} (degree {d:.3g} in "
           f"coverage), but the blend {blend_s} takes {want}.")
    notes = [f"witness: coverage {a[0]:.4g} -> output rgb {np.round(rgb[0], 4).tolist()}; "
             f"coverage x0.5 -> {np.round(rgb[1], 4).tolist()}; x0.25 -> {np.round(rgb[2], 4).tolist()}",
             "Porter-Duff over adds premultiplied color (rgb * a); it is linear in a. An encode or "
             "decode applied after premultiplying, or a second premultiply, breaks that",
             "un-premultiply before a transfer function and premultiply after it"]
    return msg, notes


def _premultiplied_hypothesis(use, interp, fid, var, base, rk, comp, n, blend, loc):
    """COVERAGE relation under the other gauge of an unanchored source: the
    4-vector the output alpha comes from holds premultiplied color, so scaling
    its coverage scales rgb and alpha together. Porter-Duff (ONE, 1 - SA) then
    needs the output to be homogeneous of degree 1 in that scaling; in
    particular a fully transparent source (rgb = a = 0) must add nothing.
    Returns a finding only when the output is affine in the scale with a
    non-zero offset (light added where coverage is zero) or a power law of a
    degree other than 1."""
    if comp != 3:
        return None
    v0 = np.asarray(base.values[rk], dtype=np.float64)
    if v0.ndim != 2 or v0.shape[1] != 4 or (np.abs(v0[:, :3]) < 1e-4).all():
        return None
    over = v0.copy()
    for i, s in enumerate(SCALES):
        over[i] = v0[i] * s
    run = _run(interp, fid, n, {rk: over})
    if run is None:
        return None
    out = run.values.get(_output_value(run, var))
    if not _vec(out) or out.ndim != 2 or out.shape[1] < 4:
        return None
    a = out[:, 3]
    if abs(a[0]) < 1e-4 or not np.allclose(a[1:], a[0] * np.array(SCALES[1:]), rtol=1e-3, atol=1e-6):
        return None
    rgb = out[:, :3]
    s = np.array(SCALES)
    if np.allclose(rgb[1:], rgb[0] * s[1:, None], rtol=1e-3, atol=1e-5):
        return None                            # degree 1: a premultiplied source is consistent
    # exact affine rgb(s) = A s + B through the three executions
    A = (rgb[0] - rgb[2]) / (s[0] - s[2])
    B = rgb[0] - A * s[0]
    if not np.allclose(rgb[1], A * s[1] + B, rtol=1e-4, atol=1e-5):
        d = _power_degree(rgb[0], [rgb[1], rgb[2]], SCALES[1:])
        if d is None:
            return None                        # neither affine nor a power law: opaque
        what = f"coverage^{d:.3g}"
        tail = f"degree {d:.3g} in the source's coverage"
    else:
        if np.max(np.abs(B)) < 1.0 / 1024:
            return None
        what = "offset"
        tail = (f"rgb = {np.round(A, 4).tolist()} * coverage + {np.round(B, 4).tolist()}: it does not "
                "vanish where coverage is 0")
    short = lambda f: str(f).replace(vk.BF, "")
    blend_s = f"({short(blend.get('src_color'))}, {short(blend.get('dst_color'))})"
    msg = (f"the color that '{use.name}' blends into attachment {loc} with {blend_s} is not premultiplied "
           f"for either convention of its source: with straight source alpha it does not scale with "
           f"coverage (degree 0), and with a premultiplied source it has {tail}.")
    notes = [f"witness (premultiplied source scaled by 1, 0.5, 0.25): output rgb "
             f"{np.round(rgb[0], 4).tolist()}, {np.round(rgb[1], 4).tolist()}, {np.round(rgb[2], 4).tolist()}; "
             f"alpha {np.round(a, 4).tolist()}",
             "Porter-Duff over with (ONE, ONE_MINUS_SRC_ALPHA) adds the source as premultiplied color: "
             "a transparent source (rgb = a = 0) must add nothing, and the source must be linear in coverage",
             "apply offsets and transfer functions to straight color, then premultiply (or scale the offset "
             "by alpha)"]
    return msg, notes


def _sdr_samples(module, run, use):
    """Sample results whose image is bound through an *_SRGB view (texels of
    display-referred SDR content)."""
    out = []
    for k, kind in run.sources.items():
        if kind != "texel":
            continue
        ins = module.defs.get(k[1])
        if ins is None or not ins.args:
            continue
        h = run.values.get((k[0], ins.args[0])) or run.values.get(((), ins.args[0]))
        var = getattr(h, "var", None)
        if var is None:
            continue
        dd = module.decos.get(var, {})
        if "Binding" not in dd:
            continue
        b = use.bindings.get((dd.get("DescriptorSet", [0])[0], dd["Binding"][0]))
        if b and vk.format_transfer(b[1]) == "srgb":
            out.append((k, b))
    return out


def sdr_white_probe(use, module, entry, loc, rid, space=None):
    """SDR WHITE relation for a fragment output presented in an HDR color
    space whose code values have absolute luminance (scRGB, HDR10 PQ)."""
    space = space or SCRGB
    model, fid, iface = _entry(module, entry)
    if fid is None or model != 4:
        return None
    var = _outputs(module, iface).get(loc)
    if var is None:
        return None
    mem = _memory(module, use.values)
    interp = probe.Interp(module, mem)
    n = 3
    base = _run(interp, fid, n)
    if base is None or _output_value(base, var) is None:
        return None
    srcs = _sdr_samples(module, base, use)
    if not srcs:
        return None
    over = {}
    levels = (1.0, 0.5, 0.25)
    for k, _ in srcs:
        v = base.values[k]
        if not _vec(v) or v.ndim != 2 or v.shape[1] < 3:
            return None
        o = np.array(v, dtype=np.float64)
        for i, lv in enumerate(levels):
            o[i, :3] = lv
            if o.shape[1] == 4:
                o[i, 3] = 1.0
        over[k] = o
    run = _run(interp, fid, n, over)
    if run is None:
        return None
    out = run.values.get(_output_value(run, var))
    if not _vec(out) or out.ndim != 2:
        return None
    code = out[:, :3]
    nits = DISPLAY_NITS[space](code)           # the light the display emits for these codes
    w = nits[0]
    if not np.allclose(nits[1], 0.5 * w, rtol=2e-3, atol=1e-3) or \
            not np.allclose(nits[2], 0.25 * w, rtol=2e-3, atol=1e-3):
        return None                            # not a pure display mapping of the SDR texels
    white = float(np.mean(w))
    if not np.allclose(w, white, rtol=0.02) or SDR_WHITE_BAND[0] <= white <= SDR_WHITE_BAND[1]:
        return None
    names = ", ".join(sorted({b[0] for _, b in srcs}))
    short = space.replace("VK_COLOR_SPACE_", "").replace("_EXT", "")
    msg = (f"'{use.name}' writes SDR content ({names}, sampled through sRGB views) to '{rid}', "
           f"presented as {short}, so that SDR white is shown at {white:.0f} cd/m2; SDR reference "
           f"white in HDR output is {SDR_WHITE_NITS:g} cd/m2 (ITU-R BT.2408), and SDR content is "
           f"shown at {100 * white / SDR_WHITE_NITS:.0f}% of that.")
    notes = [f"witness: SDR texels (1, 1, 1) -> output {np.round(code[0], 4).tolist()} = "
             f"{white:.0f} cd/m2; (0.5, 0.5, 0.5) -> {np.round(nits[1], 1).tolist()} cd/m2 (the pass is "
             "linear in the SDR texels' light)",
             "scale SDR content to the display's SDR white level (scRGB: white level / 80 cd/m2; "
             "PQ: encode white level / 10000 cd/m2)"]
    return msg, notes


SDR_SPACES = {"VK_COLOR_SPACE_SRGB_NONLINEAR_KHR", "VK_COLOR_SPACE_BT709_NONLINEAR_EXT",
              "VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT"}
HDR_REF_NITS = (203.0, 101.5, 50.75)      # BT.2408 HDR reference white, and 1/2, 1/4 of it
# reference white between 1/1000 and 10x SDR white (an implied SDR white of 20 cd/m2 to 203,000 cd/m2):
# a missing or doubled 10000/203 scale lands inside; a texel that only modulates the picture
# (an occlusion or roughness factor) puts it orders of magnitude outside
HDR_LEVEL_BAND = (1e-3, 10.0)


def _texel_sources(run):
    return [k for k, kind in run.sources.items() if kind == "texel"]


def hdr_white_probe(use, module, entry, loc, rid, fmt):
    """SDR WHITE relation, other direction: PQ-coded content (SMPTE ST 2084:
    code values are absolute, 1.0 = 10000 cd/m2) shown on an SDR display
    (sRGB-coded, 1.0 = the display's SDR white). BT.2408 puts HDR reference
    white at 203 cd/m2, so a pure display mapping (homogeneous of degree 1 in
    the PQ-decoded light) must put 203 cd/m2 near SDR 1.0; we accept an SDR
    white of 100-1000 cd/m2. The texels are fed PQ codes of 203, 101.5 and
    50.75 cd/m2; the PQ reading of the texel is not assumed: it is confirmed
    by the output being linear in that light (any other reading of the texel
    makes the three outputs non-proportional, and the probe is silent)."""
    model, fid, iface = _entry(module, entry)
    if fid is None or model != 4:
        return None
    var = _outputs(module, iface).get(loc)
    if var is None:
        return None
    mem = _memory(module, use.values)
    interp = probe.Interp(module, mem)
    n = 3
    base = _run(interp, fid, n)
    if base is None or _output_value(base, var) is None:
        return None
    srcs = _texel_sources(base)
    if not srcs:
        return None
    codes = transfer.pq_enc(np.array(HDR_REF_NITS) / 10000.0)
    over = {}
    for k in srcs:
        v = base.values[k]
        if not _vec(v) or v.ndim != 2 or v.shape[1] < 3:
            return None
        o = np.array(v, dtype=np.float64)
        for i, c in enumerate(codes):
            o[i, :3] = c
            if o.shape[1] == 4:
                o[i, 3] = 1.0
        over[k] = o
    run = _run(interp, fid, n, over)
    if run is None:
        return None
    out = run.values.get(_output_value(run, var))
    if not _vec(out) or out.ndim != 2 or out.shape[1] < 3:
        return None
    code = np.asarray(out[:, :3], dtype=np.float64)
    # the light the SDR display emits, relative to its white: an *_SRGB attachment encodes in
    # hardware (the shader writes linear), a UNORM one holds the shader's sRGB code values
    lin = code if vk.format_transfer(fmt) == "srgb" else transfer.srgb_dec(np.clip(code, 0.0, None))
    w = lin[0]
    if (w < 1e-6).any() or (code > 1.0 + 1e-6).any():
        return None
    if not np.allclose(lin[1], 0.5 * w, rtol=3e-3, atol=1e-9) or \
            not np.allclose(lin[2], 0.25 * w, rtol=3e-3, atol=1e-9):
        return None                            # not a pure display mapping of PQ light
    level = float(np.mean(w))
    if not np.allclose(w, level, rtol=0.05):
        return None
    if not (HDR_LEVEL_BAND[0] <= level <= HDR_LEVEL_BAND[1]):
        return None                            # the texel is not the picture (a factor, a mask)
    white = HDR_REF_NITS[0] / level            # cd/m2 that the display's SDR 1.0 stands for
    if SDR_WHITE_BAND[0] <= white <= SDR_WHITE_BAND[1]:
        return None
    msg = (f"'{use.name}' maps PQ-coded content to '{rid}', an SDR target, so that HDR reference white "
           f"(203 cd/m2, ITU-R BT.2408) lands at {level:.4g} of SDR white: the mapping treats SDR 1.0 as "
           f"{white:.0f} cd/m2, outside the {SDR_WHITE_BAND[0]:g}-{SDR_WHITE_BAND[1]:g} cd/m2 range of SDR "
           f"white levels (the luminance scale between the two encodings is missing or wrong).")
    notes = [f"witness: texels with PQ codes {np.round(codes, 4).tolist()} (203, 101.5, 50.75 cd/m2) -> "
             f"output {np.round(code[:, 0], 5).tolist()} = {np.round(lin[:, 0], 5).tolist()} of SDR white (the pass "
             "is linear in the decoded light)",
             "PQ (SMPTE ST 2084) code 1.0 is 10000 cd/m2; sRGB 1.0 is the display's SDR white: scale by "
             "10000 / SDR white (203 cd/m2 per BT.2408) between the two"]
    return msg, notes


def _ui_position(use, host_structs):
    """(location, struct, field, orientation) when a vertex attribute is the
    position field of a UI library vertex struct the host defines (matched by
    the binding stride = struct size and the attribute offset = field offset)."""
    from .signatures.ui_libs import UI_VERTEX
    vi = use.vertex_input or {}
    if not host_structs or not vi.get("attributes"):
        return None
    strides = {b.get("binding"): b.get("stride") for b in vi.get("bindings", [])}
    for name, fields in UI_VERTEX.items():
        hs = host_structs.get(name)
        if hs is None:
            continue
        for fname, orient in fields.items():
            f = next((x for x in hs.fields if x.name == fname), None)
            if f is None:
                continue
            for a in vi["attributes"]:
                if strides.get(a.get("binding")) == hs.size and a.get("offset") == f.offset:
                    return a.get("location"), name, fname, orient
    return None


def _input_loads(module, var):
    """SSA ids of loads of the Input variable var (directly or through an
    access chain)."""
    out, chains = set(), {var}
    for body in module.functions.values():
        for ins in body:
            if ins.op in ("OpAccessChain", "OpInBoundsAccessChain") and ins.args and ins.args[0] in chains:
                chains.add(ins.id)
    for body in module.functions.values():
        for ins in body:
            if ins.op == "OpLoad" and ins.args and ins.args[0] in chains:
                out.add(ins.id)
    return out


def orientation_probe(use, module, entry, loc, what):
    """ORIENTATION relation: y-down input coordinates must reach the
    framebuffer y-down. Executes the vertex stage on p, p + (0, d), p + (d, 0)."""
    from .spirv import STORAGE
    from .witness import _position_value
    model, fid, iface = _entry(module, entry)
    if fid is None or model != 0:
        return None
    var = next((v for v in iface if STORAGE.get(module.storage_class(v)) == "Input"
                and module.decos.get(v, {}).get("Location") == [loc]), None)
    if var is None:
        return None
    loads = _input_loads(module, var)
    mem = _memory(module, use.values)
    interp = probe.Interp(module, mem)
    n = 3
    base = _run(interp, fid, n)
    if base is None:
        return None
    keys = [k for k in base.values if k[1] in loads]
    if not keys:
        return None
    pts = np.array([[100.0, 100.0], [100.0, 140.0], [140.0, 100.0]])
    over = {}
    for k in keys:
        v = base.values[k]
        if not _vec(v) or v.ndim != 2 or v.shape[1] < 2:
            return None
        o = np.array(v, dtype=np.float64)
        o[:, :2] = pts
        over[k] = o
    run = _run(interp, fid, n, over)
    if run is None:
        return None
    pk = _position_value(module, run)
    pos = run.values.get(pk) if pk is not None else None
    if not _vec(pos) or pos.ndim != 2 or pos.shape[1] < 4 or (np.abs(pos[:, 3]) < 1e-9).any():
        return None
    ndc = pos[:, :2] / pos[:, 3:4]
    vp = use.viewport
    if vp is None or len(vp) < 4 or vp[3] == 0:
        return None
    wy = vp[1] + (ndc[:, 1] + 1.0) * 0.5 * vp[3]
    wx = vp[0] + (ndc[:, 0] + 1.0) * 0.5 * vp[2]
    dy, dx = wy[1] - wy[0], wx[2] - wx[0]
    if abs(dy) < 1e-6 or abs(dx) < 1e-6:
        return None
    if dy > 0:
        return None
    struct, field = what
    msg = (f"'{use.name}' draws {struct}.{field} positions, which are y-down display coordinates, "
           f"so that increasing y moves up the framebuffer: the UI is drawn upside down.")
    notes = [f"witness: {struct}.{field} (100, 100) -> framebuffer y {wy[0]:.1f}; (100, 140) -> "
             f"y {wy[1]:.1f} (viewport y = {vp[1]:g}, height = {vp[3]:g})",
             "Vulkan's clip space and framebuffer are +Y down: an orthographic projection for "
             "y-down input maps top (T) to clip y = -1 (swap T and B of an OpenGL-style ortho)"]
    return msg, notes


def _texel_bindings(module, run, use):
    """[(texel source key, (resource, view format))] for every sampled texel."""
    out = []
    for k, kind in run.sources.items():
        if kind != "texel":
            continue
        ins = module.defs.get(k[1])
        if ins is None or not ins.args:
            continue
        h = run.values.get((k[0], ins.args[0])) or run.values.get(((), ins.args[0]))
        var = getattr(h, "var", None)
        if var is None:
            continue
        dd = module.decos.get(var, {})
        if "Binding" not in dd:
            continue
        b = use.bindings.get((dd.get("DescriptorSet", [0])[0], dd["Binding"][0]))
        if b:
            out.append((k, b))
    return out


def unorm_bits(fmt):
    """Bits per colour channel of a UNORM format whose channels share one depth
    (R8G8B8A8_UNORM: 8, A2B10G10R10_UNORM_PACK32: 10, R16G16B16A16_UNORM: 16,
    R10X6..._UNORM: 10), else None."""
    import re
    f = str(fmt or "").replace("VK_FORMAT_", "")
    if "_UNORM" not in f or "SRGB" in f:
        return None
    comps = f.split("_")[0]
    bits = {int(b) for c, b in re.findall(r"([RGB])(\d+)", comps)}
    return bits.pop() if len(bits) == 1 else None


def limited_black(bits):
    """(Y, Cb/Cr) of black in limited (narrow) range at `bits` bits, normalised
    by 2^bits - 1 as a UNORM sample is: Y = 16 * 2^(bits-8), C = 128 * 2^(bits-8)
    (ITU-R BT.709-6 item 6.10, BT.2100-2 Table 9)."""
    s = 2.0 ** (bits - 8)
    d = 2.0 ** bits - 1.0
    return 16.0 * s / d, 128.0 * s / d


def quantization_probe(use, module, entry):
    """QUANTIZATION relation: a YCbCr conversion maps limited-range black (Y
    at the black level, chroma at the midpoint) to RGB 0 whatever its matrix.
    The black level depends on the sample's bit depth. The texels of an N-bit
    UNORM view are fed the black of each bit depth (every assignment of Y, Cb,
    Cr to r, g, b); if the output is exactly 0 for another depth's black but
    not for the view's own, the shader's offsets are the other depth's levels."""
    model, fid, iface = _entry(module, entry)
    if fid is None or model != 4:
        return None
    outs = _outputs(module, iface)
    if not outs:
        return None
    var = outs[min(outs)]
    mem = _memory(module, use.values)
    interp = probe.Interp(module, mem)
    base = _run(interp, fid, 1)
    if base is None:
        return None
    texels = [(k, b) for k, b in _texel_bindings(module, base, use) if unorm_bits(b[1])]
    if not texels:
        return None
    import itertools
    for k, (rid, fmt) in texels:
        bits = unorm_bits(fmt)
        v = base.values[k]
        if not _vec(v) or v.ndim != 2 or v.shape[1] < 3:
            continue

        def out_for(black, perm):
            y, c = black
            o = np.array(v, dtype=np.float64)
            vals = [y, c, c]
            for slot, comp in zip(perm, vals):
                o[0, slot] = comp
            run = _run(interp, fid, 1, {k: o})
            if run is None:
                return None
            got = run.values.get(_output_value(run, var))
            if not _vec(got) or got.ndim != 2 or got.shape[1] < 3:
                return None
            return np.asarray(got[0, :3], dtype=np.float64)

        for perm in itertools.permutations(range(3)):
            own = out_for(limited_black(bits), perm)
            if own is None or np.max(np.abs(own)) < 3e-4:
                continue                       # the view's own black works (or no YCbCr shape)
            for other in (8, 10, 12, 16):
                if other == bits:
                    continue
                got = out_for(limited_black(other), perm)
                if got is None or np.max(np.abs(got)) > 1e-5:
                    continue
                yo, co = limited_black(other)
                yb, cb = limited_black(bits)
                msg = (f"'{use.name}' converts '{rid}' ({fmt.replace('VK_FORMAT_', '')}, {bits}-bit samples "
                       f"normalised by {2 ** bits - 1}) with {other}-bit limited-range levels: black "
                       f"Y = {yo:.6g}, chroma = {co:.6g} (= {16 * 2 ** (other - 8)}/{2 ** other - 1}, "
                       f"{128 * 2 ** (other - 8)}/{2 ** other - 1}) maps to RGB 0, but this view's black "
                       f"is Y = {yb:.6g}, chroma = {cb:.6g}, which maps to {np.round(own, 5).tolist()}.")
                notes = [f"witness: texel (Y, Cb, Cr in channels {perm}) = {other}-bit black -> output "
                         f"{np.round(got, 7).tolist()}; = {bits}-bit black -> {np.round(own, 5).tolist()}",
                         "limited-range black is 16 * 2^(N-8) and the chroma midpoint 128 * 2^(N-8) at N bits "
                         "(ITU-R BT.709-6 6.10, BT.2100-2 Table 9); a UNORM sample divides by 2^N - 1",
                         f"derive the offsets and range scale from the sample's bit depth ({bits})"]
                return msg, notes
    return None


def _float_inputs(module, iface):
    from .spirv import STORAGE
    out = {}
    for v in iface:
        if STORAGE.get(module.storage_class(v)) != "Input" or "Location" not in module.decos.get(v, {}):
            continue
        t = module.pointee(module.defs[v].type)
        if module.tkind(t) == "vector" and module.tkind(module.types[t].args[0]) == "float" \
                and module.types[t].args[1] >= 2:
            out[module.decos[v]["Location"][0]] = (v, module.types[t].args[1])
    return out


def handedness(use, module, entry):
    """{input location: (sign, det, dims)}: the orientation of the map from a
    vertex input's xy(z) to window coordinates (vertex stage + viewport),
    measured by executing the stage on p, p + h e_x, p + h e_y (, p + h e_z).
    Vulkan decides facing from the sign of a triangle's window-space area
    (Vulkan spec, "Basic Polygon Rasterization"), so a map of negative
    determinant turns every front face of a mesh into a back face."""
    from .witness import _position_value
    model, fid, iface = _entry(module, entry)
    if fid is None or model != 0:
        return {}
    vp = use.viewport
    if vp is None or len(vp) < 4 or vp[2] == 0 or vp[3] == 0:
        return {}
    mem = _memory(module, use.values)
    interp = probe.Interp(module, mem)
    n = 4
    base = _run(interp, fid, n)
    if base is None:
        return {}
    out = {}
    for loc, (var, width) in _float_inputs(module, iface).items():
        loads = _input_loads(module, var)
        keys = [k for k in base.values if k[1] in loads]
        if not keys:
            continue
        dims = 3 if width >= 3 else 2
        p0 = np.asarray(base.values[keys[0]], dtype=np.float64)
        if p0.ndim != 2:
            continue
        h = 1e-2 * max(1.0, float(np.max(np.abs(p0[0, :dims]))))
        over = {}
        for k in keys:
            o = np.array(base.values[k], dtype=np.float64)
            if o.ndim != 2 or o.shape[1] < dims or o.shape[0] != n:
                over = None
                break
            o[:] = o[0]
            for j in range(dims):
                o[j + 1, j] += h
            over[k] = o
        if not over:
            continue
        run = _run(interp, fid, n, over)
        if run is None:
            continue
        pk = _position_value(module, run)
        pos = run.values.get(pk) if pk is not None else None
        if not _vec(pos) or pos.ndim != 2 or pos.shape[0] < n or pos.shape[1] < 4:
            continue
        if (pos[:, 3] <= 1e-9).any():
            continue                            # behind the eye: no window position
        ndc = pos[:, :3] / pos[:, 3:4]
        win = np.stack([vp[0] + (ndc[:, 0] + 1.0) * 0.5 * vp[2],
                        vp[1] + (ndc[:, 1] + 1.0) * 0.5 * vp[3], ndc[:, 2]], axis=1)
        J = (win[1:dims + 1] - win[0]).T / h      # columns: d(window)/d(input_j)
        if dims == 3:
            det = float(np.linalg.det(J[:3, :3]))
            if abs(det) < 1e-12:
                det, dims = float(np.linalg.det(J[:2, :2])), 2   # flat geometry: z unused
        else:
            det = float(np.linalg.det(J[:2, :2]))
        if abs(det) < 1e-9 * max(1.0, float(np.max(np.abs(J)))) ** dims:
            continue
        out[loc] = (1 if det > 0 else -1, det, dims)
    return out


CULL_DYNAMIC = {"VK_DYNAMIC_STATE_CULL_MODE", "VK_DYNAMIC_STATE_CULL_MODE_EXT",
                "VK_DYNAMIC_STATE_FRONT_FACE", "VK_DYNAMIC_STATE_FRONT_FACE_EXT"}


def winding_check(uses, report):
    """WINDING relation (Z2): draws through one pipeline share its frontFace and
    cullMode, so the map from a vertex input to window coordinates must keep one
    orientation across them. If one draw's map has the opposite determinant sign
    of another's (a Y-flipped projection, a negative viewport height, a mirrored
    transform), the pipeline culls the front faces of one of them (glTF 2.0
    3.7.4: a negative-determinant transform reverses the winding order, so it
    needs the opposite frontFace)."""
    groups = {}
    for use in uses:
        ra = (use.vertex_input or {}).get("raster") or {}
        if not ra or not ra.get("cull_mode") or set(ra.get("dynamic") or ()) & CULL_DYNAMIC:
            continue
        if "TRIANGLE" not in str(ra.get("topology") or "TRIANGLE"):
            continue
        vtx = [(m, e) for m, e in use.stages if _entry(m, e)[0] == 0]
        if not vtx:
            continue
        try:
            got = handedness(use, vtx[0][0], vtx[0][1])
        except Exception:
            got = {}
        for loc, (sign, det, dims) in got.items():
            groups.setdefault((ra.get("pipeline"), loc), []).append((sign, det, dims, use, ra))
    for (pipe, loc), rows in groups.items():
        signs = {r[0] for r in rows}
        if len(signs) < 2:
            continue
        pos = next(r for r in rows if r[0] > 0)
        neg = next(r for r in rows if r[0] < 0)
        ra = pos[4]
        ff = "COUNTER_CLOCKWISE" if ra.get("front_face_ccw") else "CLOCKWISE"
        vp = lambda u: tuple(round(float(x), 1) for x in (u.viewport or ())[:4])
        msg = (f"'{pos[3].name}' is drawn through one pipeline (frontFace {ff}, cullMode {ra.get('cull_mode')}) "
               f"with maps of opposite orientation: vertex input {loc} reaches the window with determinant "
               f"{pos[1]:+.4g} in one draw and {neg[1]:+.4g} in another, so one of the two culls its front "
               "faces.")
        notes = [f"witness: the vertex stage executed on p, p + h*e_x, p + h*e_y{', p + h*e_z' if pos[2] == 3 else ''} "
                 f"with each draw's recorded uniforms and viewport ({vp(pos[3])} vs {vp(neg[3])}) gives "
                 f"window-space Jacobian determinants {pos[1]:+.4g} and {neg[1]:+.4g}",
                 "Vulkan decides facing from the sign of the window-space area; a Y flip in the projection "
                 "or viewport reverses it",
                 "key the pipeline (frontFace) to the orientation of the transform, or flip the winding "
                 "together with the projection"]
        report.add("meaning", "meaning.convention-mismatch", msg, f"{pos[3].name} (pipeline {pipe})", notes)
    return report


def check(uses, report, host_structs=None):
    """Run every probe on every pipeline use (deduplicated per pipeline and
    attachment)."""
    seen = set()
    try:
        winding_check(uses, report)
    except Exception:                            # probes are best effort
        pass
    for use in uses:
        ui = _ui_position(use, host_structs)
        if ui is not None and ("ori", use.name) not in seen:
            seen.add(("ori", use.name))
            vtx = [(m, e) for m, e in use.stages if _entry(m, e)[0] == 0]
            if vtx:
                try:
                    got = orientation_probe(use, vtx[0][0], vtx[0][1], ui[0], (ui[1], ui[2]))
                except Exception:
                    got = None
                if got:
                    report.add("meaning", "meaning.convention-mismatch", got[0],
                               f"{use.name} (vertex entry {vtx[0][1]})", got[1])
        frag = [(m, e) for m, e in use.stages if _entry(m, e)[0] == 4]
        if not frag:
            continue
        m, entry = frag[-1]
        if ("quant", use.name) not in seen:
            seen.add(("quant", use.name))
            try:
                got = quantization_probe(use, m, entry)
            except Exception:
                got = None
            if got:
                report.add("meaning", "meaning.unit-mismatch", got[0], f"{use.name} (fragment entry {entry})",
                           got[1])
        for loc, bl in (use.blend or {}).items():
            key = ("cov", use.name, loc)
            if key in seen or not bl or not bl.get("enable"):
                continue
            seen.add(key)
            try:
                got = coverage_probe(use, m, entry, loc, bl)
            except Exception:                    # probes are best effort
                got = None
            if got:
                report.add("meaning", "meaning.convention-mismatch", got[0],
                           f"{use.name} (fragment entry {entry})", got[1])
        spaces = getattr(use, "color_spaces", None) or {}
        for loc, (rid, fmt) in (use.attachments or {}).items():
            if spaces.get(rid) in SDR_SPACES and ("hdrwhite", use.name, loc) not in seen:
                seen.add(("hdrwhite", use.name, loc))
                try:
                    got = hdr_white_probe(use, m, entry, loc, rid, fmt)
                except Exception:
                    got = None
                if got:
                    report.add("meaning", "meaning.color-mismatch", got[0],
                               f"{use.name} (fragment entry {entry})", got[1], severity="warning")
            if spaces.get(rid) not in DISPLAY_NITS:
                continue
            key = ("white", use.name, loc)
            if key in seen:
                continue
            seen.add(key)
            try:
                got = sdr_white_probe(use, m, entry, loc, rid, spaces.get(rid))
            except Exception:
                got = None
            if got:
                report.add("meaning", "meaning.color-mismatch", got[0],
                           f"{use.name} (fragment entry {entry})", got[1], severity="warning")
    return report
