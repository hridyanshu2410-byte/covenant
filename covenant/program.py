"""Run every COVENANT checker on one recorded program: whole-frame timing,
per-draw structure and conventions, host layouts and conventions, and the
host+shader meaning analysis.

    check_program(model, modules, host_files, host_args) -> Report

`model` is host_trace.load_trace(...); `modules` are the program's compiled
SPIR-V modules (spirv.compile_slang / compile_glsl); `host_files` the host C++
sources parsed with libclang (empty to skip the host front end).

Each draw becomes a shader_meaning.PipelineUse that also carries the
fixed-function state the meaning rules use: the encoding its presented images
must hold (present_encoding, from the swapchain color space), the blend state
of each color attachment, whether the attachment was transparent before the
draw (cleared with alpha < 1; no alpha channel and presented images count as
opaque; loaded contents are unknown), and the first viewport.
"""
import hashlib
import os
from pathlib import Path

from . import probe, shader_meaning as sm, host_static as hs, timing, witness
from . import equivariance, host_meaning
from .diagnostics import Report
from .meaning_solver import MeaningSolver
from .signatures import vulkan as vk
from .values import Values

MAX_VALUE_VARIANTS = int(os.environ.get("COVENANT_MAX_VALUE_VARIANTS", "3"))
REVERSE_Z_OK = {"VK_COMPARE_OP_GREATER", "VK_COMPARE_OP_GREATER_OR_EQUAL"}
# ablation: COVENANT_MEANING=legacy runs the per-bug rules from before the
# labelled redesign (covenant/legacy), "new" (default) the labelled solver
MEANING = os.environ.get("COVENANT_MEANING", "new")


def _projection_witness(stages, values, cache):
    """witness.projection facts of the last pre-rasterization stage, with the
    draw's concrete uniform / push-constant bytes."""
    vs = [(m, e) for m, e in stages if any(n == e and mdl in (0, 1, 2, 3) for mdl, _, n, _ in m.entries)]
    if not vs:
        return None
    m, entry = vs[-1]
    fid = next(f for mdl, f, n, _ in m.entries if n == entry)
    blocks = {}
    for vid, dd in m.defs.items():
        if dd.op != "OpVariable":
            continue
        sc = dd.args[0] if dd.args else None
        data = None
        if sc == 9:
            data = values.get("push")
        elif sc in (2, 12):
            dec = m.decos.get(vid, {})
            if "Binding" in dec:
                data = values.get((dec.get("DescriptorSet", [0])[0], dec["Binding"][0]))
        if data is not None:
            blocks[vid] = data
    key = (id(m), fid, tuple(sorted((k, hashlib.sha1(v[0]).hexdigest()) for k, v in blocks.items())))
    if key not in cache:
        cache[key] = witness.projection(m, fid, probe.Memory(m, blocks))
    return cache[key]


def _padding_witness(stages, values, where, seen, r):
    """Nonzero floats the host wrote into the padding of a uniform block."""
    for m, entry in stages:
        for vid, dd in m.defs.items():
            if dd.op != "OpVariable" or not dd.args or dd.args[0] != 2:
                continue
            dec = m.decos.get(vid, {})
            if "Binding" not in dec:
                continue
            got = values.get((dec.get("DescriptorSet", [0])[0], dec["Binding"][0]))
            if got is None:
                continue
            hits = witness.padding_writes(m, vid, got[0], got[1])
            if hits:
                # a host vec4 mirroring a vec3 member writes one constant 4th component
                # (alpha / w) into the vec3's tail word: the offsets agree. A packed
                # (12-byte) host array instead puts varying data there, which stays reported.
                tails = hs.vec3_tails(m, m.pointee(dd.type))
                tv = {h[1] for h in hits if h[0] in tails}
                if len(tv) == 1:
                    hits = [h for h in hits if h[0] not in tails]
            key = (id(m), vid, tuple(h[0] for h in hits))
            if not hits or key in seen:
                continue
            seen.add(key)
            blk = m.struct_name(m.pointee(dd.type))
            off, v, before, after = hits[0]
            r.add("structure", "structure.host-layout",
                  f"the host writes {v:g} into padding byte {off} of uniform block '{blk}' (between "
                  f"'{before}' and '{after}' in the SPIR-V layout): the host's offsets do not follow "
                  "the block layout.", where,
                  [f"{len(hits)} padding word(s) hold nonzero floats: "
                   + ", ".join(f"byte {o} = {x:g}" for o, x, _, _ in hits[:6]),
                   "vec3/vec4 members are 16-byte aligned in std140; match the host offsets to the "
                   "shader's Offset decorations"], severity="warning")


def _depth_direction(module, d, kind, where, r, proj=None):
    """Depth direction (Z2: forward / reversed) as signatures: a projection
    declared reverse-Z, the draw's VkCompareOp and its depth clear value each
    state the direction of the one depth buffer; any two that differ close a
    cycle with residual 'reverse'."""
    s = MeaningSolver(r)
    node = ("depth", d.seq)
    for (st, mem), tag in (module.meta.get("fields", {}) if module is not None else {}).items():
        if tag.get("conv.depth") == "reverse_z":
            s.set(node, "depth.dir", "reversed",
                  f"'{st}.{mem}', declared reverse-Z (near = 1, far = 0)", where)
            break
    cd = vk.COMPARE_DIR.get(d.depth_compare)
    if cd is not None:
        s.set(node, "depth.dir", "reversed" if cd else "forward",
              f"depth compare {d.depth_compare} at this {kind} (the nearer fragment wins only if "
              f"nearer means {'larger' if cd else 'smaller'} depth)", where)
    cl = vk.clear_dir(d.depth_clear)
    if cl is not None:
        s.set(node, "depth.dir", "reversed" if cl else "forward",
              f"the depth clear to {d.depth_clear} (the far plane of "
              f"{'reversed' if cl else 'forward'} depth)", where)
    if proj is not None and d.depth_test:
        s.set(node, "depth.dir", "reversed" if proj["direction"] else "forward",
              f"the projection reaching the position output (with the recorded uniform values): "
              f"nearer points get {'larger' if proj['direction'] else 'smaller'} depth "
              f"(z/w = {proj['near_ndc']:.3f} at w = 1, tending to {proj['alpha']:.3f})", where,
              weak=False)
    s.finish()


def module_for(sha, modules):
    for m in modules:
        if hashlib.sha1(Path(m.path).read_bytes()).hexdigest() == sha:
            return m
    return None


def spirv_version(module):
    w = int.from_bytes(Path(module.path).read_bytes()[4:8], "little")
    return (w >> 16) & 0xFF, (w >> 8) & 0xFF


def used_bindings(module, entry):
    """(set, binding) -> variable name, for the descriptor globals the entry
    point statically uses (reachable through its call graph,
    spirv.entry_bindings). A module that declares a resource the entry never
    touches does not need it bound, whatever the SPIR-V version (before 1.4
    the interface lists only Input / Output, so it cannot be used for this)."""
    from . import spirv
    used = spirv.entry_bindings(module, entry) or {}
    names = {}
    for v, d in module.decos.items():
        if "Binding" in d:
            names.setdefault((d.get("DescriptorSet", [0])[0], d["Binding"][0]), module.name(v))
    return {k: names.get(k, f"set {k[0]} binding {k[1]}") for k in used}


def present_encoding(swapchain):
    """(encoding, origin) the display expects in a swapchain image, from its
    VkColorSpaceKHR (shader_meaning.COLOR_SPACE_ENCODING). Traces without a
    color space keep the old assumption of an sRGB display."""
    cs = getattr(swapchain, "color_space", None)
    if cs is None:
        return ("srgb", f"'{swapchain.name}' being shown on an sRGB display")
    enc = sm.COLOR_SPACE_ENCODING.get(cs)
    short = str(cs).replace("VK_COLOR_SPACE_", "")
    what = {"srgb": "sRGB-encoded", "pq": "PQ (SMPTE ST 2084) encoded BT.2020",
            "hlg": "HLG-encoded", "linear": "linear"}.get(enc, enc)
    return (enc, f"'{swapchain.name}' being presented with color space {short} "
                 f"(format {swapchain.format}), which expects {what} values")


def check_program(model, modules, host_files=(), host_args=(), report=None, meaning=None,
                  io_trace=None):
    r = report if report is not None else Report()
    meaning = meaning or MEANING
    if meaning == "legacy":
        from .legacy import shader_meaning as smx
    else:
        smx = sm
    vals = Values(model) if meaning != "legacy" else None
    if vals is not None and io_trace:
        vals.load_io(io_trace)
    proj_cache, pad_seen, proj_seen = {}, set(), set()
    if meaning != "legacy":
        for where, msg, notes in witness.extents(model.records, model.name):
            r.add("structure", "structure.extent-mismatch", msg, where, notes)
    # ---- timing over the whole recorded run
    timing.check(model.frame, r)

    # ---- per draw: bindings, conventions; collect pipeline uses for meaning
    presented = {img: present_encoding(s) for s in model.swapchains.values() for img in s.images}
    spaces = {img: getattr(s, "color_space", None) for s in model.swapchains.values()
              for img in s.images}
    uses, seen, value_variants = [], set(), {}
    for d in model.draws:
        p = model.pipelines.get(d.pipeline)
        if p is None:
            continue
        stages = []
        for stage, sha, entry in p.stages:
            m = module_for(sha, modules)
            if m is not None:
                stages.append((m, entry))
        where = f"{model.name(d.pipeline)} {d.fn} (trace seq {d.seq})"
        proj, declared_rz = None, False
        if meaning != "legacy" and d.kind == "draw" and stages:
            proj = _projection_witness(stages, vals.for_draw(d), proj_cache)
            bad = witness.projection_violation(proj)
            if bad and (d.pipeline, bad) not in proj_seen:
                proj_seen.add((d.pipeline, bad))
                r.add("meaning", "meaning.convention-mismatch", bad, where,
                      ["the depth of a point at distance t along the view axis is alpha + beta / t "
                       "(witness from the recorded matrix values)",
                       "check the projection's depth-range remap (e.g. a transposed "
                       "OpenGL-to-Vulkan correction matrix)"])
        for m, entry in stages:
            for key, var in used_bindings(m, entry).items():
                if key not in d.descriptors or d.descriptors[key].resource is None:
                    r.add("structure", "structure.missing-binding",
                          f"'{model.name(d.pipeline)}' reads '{var}' (set {key[0]}, binding "
                          f"{key[1]}) but nothing is bound there at this {d.kind}.", where,
                          ["bind a descriptor set that provides it before the call"])
            conv = {v.get("conv.depth") for (_, _), v in m.meta.get("fields", {}).items()}
            if meaning != "legacy":
                if "reverse_z" in conv and d.depth_test:
                    declared_rz = True
                    _depth_direction(m, d, d.kind, where, r, proj)
            elif "reverse_z" in conv and d.depth_test:
                bad = []
                if d.depth_compare not in REVERSE_Z_OK:
                    bad.append(f"depth compare {d.depth_compare}")
                if d.depth_clear not in (None, 0.0):
                    bad.append(f"depth cleared to {d.depth_clear}")
                if bad:
                    r.add("meaning", "meaning.convention-mismatch",
                          f"the projection is declared reverse-Z, but this {d.kind} uses "
                          f"{' and '.join(bad)}, which assume forward Z.", where,
                          ["reverse-Z needs GREATER (or GREATER_OR_EQUAL) and a depth clear of 0.0"])
        if meaning != "legacy":
            _padding_witness(stages, vals.for_draw(d), where, pad_seen, r)
        if proj is not None and not declared_rz and d.depth_test:
            _depth_direction(None, d, d.kind, where, r, proj)
        bindings = {k: (b.resource, b.view_format) for k, b in d.descriptors.items() if b.resource}
        colors = [a for a in d.attachments if a[0] == "color"]
        atts = {i: (a[1], a[2]) for i, a in enumerate(colors)}
        blend = {i: b for i, b in enumerate(p.record.get("blend") or []) if isinstance(b, dict)}
        dst_alpha = {}
        for i, a in enumerate(colors):
            clear = (getattr(d, "color_clears", None) or {}).get(a[1])
            if not sm._has_alpha(a[2]) or a[1] in presented:
                dst_alpha[i] = "opaque"        # no alpha channel / shown on an opaque display
            elif clear is not None and len(clear) >= 4:
                dst_alpha[i] = "opaque" if clear[3] >= 1.0 else "transparent"
        vp = getattr(d, "viewport", None)
        dv = vals.for_draw(d) if vals is not None else {}
        vsig = tuple(sorted((str(k), hashlib.sha1(v[0]).hexdigest()) for k, v in dv.items()))
        sig = (d.pipeline, tuple(sorted(bindings.items())), tuple(sorted(atts.items())),
               tuple(sorted(dst_alpha.items())), vp is not None and vp[3] < 0, vsig)
        # per-draw uniform bytes (e.g. one model matrix per object) make every
        # draw a new use; the meaning of a pipeline does not change with them,
        # so at most MAX_VALUE_VARIANTS concrete variants per pipeline state are
        # probed (the probe and fingerprint dominate the check time otherwise)
        sig0 = sig[:-1]
        if stages and sig not in seen and value_variants.get(sig0, 0) < MAX_VALUE_VARIANTS:
            seen.add(sig)
            value_variants[sig0] = value_variants.get(sig0, 0) + 1
            pres = {a[1]: presented[a[1]] for a in colors if a[1] in presented}
            if meaning == "legacy":
                uses.append(smx.PipelineUse(model.name(d.pipeline), stages, bindings, atts,
                                            pres, blend, dst_alpha, vp))
            else:
                uses.append(sm.PipelineUse(model.name(d.pipeline), stages, bindings, atts,
                                           pres, blend, dst_alpha, vp, dv,
                                           {a[1]: spaces[a[1]] for a in colors if a[1] in spaces},
                                           {"bindings": p.record.get("vertex_bindings") or [],
                                            "attributes": p.record.get("vertex_attributes") or [],
                                            # rasterization state for the winding relation
                                            "raster": {"pipeline": d.pipeline,
                                                       "cull_mode": p.record.get("cull_mode"),
                                                       "front_face_ccw": p.record.get("front_face_ccw"),
                                                       "dynamic": p.record.get("dynamic_states") or [],
                                                       "topology": p.record.get("topology")}}))

    # ---- host source: layouts and meaning flows
    try:
        facts = hs.parse(list(host_files), args=list(host_args)) if host_files else None
    except hs.HostParseError as e:              # visible, then the inferred-only checks
        r.meta["host_static"] = f"failed: {str(e)[:300]}"
        r.add("structure", "host.parse-failed",
              "the host source could not be parsed, so the host-side layout, clip-range and convention "
              "checks did not run; the result covers only what the trace and shaders show.",
              notes=[str(e)[:300], f"libclang {hs._libclang_major()}, builtin headers {hs.CLANG_INC}"],
              severity="warning")
        host_files = ()
    if not host_files:
        if meaning == "legacy":
            smx.check(uses, (), r)
        else:
            smx.check(uses, (), r, None, witness.host_pixel_flows(vals, model.name))
            equivariance.check(uses, r)
        return r
    r.meta["host_static"] = "parsed"
    hs.check_layouts(facts, modules, r)
    hs.check_clip_depth(facts, modules, r)
    hmx = None
    if meaning == "legacy":
        hs.check_angle_units(facts, r)
    else:
        hmx = host_meaning.check(facts.tus, facts.files, r)   # whole-host convention typing
    if meaning == "legacy":
        smx.check(uses, hs.meaning_seeds(facts, modules), r, hs.host_values(facts, modules))
    else:
        smx.check(uses, hs.meaning_seeds(facts, modules) + host_meaning.gpu_seeds(hmx, uses), r,
                  hs.host_values(facts, modules), witness.host_pixel_flows(vals, model.name))
        equivariance.check(uses, r, facts.structs)            # measured gauge relations
    return r
