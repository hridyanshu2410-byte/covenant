"""Witness invariants on concrete values.

A witness is a concrete number the program computes (with the bytes the model
driver recorded, or constants in the shader) that violates an invariant every
correct program satisfies. No witness, no error: when a value is not concrete
the check says nothing.

* color_matrix: an RGB -> RGB primaries conversion maps white to white
  (its rows sum to 1). A 3x3 matrix applied to color whose *columns* sum to 1
  while its rows do not is applied transposed.
* projection: the clip position as an affine function of the vertex position
  (probe.affine_map). Along the central ray, z_ndc(t) = alpha + beta / t; a
  perspective projection for Vulkan's [0, 1] depth has alpha >= 1 - eps
  (forward) or alpha <= eps (reversed). alpha strictly inside (eps, 1 - eps)
  means the depth range remap is wrong. The sign of beta says whether nearer
  points get smaller (forward) or larger (reversed) depth: the "nearer wins"
  direction the draw's compare op must agree with.
* extents: framebuffer and render area extents fit inside every attachment
  (VUID-VkFramebufferCreateInfo-flags-04533 / -04534,
  VUID-VkRenderingInfo-pNext-06079 / -06080).
* padding: bytes the host writes into a uniform block's padding (between
  members of the SPIR-V block layout) hold nonzero finite floats: the host
  struct does not follow the block layout.
* host pixel path: a readback buffer filled with the driver's ramp, followed
  to the bytes the program hands an image writer (values.Values.io), fitted
  against the transfer library.
"""
import math
import struct as _st

import numpy as np

from . import probe, transfer
from .signatures import image_io, vulkan as vk

EPS = 0.05


# ------------------------------------------------------------ helpers
def _closure(run, key):
    seen, todo = set(), [key]
    while todo:
        k = todo.pop()
        if k in seen:
            continue
        seen.add(k)
        todo.extend(run.deps.get(k, ()))
    return seen


def _concrete(run, key):
    """key's value depends on no unknown uniform parameter."""
    return not any(run.sources.get(k) in ("param", "uniform", "storage")
                   for k in _closure(run, key))


def _const_lanes(v):
    return isinstance(v, np.ndarray) and v.shape[0] > 0 and np.allclose(v, v[:1])


# ------------------------------------------------------------ color matrix
def color_matrix(module, fp, loc):
    """[(where, message, notes)] for transposed color matrices in one entry
    point (fp = probe.Fingerprint of a fragment entry)."""
    out = []
    if fp is None or fp.error or fp.base is None:
        return out
    run, m = fp.base, module
    for k in run.order:
        d = m.defs.get(k[1]) if isinstance(k, tuple) and len(k) == 2 else None
        if d is None or d.op not in ("OpMatrixTimesVector", "OpVectorTimesMatrix"):
            continue
        mat_id, vec_id = (d.args[0], d.args[1]) if d.op == "OpMatrixTimesVector" else (d.args[1], d.args[0])
        mk = (k[0], mat_id) if (k[0], mat_id) in run.values else ((), mat_id)
        vk_ = (k[0], vec_id) if (k[0], vec_id) in run.values else ((), vec_id)
        M = run.values.get(mk)
        if M is None:
            try:
                M = probe.Interp(m).const(mat_id, 1)      # a constant matrix in the shader
            except Exception:
                continue
        if not isinstance(M, np.ndarray) or M.ndim != 3 or M.shape[1:] != (3, 3):
            continue
        if not _const_lanes(M) or not _concrete(run, mk):
            continue
        if vk_ not in fp.head_of:                    # the vector must vary per pixel (a color)
            continue
        C = M[0]                                     # C[col][row]
        A = C.T                                      # math matrix of M * v
        E = A if d.op == "OpMatrixTimesVector" else A.T
        rows, cols = E @ np.ones(3), E.T @ np.ones(3)
        if np.allclose(E, np.eye(3), atol=1e-6):
            continue
        if np.max(np.abs(cols - 1)) < 2e-3 and np.max(np.abs(rows - 1)) > 1e-2:
            w = loc(d)
            out.append((w, "the 3x3 color matrix applied here maps white (1, 1, 1) to "
                           f"({rows[0]:.3f}, {rows[1]:.3f}, {rows[2]:.3f}); its columns, not its rows, "
                           "sum to 1, so it is applied transposed.",
                        ["an RGB primaries conversion preserves white: each row sums to 1",
                         f"the transposed matrix maps white to ({cols[0]:.3f}, {cols[1]:.3f}, {cols[2]:.3f})",
                         "swap the operands of mul() (or transpose the matrix)"]))
    return out


# ------------------------------------------------------------ projection
def _position_value(module, run):
    m = module
    for var, path, vkey, ctx in run.stores:
        dd = m.decos.get(var, {})
        if dd.get("BuiltIn") == [0] and not path:
            return vkey
        t = m.pointee(m.defs[var].type)
        if m.tkind(t) == "struct" and path and isinstance(path[0], int):
            md = m.member_decos.get((t, path[0]), {})
            if md.get("BuiltIn") == [0] and len(path) == 1:
                return vkey
    return None


def projection(module, fid, memory=None):
    """Projection facts of the vertex entry fid, or None when not concrete:
    {"alpha", "direction" (0 forward, 1 reversed), "near_ndc", "input"}."""
    interp = probe.Interp(module, memory)
    try:
        base = probe.run_lanes(interp, fid, 1)
    except Exception:
        return None
    pos = _position_value(module, base)
    if pos is None:
        return None
    srcs = [k for k, kind in base.sources.items() if kind == "input"]
    clo = _closure(base, pos)
    if any(base.sources.get(k) in ("param", "uniform", "storage") for k in clo):
        return None
    cands = [k for k in srcs if k in clo and probe._float_kind(module, base.types.get(k)) in (3, 4)]
    for head in cands:
        n_in = probe._float_kind(module, base.types.get(head))
        try:
            got, run = probe.affine_map(module, fid, head, n_in, memory)
        except Exception:
            continue
        if run is None or pos not in got:
            continue
        A, b = got[pos]
        if A.shape[0] != 4:
            continue
        if n_in == 4:                    # homogeneous input: w = 1
            b = b + A[:, 3]
            A = A[:, :3]
        res = _depth_along_central_ray(A, b)
        if res is not None:
            res["input"] = head
            return res
    return None


def _depth_along_central_ray(A, b):
    Axy, bxy = A[:2], b[:2]
    try:
        _, sv, vt = np.linalg.svd(Axy)
    except np.linalg.LinAlgError:
        return None
    d = vt[-1]
    if np.linalg.norm(Axy @ d) > 1e-6 * max(1.0, np.linalg.norm(Axy)):
        return None
    x0 = np.linalg.lstsq(Axy, -bxy, rcond=None)[0]
    aw, bw = float(A[3] @ d), float(A[3] @ x0 + b[3])
    if abs(aw) < 1e-9:
        return None                       # orthographic: w does not grow with distance
    if aw < 0:
        d, aw = -d, -aw
    az = float(A[2] @ d)
    bz = float(A[2] @ x0 + b[2])
    alpha = az / aw
    # z_ndc at w = 1 and w = 10 (both in front of the camera)
    def ndc(wv):
        t = (wv - bw) / aw
        return (az * t + bz) / wv
    near, far = ndc(1.0), ndc(10.0)
    if not (math.isfinite(alpha) and math.isfinite(near) and math.isfinite(far)) or abs(far - near) < 1e-9:
        return None
    return {"alpha": alpha, "direction": 0 if far > near else 1, "near_ndc": near}


def projection_violation(facts):
    """Message when the far-plane asymptote lies strictly inside (eps, 1 - eps)."""
    if not facts:
        return None
    a = facts["alpha"]
    if EPS < a < 1 - EPS:
        return (f"distant points converge to clip depth z/w = {a:.3f}: a perspective projection "
                "for Vulkan's [0, 1] depth range converges to 1 (or to 0 with reverse-Z), so the "
                "depth range of the projection reaching the position output is wrong")
    return None


# ------------------------------------------------------------ extents
def extents(records, name=lambda x: x):
    """[(where, message, notes)] for framebuffers and render areas larger than
    their attachments."""
    img_ext, view_img, fb = {}, {}, {}
    out = []
    for r in records:
        fn = r.get("fn", "")
        if fn == "vkCreateImage":
            e = r.get("extent") or []
            if len(e) >= 2:
                img_ext[r.get("image")] = (e[0], e[1])
        elif fn == "vkCreateSwapchainKHR":
            e = r.get("extent") or []
            for im in r.get("images", []) or []:
                if len(e) >= 2:
                    img_ext[im] = (e[0], e[1])
        elif fn == "vkGetSwapchainImagesKHR":
            pass
        elif fn == "vkCreateImageView":
            view_img[r.get("view")] = r.get("image")
        elif fn == "vkCreateFramebuffer":
            e = r.get("extent") or []
            if len(e) < 2:
                continue
            fb[r.get("framebuffer")] = (e[0], e[1], r.get("attachments") or [])
            for v in r.get("attachments") or []:
                ie = img_ext.get(view_img.get(v))
                if ie and (e[0] > ie[0] or e[1] > ie[1]):
                    out.append((f"vkCreateFramebuffer (trace seq {r['seq']})",
                                f"framebuffer '{name(r.get('framebuffer'))}' is {e[0]}x{e[1]}, larger than its "
                                f"attachment '{name(view_img.get(v))}' ({ie[0]}x{ie[1]}).",
                                ["each attachment must be at least as large as the framebuffer "
                                 "(VUID-VkFramebufferCreateInfo-flags-04533/04534)",
                                 "take the framebuffer extent from the attachment (swapchain) extent"]))
    return out


# ------------------------------------------------------------ uniform padding
def _covered(module, tid, off, out, deco=None, depth=0):
    m = module
    k = m.tkind(tid)
    if depth > 8:
        return
    if k == "struct":
        for i, mt in enumerate(m.types[tid].args):
            md = m.member_decos.get((tid, i), {})
            _covered(m, mt, off + md.get("Offset", [0])[0], out, md, depth + 1)
    elif k == "array":
        n = m.consts.get(m.types[tid].args[1])
        stride = m.decos.get(tid, {}).get("ArrayStride", [0])[0]
        if isinstance(n, int) and stride:
            for i in range(min(n, 4096)):
                _covered(m, m.types[tid].args[0], off + i * stride, out, deco, depth + 1)
    elif k == "matrix":
        colt, ncol = m.types[tid].args
        nrow = m.types[colt].args[1]
        ms = (deco or {}).get("MatrixStride", [16])[0]
        if probe.ROW_MAJOR in (deco or {}):
            for r in range(nrow):
                out.update(range(off + r * ms, off + r * ms + 4 * ncol))
        else:
            for c in range(ncol):
                out.update(range(off + c * ms, off + c * ms + 4 * nrow))
    elif k == "vector":
        ct, n = m.types[tid].args
        out.update(range(off, off + 4 * n))
    elif k in ("float", "int", "bool"):
        out.update(range(off, off + 4))


def block_size(module, tid):
    cov = set()
    _covered(module, tid, 0, cov)
    return (max(cov) + 1) if cov else 0, cov


def padding_writes(module, var, data, mask):
    """[(offset, float value, before member, after member)] of nonzero finite
    floats in the padding of the block of variable var."""
    m = module
    tid = m.pointee(m.defs[var].type)
    if m.tkind(tid) != "struct":
        return []
    size, cov = block_size(m, tid)
    names = []
    for i in range(len(m.types[tid].args)):
        md = m.member_decos.get((tid, i), {})
        names.append((md.get("Offset", [0])[0], m.member_names.get((tid, i), str(i))))
    out = []
    for off in range(0, min(size, len(data)) - 3, 4):
        if any(o in cov for o in range(off, off + 4)) or not all(mask[off:off + 4]):
            continue
        v = _st.unpack("<f", bytes(data[off:off + 4]))[0]
        if v != 0 and math.isfinite(v) and 1e-6 < abs(v) < 1e6:
            before = max((n for n in names if n[0] <= off), default=(0, "?"))
            after = min((n for n in names if n[0] > off), default=(size, "end"))
            out.append((off, v, before[1], after[1]))
    return out


# ------------------------------------------------------------ host pixel path
def _ramp(n):
    i = np.arange(n) % 48
    return np.where(i < 16, 10.0 ** (-3 + 2 * i / 16.0), 0.1 + 0.9 * (i - 16) / 31.0)


def _fit_row(row, ch, scale, ncols):
    n = min(ncols, len(row) // ch)
    if n < 48:
        return None
    x = _ramp(n)
    fits = []
    for k in range(min(3, ch)):
        y = row[k: n * ch: ch].astype(float) / scale
        f = transfer.fit(x, y)
        if f is None:
            return None
        fits.append(f)
    if not fits or any(f.element != fits[0].element for f in fits):
        return None
    return fits[0]


def host_pixel_flows(values, name=lambda x: x):
    """[(image, fit, sink encoding, origin, where)]: what the host does to a
    readback image on its way to an image file."""
    fills = values.readback_fills
    if not fills or not values.io:
        return []
    sizes = {r.get("path"): r.get("size") for r in values.io if r.get("call") == "fclose"}
    declared = {}
    for r in values.io:
        if r.get("call") == "png_set_gAMA":
            declared.setdefault(r.get("png"), {})["gamma"] = r.get("gamma")
        elif r.get("call") == "png_set_sRGB":
            declared.setdefault(r.get("png"), {})["srgb"] = r.get("intent")
    out = []
    for fill in fills:
        W, H = (fill.get("extent") or [0, 0])[:2]
        img = name(fill.get("image"))
        for r in values.io:
            call = r.get("call")
            fit, sink, where = None, None, None
            if call == "fwrite":
                total = sizes.get(r.get("path")) or r.get("size") or 0
                head = bytes.fromhex(r.get("head", ""))
                for ch in (4, 3):
                    hdr = total - W * H * ch
                    if not (0 <= hdr <= 4096):
                        continue
                    lo = hdr - (r.get("pos") or 0)
                    if lo < 0 or lo >= len(head):
                        continue
                    row = np.frombuffer(head[lo: lo + W * ch], dtype=np.uint8)
                    fit = _fit_row(row, ch, 255.0, W)
                    if fit is not None:
                        sink = image_io.SRGB          # 8-bit pixels in a file: display-referred
                        where = f"fwrite of 8-bit pixels to '{r.get('path')}'"
                        break
            elif call in ("png_write_row", "lodepng_encode32_file", "lodepng_encode24_file",
                          "stbi_write_png", "png_image_write_to_file"):
                data = np.frombuffer(bytes.fromhex(r.get("data", "")), dtype=np.uint8)
                if call == "png_write_row":
                    ch = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(r.get("color_type"), 4)
                    if r.get("bit_depth", 8) != 8 or ch < 3:
                        continue
                    dec = declared.get(r.get("png"), {})
                    sink = image_io.sink_encoding("png_write_row",
                                                  image_io.png_declared(dec.get("gamma"), dec.get("srgb")))
                else:
                    ch = r.get("channels") or (4 if "32" in call else 3)
                    sink = image_io.sink_encoding(call)
                fit = _fit_row(data, ch, 255.0, r.get("width") or W)
                where = f"{call} ('{r.get('path') or r.get('png')}')"
            if fit is not None and sink:
                out.append((img, fit, sink,
                            f"the host's pixel path from the readback of '{img}' to {where}: "
                            f"{fit.describe()} (fitted on the driver's readback ramp)", where))
    return out
