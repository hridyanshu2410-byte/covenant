"""The labelled meaning layer: the group-labelled
union-find, the transfer library, the signature database, probe
fingerprinting and the witness invariants. Every rule is tested both ways: a
consistent use is clean, a conflicting one is reported."""
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIX = ROOT / "tests" / "fixtures"

from covenant import labels, probe, spirv, transfer, witness, shader_meaning as sm  # noqa: E402
from covenant.diagnostics import Report  # noqa: E402
from covenant.meaning_solver import MeaningSolver  # noqa: E402
from covenant.program import _depth_direction  # noqa: E402
from covenant.signatures import (SRGB_DEC, SRGB_ENC, PQ_ENC, encoding_element, glm, glsl_std450,  # noqa: E402
                                 image_io, std_trig, vulkan)

HAVE_SLANG = Path(spirv.SLANGC).exists()
OUT = Path(tempfile.mkdtemp(prefix="cov-meaning-labels-"))
_cache = {}


def slang(name):
    if name not in _cache:
        _cache[name] = spirv.compile_slang(FIX / name, OUT / name)
    return _cache[name]


def fid_of(m, entry):
    return next(f for _, f, n, _ in m.entries if n == entry)


def codes(r):
    return sorted(set(r.codes()))


# ------------------------------------------------------------------ groups / solver
class Groups(unittest.TestCase):
    def test_free_group_reduces(self):
        G = labels.FreeGroup()
        self.assertEqual(G.mul(SRGB_ENC, SRGB_DEC), ())
        self.assertEqual(G.mul(SRGB_DEC, SRGB_DEC), (("srgb", -2),))
        self.assertEqual(G.inv((("srgb", 1), ("pq", -1))), (("pq", 1), ("srgb", -1)))
        self.assertNotEqual(G.mul(SRGB_DEC, PQ_ENC), G.mul(PQ_ENC, SRGB_DEC))     # not abelian

    def test_unit_group(self):
        U = labels.UnitGroup()
        cm, m = U.of("cm"), U.of("m")
        self.assertFalse(U.is_id(U.mul(cm, U.inv(m))))
        self.assertTrue(U.is_id(U.mul(cm, U.inv(cm))))
        self.assertEqual(U.fmt(cm), "cm")

    def test_consistent_cycle_is_clean(self):
        uf = labels.LabelledUF("color", labels.FreeGroup())
        a = uf.anchor("linear")
        uf.add(a, "img", SRGB_ENC, "srgb texture")
        uf.add("img", "texel", SRGB_DEC, "hardware decode")
        self.assertIsNone(uf.add(a, "texel", (), "texel is linear"))
        self.assertEqual(uf.conflicts, [])

    def test_inconsistent_cycle_reports_both_paths(self):
        uf = labels.LabelledUF("color", labels.FreeGroup())
        a = uf.anchor("linear")
        uf.add(a, "img", SRGB_ENC, "srgb texture")
        uf.add("img", "texel", SRGB_DEC, "hardware decode")
        uf.add("texel", "out", SRGB_DEC, "manual decode")
        c = uf.add(a, "out", (), "target is linear")
        self.assertIsNotNone(c)
        self.assertEqual(c.residual, (("srgb", 1),))
        self.assertIn("manual decode", [e.origin for e in c.edges()])

    def test_equality_dimension_anchors(self):
        uf = labels.LabelledUF("frame", labels.Trivial(), equality=True)
        uf.add(uf.anchor("world"), "p", ())
        uf.add(uf.anchor("world"), "q", ())
        self.assertIsNone(uf.add("p", "q", ()))
        uf.add(uf.anchor("view"), "r", ())
        self.assertIsNotNone(uf.add("p", "r", ()))

    def test_domain_constraint(self):
        r = Report()
        s = MeaningSolver(r)
        s.require_domain("conv.alpha", {0, 1}, "Porter-Duff once")
        s.set("tex", "conv.alpha", "premultiplied", "a premultiplied texture", "w")
        s.edge("tex", "out", "conv.alpha", 1, "rgb * a", "w")
        s.finish()
        self.assertEqual(codes(r), ["meaning.convention-mismatch"])
        r = Report()
        s = MeaningSolver(r)
        s.require_domain("conv.alpha", {0, 1}, "Porter-Duff once")
        s.set("tex", "conv.alpha", "straight", "a straight texture", "w")
        s.edge("tex", "out", "conv.alpha", 1, "rgb * a", "w")
        s.finish()
        self.assertTrue(r.ok, str(r))

    def test_weak_edges_give_warnings(self):
        r = Report()
        s = MeaningSolver(r)
        s.set("a", "color", "linear", "x", "w")
        s.edge("a", "b", "color", SRGB_ENC, "a fit", "w", weak=True)
        s.set("b", "color", "linear", "y", "w")
        s.finish()
        self.assertTrue(r.items and all(d.severity == "warning" for d in r.items), str(r))


# ------------------------------------------------------------------ transfer library
class TransferFit(unittest.TestCase):
    x = transfer.RAMP

    def test_library_curves_classify(self):
        for fn, cls, el in ((transfer.srgb_enc, "srgb", SRGB_ENC), (transfer.srgb_dec, "srgb", SRGB_DEC),
                            (lambda v: v ** (1 / 2.2), "srgb", SRGB_ENC), (lambda v: v ** 2.2, "srgb", SRGB_DEC),
                            (lambda v: transfer.pq_enc(v * 0.02), "pq", PQ_ENC),
                            (transfer.hlg_enc, "hlg", (("hlg", 1),)),
                            (transfer.reinhard, "tonemap", ()), (lambda v: 3 * v + 0.1, "affine", ())):
            f = transfer.fit(self.x, fn(self.x))
            self.assertIsNotNone(f, cls)
            self.assertEqual((f.cls, f.element), (cls, el))

    def test_ambiguous_curves_stay_opaque(self):
        for y in (self.x ** 3.7, self.x ** 1.8, np.sin(self.x * 7), transfer.pq_enc(self.x * 1e-4)):
            self.assertIsNone(transfer.fit(self.x, y))

    def test_affine_remap(self):
        np.testing.assert_allclose(transfer.fit_affine([0, .5, 1], [-1, 0, 1]), (2.0, -1.0), atol=1e-9)
        self.assertIsNone(transfer.fit_affine([0, .5, 1], [0, .25, 1]))


# ------------------------------------------------------------------ signatures
class Signatures(unittest.TestCase):
    def test_vulkan_formats_and_color_spaces(self):
        self.assertEqual(vulkan.format_transfer("VK_FORMAT_R8G8B8A8_SRGB"), "srgb")
        self.assertIsNone(vulkan.format_transfer("VK_FORMAT_R8G8B8A8_UNORM"))
        self.assertTrue(vulkan.is_depth_format("VK_FORMAT_D32_SFLOAT"))
        self.assertFalse(vulkan.is_depth_format("VK_FORMAT_R32_SFLOAT"))
        self.assertEqual(vulkan.COLOR_SPACE_ENCODING["VK_COLOR_SPACE_HDR10_ST2084_EXT"], "pq")
        self.assertEqual(vulkan.COLOR_SPACE_ENCODING["VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT"], "linear")

    def test_vulkan_blend(self):
        bl = lambda sc, dc, sa, da: {"enable": True, "src_color": vulkan.BF + sc, "dst_color": vulkan.BF + dc,
                                     "src_alpha": vulkan.BF + sa, "dst_alpha": vulkan.BF + da}
        self.assertEqual(vulkan.color_blend(bl("SRC_ALPHA", "ONE_MINUS_SRC_ALPHA", "ONE", "ONE")), (0, 1))
        self.assertEqual(vulkan.color_blend(bl("ONE", "ONE_MINUS_SRC_ALPHA", "ONE", "ONE")), (1, 0))
        self.assertIsNone(vulkan.color_blend(bl("ONE", "ONE", "ONE", "ONE")))
        self.assertEqual(vulkan.alpha_blend(bl("ONE", "ONE", "SRC_ALPHA", "ONE_MINUS_SRC_ALPHA")), 1)
        self.assertEqual(vulkan.alpha_blend(bl("ONE", "ONE", "ONE", "ONE_MINUS_SRC_ALPHA")), 0)
        self.assertIsNone(vulkan.color_blend({"enable": False}))

    def test_vulkan_viewport_and_depth(self):
        self.assertTrue(vulkan.viewport_flips((0, 480, 640, -480)))
        self.assertFalse(vulkan.viewport_flips((0, 0, 640, 480)))
        self.assertEqual(vulkan.COMPARE_DIR["VK_COMPARE_OP_GREATER"], 1)
        self.assertEqual(vulkan.COMPARE_DIR["VK_COMPARE_OP_LESS_OR_EQUAL"], 0)
        self.assertEqual((vulkan.clear_dir(1.0), vulkan.clear_dir(0.0), vulkan.clear_dir(0.5)), (0, 1, None))

    def test_glsl_std450(self):
        self.assertIn("FMix", glsl_std450.COLOR_SAME)
        self.assertIn("Pow", glsl_std450.NONLINEAR)
        self.assertNotIn("Pow", glsl_std450.COLOR_SAME)
        self.assertEqual(glsl_std450.ANGLE_CONVERT[glsl_std450.RADIANS], -1)

    def test_glm(self):
        self.assertEqual(glm.depth_range("perspective", zo_defined=False), 1)
        self.assertEqual(glm.depth_range("perspective", zo_defined=True), 0)
        self.assertEqual(glm.depth_range("perspectiveRH_ZO", zo_defined=False), 0)
        self.assertEqual(glm.depth_range("perspectiveRH_NO", zo_defined=True), 1)
        self.assertTrue(glm.is_projection("glm::perspectiveRH_ZO"))
        self.assertFalse(glm.is_projection("lookAt"))

    def test_std_trig(self):
        self.assertIn("tan", std_trig.ANGLE_IN)
        self.assertEqual(std_trig.scale_element(np.pi / 180), -1)
        self.assertEqual(std_trig.scale_element(180 / np.pi), 1)
        self.assertEqual(std_trig.scale_element(0.5), 0)

    def test_image_io(self):
        self.assertEqual(image_io.sink_encoding("stbi_write_png"), "srgb")
        self.assertEqual(image_io.sink_encoding("SaveEXR"), "linear")
        self.assertEqual(image_io.sink_encoding("stbi_write_hdr"), "linear")
        self.assertEqual(image_io.png_declared(gamma=1.0), "linear")
        self.assertEqual(image_io.png_declared(gamma=1 / 2.2), "srgb")
        self.assertIsNone(image_io.png_declared())
        self.assertEqual(image_io.sink_encoding("png_write_row", image_io.png_declared(gamma=1.0)), "linear")
        self.assertEqual(image_io.SOURCES["stbi_loadf"], "linear")

    def test_encoding_elements(self):
        self.assertEqual(encoding_element("srgb"), SRGB_ENC)
        self.assertEqual(encoding_element("linear"), ())
        self.assertIsNone(encoding_element("data"))
        self.assertEqual(encoding_element("rgbm"), (("rgbm", 1),))


# ------------------------------------------------------------------ probe
@unittest.skipUnless(HAVE_SLANG, "slangc not installed")
class Fingerprint(unittest.TestCase):
    def labels_of(self, entry, push=None):
        m = slang("probe_curves.slang")
        blocks = {}
        if push is not None:
            pv = next(v for v, d in m.defs.items() if d.op == "OpVariable" and d.args and d.args[0] == 9)
            blocks[pv] = (push, b"\x01" * len(push))
        fp = probe.fingerprint(m, fid_of(m, entry), probe.Memory(m, blocks))
        self.assertIsNone(fp.error)
        return fp, sorted({f.name for _, f in fp.edges.values() if f.element})

    def test_select_and_lerp_decodes(self):
        for entry in ("fsSelect", "fsLerp"):
            fp, names = self.labels_of(entry)
            self.assertIn("sRGB decode", names, entry)
            self.assertFalse([n for n in names if "encode" in n], entry)

    def test_decode_then_pq_with_concrete_push_constants(self):
        _, names = self.labels_of("fsDecodeThenPQ", struct.pack("<4f", 203.0, 0, 0, 0))
        self.assertIn("sRGB decode", names)
        self.assertIn("PQ (ST 2084) encode", names)

    def test_gamma_encode(self):
        _, names = self.labels_of("fsGamma22")
        self.assertEqual(names, ["gamma 2.2 encode"])

    def test_unknown_curves_are_opaque(self):
        for entry in ("fsOdd", "fsSmooth"):
            fp, names = self.labels_of(entry)
            self.assertEqual(names, [], entry)
            self.assertTrue(fp.opaque, entry)

    def test_lerp_decode_is_clean_in_the_solver(self):
        m = slang("probe_curves.slang")
        use = lambda entry, fmt: sm.PipelineUse(entry, [(m, entry)], bindings={(0, 0): ("photo", "VK_FORMAT_R8G8B8A8_UNORM")},
                                                attachments={0: ("rt", fmt)})
        seed = [sm.HostSeed({"color": "srgb"}, "the photo is sRGB", resource="photo"),
                sm.HostSeed({"color": "linear"}, "rt is linear", resource="rt")]
        self.assertTrue(sm.check([use("fsLerp", "VK_FORMAT_R16G16B16A16_SFLOAT")], seed).ok)
        r = sm.check([use("fsGamma22", "VK_FORMAT_R16G16B16A16_SFLOAT")], seed)   # encodes again
        self.assertEqual(codes(r), ["meaning.color-mismatch"], str(r))
        self.assertTrue(all(d.severity == "warning" for d in r.items), str(r))


# ------------------------------------------------------------------ witnesses
@unittest.skipUnless(HAVE_SLANG, "slangc not installed")
class Witnesses(unittest.TestCase):
    def test_color_matrix(self):
        m = slang("color_matrix.slang")
        got = {}
        for e in ("fsRight", "fsTransposed", "fsNotColor"):
            fp = probe.fingerprint(m, fid_of(m, e))
            got[e] = witness.color_matrix(m, fp, lambda d: "here")
        self.assertEqual(got["fsRight"], [])
        self.assertEqual(len(got["fsTransposed"]), 1)
        self.assertIn("transposed", got["fsTransposed"][0][1])
        self.assertEqual(got["fsNotColor"], [])
        r = sm.check([sm.PipelineUse("p", [(m, "fsTransposed")], bindings={(0, 0): ("scene", None)})])
        self.assertEqual(codes(r), ["meaning.convention-mismatch"], str(r))

    def _proj(self, P):
        m = slang("projection.slang")
        var = next(v for v, d in m.defs.items() if d.op == "OpVariable" and m.decos.get(v, {}).get("Binding") == [0])
        data = np.asarray(P, dtype="<f4").T.tobytes()          # column-major
        return witness.projection(m, fid_of(m, "vsMain"), probe.Memory(m, {var: (data, b"\x01" * 64)}))

    @staticmethod
    def persp(n=0.1, f=100.0, zo=True, reverse=False):
        P = np.zeros((4, 4))
        P[0, 0] = P[1, 1] = 1.0
        if zo and not reverse:
            P[2, 2], P[2, 3] = f / (n - f), n * f / (n - f)
        elif zo:
            P[2, 2], P[2, 3] = n / (f - n), n * f / (f - n)
        else:
            P[2, 2], P[2, 3] = (f + n) / (n - f), 2 * f * n / (n - f)
        P[3, 2] = -1
        return P

    def test_projection_forward_and_reverse(self):
        fw = self._proj(self.persp())
        self.assertIsNotNone(fw)
        self.assertEqual(fw["direction"], 0)
        self.assertIsNone(witness.projection_violation(fw))
        rz = self._proj(self.persp(reverse=True))
        self.assertEqual(rz["direction"], 1)
        self.assertIsNone(witness.projection_violation(rz))

    def test_projection_transposed_remap(self):
        gl_to_vk = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, .5, .5], [0, 0, 0, 1]])
        good = self._proj(gl_to_vk @ self.persp(zo=False))
        self.assertIsNone(witness.projection_violation(good))
        bad = self._proj(gl_to_vk.T @ self.persp(zo=False))
        self.assertIsNotNone(witness.projection_violation(bad))

    def test_nearer_wins(self):
        d = SimpleNamespace(seq=1, depth_compare="VK_COMPARE_OP_LESS", depth_clear=1.0, depth_test=True)
        rz = self._proj(self.persp(reverse=True))
        fw = self._proj(self.persp())
        r = Report()
        _depth_direction(None, d, "draw", "w", r, fw)
        self.assertTrue(r.ok, str(r))
        r = Report()
        _depth_direction(None, d, "draw", "w", r, rz)
        self.assertEqual(codes(r), ["meaning.convention-mismatch"], str(r))

    def test_padding(self):
        m = slang("padding.slang")
        var = next(v for v, d in m.defs.items() if d.op == "OpVariable" and m.decos.get(v, {}).get("Binding") == [0])
        clean = struct.pack("<f3f4f", 2.0, 0, 0, 0, 1, 2, 3, 4)
        bad = struct.pack("<f3f4f", 2.0, 1.0, 0, 0, 1, 2, 3, 4)
        self.assertEqual(witness.padding_writes(m, var, clean, b"\x01" * 32), [])
        hits = witness.padding_writes(m, var, bad, b"\x01" * 32)
        self.assertEqual([h[0] for h in hits], [4])

    def test_depth_range_dimension(self):
        m = slang("depth_unproject.slang")
        seed = [sm.HostSeed({"depth_range": "zero_to_one"}, "glm::perspectiveRH_ZO", struct="View", member="invProj")]
        use = lambda e: sm.PipelineUse(e, [(m, e)], bindings={(0, 0): ("view", None),
                                                             (0, 1): ("depth", "VK_FORMAT_D32_SFLOAT")})
        self.assertTrue(sm.check([use("fsGood")], seed).ok)
        r = sm.check([use("fsBad")], seed)
        self.assertEqual(codes(r), ["meaning.convention-mismatch"], str(r))


class Extents(unittest.TestCase):
    def recs(self, fb_h):
        return [{"seq": 1, "fn": "vkCreateSwapchainKHR", "extent": [875, 700], "images": ["image#1"]},
                {"seq": 2, "fn": "vkCreateImageView", "view": "image_view#1", "image": "image#1"},
                {"seq": 3, "fn": "vkCreateFramebuffer", "framebuffer": "fb#1", "attachments": ["image_view#1"],
                 "extent": [875, fb_h, 1]}]

    def test_framebuffer_extent(self):
        self.assertEqual(witness.extents(self.recs(700)), [])
        self.assertEqual(len(witness.extents(self.recs(875))), 1)


class HostPixelPath(unittest.TestCase):
    def values(self, encode):
        W, H = 96, 2
        x = witness._ramp(W)
        y = transfer.srgb_enc(x) if encode else x
        row = np.repeat(np.round(np.clip(y, 0, 1) * 255).astype(np.uint8)[:, None], 4, axis=1)
        row[:, 3] = 255
        data = np.tile(row.reshape(-1), H).tobytes()
        return SimpleNamespace(
            readback_fills=[{"image": "viewport", "extent": [W, H]}],
            io=[{"call": "fwrite", "path": "shot.rgba", "pos": 0, "size": len(data), "head": data.hex()},
                {"call": "fclose", "path": "shot.rgba", "size": len(data)}])

    def test_fit_of_the_host_path(self):
        lin = witness.host_pixel_flows(self.values(False))
        enc = witness.host_pixel_flows(self.values(True))
        self.assertEqual(len(lin), 1)
        self.assertEqual(lin[0][1].element, ())
        self.assertEqual(enc[0][1].element, SRGB_ENC)
        self.assertEqual(lin[0][2], "srgb")

    def test_flow_into_the_solver(self):
        m = slang("tonemap.slang") if HAVE_SLANG else None
        if m is None:
            self.skipTest("slangc not installed")
        seeds = [sm.HostSeed({"color": "linear"}, "viewport holds linear light", resource="viewport")]
        r = sm.check([], seeds, None, None, witness.host_pixel_flows(self.values(False)))
        self.assertEqual(codes(r), ["meaning.color-mismatch"], str(r))
        r = sm.check([], seeds, None, None, witness.host_pixel_flows(self.values(True)))
        self.assertTrue(r.ok, str(r))


@unittest.skipUnless(shutil.which(spirv.GLSLANG), "glslangValidator not installed")
class FlowSensitiveLocals(unittest.TestCase):
    def setUp(self):
        self.m = spirv.compile_glsl(FIX / "inplace_encode.frag", OUT / "glsl")

    def use(self, fmt):
        return sm.PipelineUse("p", [(self.m, "main")], bindings={(0, 0): ("hdr", None)},
                              attachments={0: ("swap", fmt)}, presented={"swap"})

    def test_in_place_encode_is_one_value_per_store(self):
        seeds = [sm.HostSeed({"color": "linear"}, "hdr holds linear light", resource="hdr")]
        r = sm.check([self.use("VK_FORMAT_B8G8R8A8_UNORM")], seeds)
        self.assertTrue(r.ok, str(r))
        r = sm.check([self.use("VK_FORMAT_B8G8R8A8_SRGB")], seeds)   # encoded twice
        self.assertEqual(codes(r), ["meaning.color-mismatch"], str(r))

    def test_unused_binding_is_not_required(self):
        from covenant.program import used_bindings
        self.assertEqual(set(used_bindings(self.m, "main")), {(0, 0)})


if __name__ == "__main__":
    unittest.main()
