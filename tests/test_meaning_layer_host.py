"""Meaning layer on the host side: whole-host
convention typing (host_meaning), the host -> GPU value-provenance join, and
the gauge-equivariance probes (equivariance). Each relation is tested both
ways: the consistent program is clean, the inconsistent one is reported with
its witness."""
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import clang.cindex as ci

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIX = ROOT / "tests" / "fixtures"

from covenant import equivariance, host_meaning, host_static as hs, spirv, shader_meaning as sm  # noqa: E402
from covenant.diagnostics import Report  # noqa: E402

HAVE_SLANG = Path(spirv.SLANGC).exists()
OUT = Path(tempfile.mkdtemp(prefix="cov-meaning-host-"))
_cache = {}


def slang(name):
    if name not in _cache:
        _cache[name] = spirv.compile_slang(FIX / name, OUT / name)
    return _cache[name]


def host(name, fixed):
    f = FIX / name
    idx = ci.Index.create()
    args = ["-x", "c++", "-std=c++20", f"-I{hs.INCLUDE}"] + [f"-isystem{i}" for i in hs.CLANG_INC]
    if fixed:
        args.append("-DFIXED")
    return [idx.parse(str(f), args=args + [f"-I{f.parent}"])], [f]


def f32(*xs):
    b = struct.pack(f"<{len(xs)}f", *xs)
    return (b, b"\x01" * len(b))


class AngleTyping(unittest.TestCase):
    def test_scale_elements(self):
        self.assertEqual(host_meaning.angle_element(3.14159265 / 180), -1)
        self.assertEqual(host_meaning.angle_element(180 / 3.14159265), 1)
        self.assertEqual(host_meaning.angle_element(0.5 * 3.14159265 / 180), -1)
        self.assertEqual(host_meaning.angle_element(0.5), 0)
        self.assertEqual(host_meaning.angle_element(1 / 255.0), 0)

    def test_degrees_through_helper_reach_cos(self):
        for fixed in (False, True):
            tus, files = host("angles.cpp", fixed)
            r = Report()
            host_meaning.check(tus, files, r)
            if fixed:
                self.assertTrue(r.ok, str(r))      # also: the polymorphic helper is not a conflict
            else:
                self.assertIn("meaning.unit-mismatch", r.codes())
                self.assertIn("passed as parameter 'angle' of falloff()", str(r))


class ValueJoin(unittest.TestCase):
    def _gltf_use(self):
        m = slang("gltf.slang")
        bytes_ = f32(7 / 255, 29 / 255, 9 / 255, 1.0)
        return m, sm.PipelineUse("pbr", [(m, "fsMain")], values={(0, 0): bytes_})

    def test_host_values_are_folded(self):
        tus, files = host("gltf.cpp", False)
        hm = host_meaning.check(tus, files, Report())
        got = set()
        for n, vals in hm.vals.items():
            if n[0] == "v" and hm.names.get(n) == "u" and vals:
                got |= vals
        self.assertTrue(any(abs(x - 7 / 255) < 1e-9 for x in got), got)

    @unittest.skipUnless(HAVE_SLANG, "slangc not available")
    def test_linear_host_value_into_srgb_field(self):
        m, use = self._gltf_use()
        tus, files = host("gltf.cpp", False)
        hm = host_meaning.check(tus, files, Report())
        seeds = host_meaning.gpu_seeds(hm, [use])
        self.assertEqual([(s.struct, s.member, s.tag) for s in seeds],
                         [("Uniforms", "colDiffuse", {"color": "linear"})])
        r = sm.check([use], seeds)
        self.assertIn("meaning.color-mismatch", r.codes())
        # the encoded host values do not match these bytes: no seed, no report
        tus, files = host("gltf.cpp", True)
        hm = host_meaning.check(tus, files, Report())
        self.assertEqual(host_meaning.gpu_seeds(hm, [use]), [])

    def test_plain_numbers_never_join(self):
        self.assertFalse(host_meaning._distinctive(1.0))
        self.assertFalse(host_meaning._distinctive(0.5))
        self.assertTrue(host_meaning._distinctive(7 / 255))


PREMUL = {"enable": True, "src_color": "VK_BLEND_FACTOR_ONE", "dst_color": "VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA",
          "color_op": "VK_BLEND_OP_ADD", "src_alpha": "VK_BLEND_FACTOR_ONE",
          "dst_alpha": "VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA", "alpha_op": "VK_BLEND_OP_ADD"}
STRAIGHT = dict(PREMUL, src_color="VK_BLEND_FACTOR_SRC_ALPHA")


@unittest.skipUnless(HAVE_SLANG, "slangc not available")
class Coverage(unittest.TestCase):
    def run_probe(self, entry, blend):
        m = slang("coverage.slang")
        use = sm.PipelineUse(entry, [(m, entry)], attachments={0: ("rt", "VK_FORMAT_B8G8R8A8_UNORM")},
                             blend={0: blend}, values={(0, 0): f32(0.6, 0.4, 0.2, 0.3)})
        return equivariance.coverage_probe(use, m, entry, 0, blend)

    def test_encode_after_premultiply_is_fractional(self):
        got = self.run_probe("encodeAfterPremultiply", PREMUL)
        self.assertIsNotNone(got)
        self.assertIn("degree 0.455", got[0])

    def test_encode_then_premultiply_is_clean(self):
        self.assertIsNone(self.run_probe("encodeThenPremultiply", PREMUL))

    def test_straight_blend(self):
        self.assertIsNone(self.run_probe("straightOut", STRAIGHT))
        # straight-looking color into a premultiplied blend: maybe a premultiplied source, or the
        # additive/over mixing idiom; not a convention fact, so silent
        self.assertIsNone(self.run_probe("straightOut", PREMUL))
        got = self.run_probe("encodeThenPremultiply", STRAIGHT)    # premultiplied into SRC_ALPHA
        self.assertIsNotNone(got)
        self.assertIn("premultiplied", got[0])


@unittest.skipUnless(HAVE_SLANG, "slangc not available")
class SdrWhite(unittest.TestCase):
    def run_probe(self, entry, scale, space=equivariance.SCRGB):
        m = slang("sdr_white.slang")
        use = sm.PipelineUse(entry, [(m, entry)],
                             bindings={(0, 1): ("slide", "VK_FORMAT_R8G8B8A8_SRGB")},
                             attachments={0: ("swap", "VK_FORMAT_R16G16B16A16_SFLOAT")},
                             values={(0, 0): f32(scale, 0, 0, 0)}, color_spaces={"swap": space})
        r = Report()
        equivariance.check([use], r)
        return r

    def test_unscaled_sdr_on_scrgb(self):
        r = self.run_probe("composite", 1.0)
        self.assertEqual(r.codes(), ["meaning.color-mismatch"])
        self.assertIn("80 cd/m2", str(r))

    def test_scaled_sdr_is_clean(self):
        self.assertTrue(self.run_probe("composite", 203 / 80).ok)

    def test_not_a_display_mapping(self):
        self.assertTrue(self.run_probe("lit", 1.0).ok)

    def test_only_scrgb(self):
        self.assertTrue(self.run_probe("composite", 1.0, "VK_COLOR_SPACE_SRGB_NONLINEAR_KHR").ok)


@unittest.skipUnless(HAVE_SLANG, "slangc not available")
class Orientation(unittest.TestCase):
    def run_probe(self, top, bottom):
        m = slang("ui.slang")
        L, R = 0.0, 640.0
        ortho = [2 / (R - L), 0, 0, 0, 0, 2 / (top - bottom), 0, 0, 0, 0, -1, 0,
                 (R + L) / (L - R), (top + bottom) / (bottom - top), 0, 1]
        use = sm.PipelineUse("imgui", [(m, "vsMain")], values={(0, 0): f32(*ortho)},
                             viewport=(0, 0, 640, 480),
                             vertex_input={"bindings": [{"binding": 0, "stride": 20}],
                                           "attributes": [{"location": 0, "binding": 0, "offset": 0},
                                                          {"location": 1, "binding": 0, "offset": 8},
                                                          {"location": 2, "binding": 0, "offset": 16}]})
        structs = {"ImDrawVert": SimpleNamespace(size=20, fields=[SimpleNamespace(name="pos", offset=0),
                                                                  SimpleNamespace(name="uv", offset=8),
                                                                  SimpleNamespace(name="col", offset=16)])}
        r = Report()
        equivariance.check([use], r, structs)
        return r

    def test_lower_left_ortho_flips_ui(self):
        r = self.run_probe(0.0, 480.0)          # OpenGL-style: top -> clip +1
        self.assertEqual(r.codes(), ["meaning.convention-mismatch"])
        self.assertIn("upside down", str(r))

    def test_upper_left_ortho_is_clean(self):
        self.assertTrue(self.run_probe(480.0, 0.0).ok)


HAVE_GLSLANG = shutil.which("glslangValidator") is not None


class HostFrontEnd(unittest.TestCase):
    """host_static robustness: builtin headers match
    the loaded libclang; a parse failure is a visible warning, not silence;
    a vector-typed buffer size is matched against the shader array stride."""

    def test_resource_dir_matches_libclang(self):
        ver = hs._libclang_major()
        for d in hs.CLANG_INC:
            self.assertLessEqual(int(Path(d).parent.name.split(".")[0]), ver or 999)

    def test_parse_failure_is_reported(self):
        from covenant import program
        from covenant.host_trace import load_trace
        bad, empty = OUT / "broken.cpp", OUT / "empty.jsonl"
        bad.write_text("struct S { int x; ;\n")
        empty.write_text("")
        r = Report()
        program.check_program(load_trace(str(empty)), [], [bad], [], r)
        self.assertEqual(r.codes(), ["host.parse-failed"])
        self.assertTrue(r.meta["host_static"].startswith("failed"))

    @unittest.skipUnless(HAVE_GLSLANG, "glslangValidator not available")
    def test_vector_sizeof_against_stride(self):
        spv = OUT / "vecsize.spv"
        subprocess.run(["glslangValidator", "-V", str(FIX / "vecsize.frag"), "-o", str(spv)],
                       check=True, capture_output=True)
        m = spirv.parse(spv)
        m.meta = spirv._empty_seeds()
        self.assertEqual(hs.shader_vector_arrays([m]), [("FragmentBuffer", "data", 3, "uint", 16)])
        for fixed in (False, True):
            facts = hs.parse([FIX / "vecsize.cpp"], ["-DFIXED"] if fixed else [])
            r = hs.check_layouts(facts, [m])
            if fixed:
                self.assertTrue(r.ok, str(r))
            else:
                self.assertEqual(r.codes(), ["structure.host-layout"])
                self.assertIn("25% too small", str(r))


if __name__ == "__main__":
    unittest.main()
