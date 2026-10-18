"""Meaning layer conventions: field-of-view units, packed colors, quantization,
winding and name matching. Each relation is tested both ways (the consistent program is
clean, the inconsistent one is reported) and every rule cites the spec or
library signature it comes from."""
import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_meaning_layer_host import HAVE_SLANG, host, slang, f32  # noqa: E402
from covenant import equivariance, host_meaning, host_static as hs, shader_meaning as sm  # noqa: E402
from covenant.diagnostics import Report  # noqa: E402
from covenant.signatures import projection  # noqa: E402


class FieldOfView(unittest.TestCase):
    def test_library_table(self):
        self.assertEqual(projection.lookup("glm_perspective", "")[:2], (0, "rad"))
        self.assertEqual(projection.lookup("mtxProj", "bx")[:2], (1, "deg"))
        self.assertIsNone(projection.lookup("mtxProj", ""))
        self.assertIsNone(projection.lookup("perspective_fov", ""))

    def test_degrees_into_half_angle(self):
        for fixed in (False, True):
            tus, files = host("fov_half_angle.cpp", fixed)
            r = Report()
            host_meaning.check(tus, files, r)
            if fixed:
                self.assertTrue(r.ok, str(r))
            else:
                self.assertEqual(r.codes(), ["meaning.unit-mismatch"] * 2)
                self.assertIn("parameter 'fovy' of make_perspective()", str(r))   # in-tree, by structure
                self.assertIn("glm_perspective", str(r))                          # library, by table
                self.assertIn("is 30", str(r))                                    # default initializer 60 / 2


@unittest.skipUnless(HAVE_SLANG, "slangc not available")
class PackedColor(unittest.TestCase):
    def run_entry(self, entry):
        m = slang("packed_color.slang")
        w = struct.pack("<I", 0x80402010)
        use = sm.PipelineUse(entry, [(m, entry)], attachments={0: ("rt", "VK_FORMAT_B8G8R8A8_SRGB")},
                             values={(0, 0): (w, b"\x01" * 4)})
        return sm.check([use])

    def test_hand_unpacked_word_keeps_encoding(self):
        self.assertIn("meaning.color-mismatch", self.run_entry("fsBug").codes())
        self.assertTrue(self.run_entry("fsFixed").ok)


@unittest.skipUnless(HAVE_SLANG, "slangc not available")
class Quantization(unittest.TestCase):
    def run_probe(self, fmt):
        m = slang("ycbcr.slang")
        use = sm.PipelineUse("video", [(m, "fsMain")], bindings={(0, 0): ("yuv", fmt)},
                             attachments={0: ("rt", "VK_FORMAT_B8G8R8A8_UNORM")})
        return equivariance.quantization_probe(use, m, "fsMain")

    def test_levels(self):
        self.assertEqual(equivariance.limited_black(8), (16 / 255, 128 / 255))
        self.assertAlmostEqual(equivariance.limited_black(10)[0], 64 / 1023)
        self.assertEqual(equivariance.unorm_bits("VK_FORMAT_R16_UNORM"), 16)
        self.assertEqual(equivariance.unorm_bits("VK_FORMAT_R10X6_UNORM_PACK16"), 10)

    def test_eight_bit_levels_on_sixteen_bit_view(self):
        self.assertIsNone(self.run_probe("VK_FORMAT_R8G8B8A8_UNORM"))
        got = self.run_probe("VK_FORMAT_R16G16B16A16_UNORM")
        self.assertIsNotNone(got)
        self.assertIn("8-bit limited-range levels", got[0])


def mat(sy):
    # column-major 4x4: scale (1, sy, 1), translate z by 0.5
    cols = [(1, 0, 0, 0), (0, sy, 0, 0), (0, 0, 1, 0), (0, 0, 0.5, 1)]
    return f32(*[x for c in cols for x in c])


@unittest.skipUnless(HAVE_SLANG, "slangc not available")
class Winding(unittest.TestCase):
    def use(self, name, sy, cull=2, pipe=7, dynamic=()):
        m = slang("winding.slang")
        ra = {"pipeline": pipe, "cull_mode": cull, "front_face_ccw": True, "dynamic": list(dynamic),
              "topology": "VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST"}
        return sm.PipelineUse(name, [(m, "vsMain")], viewport=(0, 0, 640, 480), values={(0, 0): mat(sy)},
                              vertex_input={"bindings": [], "attributes": [], "raster": ra})

    def test_mirrored_draw_through_one_pipeline(self):
        r = equivariance.winding_check([self.use("a", 1), self.use("b", -1)], Report())
        self.assertEqual(r.codes(), ["meaning.convention-mismatch"])

    def test_consistent_or_unculled_is_clean(self):
        self.assertTrue(equivariance.winding_check([self.use("a", 1), self.use("b", 2)], Report()).ok)
        self.assertTrue(equivariance.winding_check([self.use("a", 1, cull=0), self.use("b", -1, cull=0)],
                                                   Report()).ok)
        self.assertTrue(equivariance.winding_check([self.use("a", 1), self.use("b", -1, pipe=8)], Report()).ok)
        dyn = ("VK_DYNAMIC_STATE_FRONT_FACE",)
        self.assertTrue(equivariance.winding_check([self.use("a", 1, dynamic=dyn), self.use("b", -1, dynamic=dyn)],
                                                   Report()).ok)


class NameMatching(unittest.TestCase):
    def test_tokens(self):
        self.assertEqual(hs.name_tokens("uLightColor"), hs.name_tokens("light_color"))
        self.assertEqual(hs.name_tokens("mViewProj"), ("view", "proj"))


if __name__ == "__main__":
    unittest.main()
