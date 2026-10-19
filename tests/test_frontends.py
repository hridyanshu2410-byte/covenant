"""Shader (SPIR-V) and C++ front ends on real source files."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIX = ROOT / "tests" / "fixtures"

from covenant import spirv, shader_meaning as sm, host_static as hs  # noqa: E402

HAVE_SLANG = Path(spirv.SLANGC).exists()
HAVE_GLSL = shutil.which(spirv.GLSLANG) is not None
OUT = Path(tempfile.mkdtemp(prefix="cov-"))


def slang(name, *extra):
    return spirv.compile_slang(FIX / name, OUT / name, extra)


def codes(r):
    return sorted(set(r.codes()))


@unittest.skipUnless(HAVE_SLANG, "slangc not installed")
class SlangMeaning(unittest.TestCase):
    def forward(self, albedo_format):
        m = slang("forward.slang")
        return sm.PipelineUse("forward", [(m, "vsMain"), (m, "fsMain")],
                              bindings={(0, 0): ("camera", None), (0, 1): ("lights", None),
                                        (0, 2): ("albedo_tex", albedo_format)},
                              attachments={0: ("hdr", "VK_FORMAT_R16G16B16A16_SFLOAT")})

    def test_srgb_view_is_clean(self):
        self.assertTrue(sm.check([self.forward("VK_FORMAT_R8G8B8A8_SRGB")]).ok)

    def test_unorm_view_of_srgb_texture(self):
        r = sm.check([self.forward("VK_FORMAT_R8G8B8A8_UNORM")])
        self.assertEqual(codes(r), ["meaning.color-mismatch"], str(r))

    def test_matrix_order(self):
        self.assertTrue(sm.check([sm.PipelineUse("p", [(slang("mvp_ok.slang"), "vsMain")])]).ok)
        r = sm.check([sm.PipelineUse("p", [(slang("mvp_wrong_order.slang"), "vsMain")])])
        self.assertEqual(codes(r), ["meaning.frame-mismatch"], str(r))

    def test_missing_projection(self):
        r = sm.check([sm.PipelineUse("p", [(slang("mvp_no_proj.slang"), "vsMain")])])
        self.assertIn("clip", str(r))

    def tonemap(self, swap_format, *extra):
        m = slang("tonemap.slang", *extra)
        return sm.check([sm.PipelineUse("tonemap", [(m, "fsMain")],
                                        bindings={(0, 0): ("hdr", "VK_FORMAT_R16G16B16A16_SFLOAT")},
                                        attachments={0: ("swapchain", swap_format)},
                                        presented={"swapchain"})])

    def test_display_encoding(self):
        self.assertTrue(self.tonemap("VK_FORMAT_B8G8R8A8_SRGB").ok)
        self.assertTrue(self.tonemap("VK_FORMAT_B8G8R8A8_UNORM", "-DMANUAL_ENCODE").ok)
        # encoded twice: by hand and by the sRGB swapchain
        r = self.tonemap("VK_FORMAT_B8G8R8A8_SRGB", "-DMANUAL_ENCODE")
        self.assertEqual(codes(r), ["meaning.color-mismatch"], str(r))
        # never encoded: linear values shown on an sRGB display
        r = self.tonemap("VK_FORMAT_B8G8R8A8_UNORM")
        self.assertEqual(codes(r), ["meaning.color-mismatch"], str(r))


@unittest.skipUnless(HAVE_SLANG, "slangc not installed")
class Idioms(unittest.TestCase):
    def test_data_texture_is_not_color(self):
        m = slang("packed.slang")
        r = sm.check([sm.PipelineUse("p", [(m, "fsMain")],
                                     bindings={(0, 0): ("lights", None),
                                               (0, 1): ("normals", "VK_FORMAT_R8G8B8A8_UNORM")})])
        self.assertTrue(r.ok, str(r))


@unittest.skipUnless(HAVE_SLANG, "slangc not installed")
class Vectors(unittest.TestCase):
    def test_2d_vector_carries_no_frame(self):
        # float3(p, 0) is world, float4(p * 2 - 1, z, 1) is clip: p itself is neither
        m = slang("uv_fullscreen.slang")
        r = sm.check([sm.PipelineUse("fs", [(m, "vsMain")])])
        self.assertTrue(r.ok, str(r))


@unittest.skipUnless(HAVE_SLANG, "slangc not installed")
class Premultiplied(unittest.TestCase):
    def test_alpha_applied_twice(self):
        m = slang("premul.slang")
        r = sm.check([sm.PipelineUse("p", [(m, "twice")])])
        self.assertEqual(codes(r), ["meaning.convention-mismatch"], str(r))
        # straight alpha may be multiplied in; the premultiplied layer is used as is
        self.assertTrue(sm.check([sm.PipelineUse("p", [(m, "once")])]).ok)


@unittest.skipUnless(HAVE_SLANG, "slangc not installed")
class EntryBindings(unittest.TestCase):
    def test_per_entry_reads_and_writes(self):
        m = slang("oit.slang")
        self.assertEqual(spirv.entry_bindings(m, "vsMain"), {})
        # writes through a helper function and an atomic count
        self.assertEqual(spirv.entry_bindings(m, "appendMain"), {(0, 0): True, (0, 1): True})
        # the resolve pass reads the list and never touches the counter
        self.assertEqual(spirv.entry_bindings(m, "resolveMain"), {(0, 0): False})


@unittest.skipUnless(hs.ci is not None, "needs the libclang Python bindings (clang.cindex)")
class HostConventions(unittest.TestCase):
    def test_glm_define_after_include(self):
        late = hs.parse([FIX / "glm_order.cpp"])
        self.assertEqual(set(late.clip_range.values()), {"neg_one_to_one"})
        tags = {(st, mem): t.get("depth_range") for st, mem, t, _, _ in late.flows}
        # a projection times a view matrix keeps the projection's depth range
        self.assertEqual(tags[("Camera", "proj")], "neg_one_to_one")
        self.assertEqual(tags[("Camera", "viewProj")], "neg_one_to_one")
        early = hs.parse([FIX / "glm_order.cpp"], args=["-DZO_FIRST"])
        self.assertEqual(set(early.clip_range.values()), {"zero_to_one"})

    def test_known_host_values(self):
        f = hs.parse([FIX / "glm_order.cpp"])
        self.assertEqual(f.values[("Camera", "gamma")], {2.2})
        self.assertNotIn(("Settings", "pad"), f.values)      # an array size is not a value

    def test_degrees_inferred_from_radians_call(self):
        bug = hs.parse([FIX / "fov_units.cpp"])
        self.assertEqual([(c, w.split(":")[1]) for c, w, _ in bug.angle_uses], [("std::tan", "15")])
        self.assertEqual(hs.parse([FIX / "fov_units.cpp"], args=["-DFIXED"]).angle_uses, [])


@unittest.skipUnless(HAVE_GLSL, "glslangValidator not installed")
class GlslStructure(unittest.TestCase):
    def test_varyings_paired_by_location(self):
        # Vulkan pairs by location: differing names are not a finding (the old
        # name heuristic gave 67 false alarms on real engines)
        v = spirv.compile_glsl(FIX / "swapped.vert", OUT / "glsl")
        f = spirv.compile_glsl(FIX / "swapped.frag", OUT / "glsl")
        r = sm.check([sm.PipelineUse("p", [(v, "main"), (f, "main")])])
        self.assertNotIn("structure.varying-mismatch", r.codes(), str(r))


@unittest.skipUnless(HAVE_SLANG, "slangc not installed")
class HostCpp(unittest.TestCase):
    def run_host(self, cpp):
        m = slang("lights.slang")
        facts = hs.parse([FIX / cpp])
        r = hs.check_layouts(facts, [m])
        sm.check([sm.PipelineUse("p", [(m, "fsMain")], bindings={(0, 0): ("lights", None)})],
                 hs.meaning_seeds(facts, [m]), r)
        return r

    def test_layout(self):
        self.assertTrue(self.run_host("light_pad.cpp").ok)
        self.assertEqual(codes(self.run_host("light_nopad.cpp")), ["structure.host-layout"])

    def test_centimeters_flow_into_meters(self):
        r = self.run_host("lights_cm.cpp")
        self.assertIn("meaning.unit-mismatch", r.codes(), str(r))

    def test_conversion_clears_unit(self):
        r = self.run_host("lights_cm_fixed.cpp")
        self.assertNotIn("meaning.unit-mismatch", r.codes(), str(r))


if __name__ == "__main__":
    unittest.main(verbosity=2)
