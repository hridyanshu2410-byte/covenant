"""Fixed-function Vulkan state as a source of meaning (shader_meaning module
docstring): swapchain color space, blend state vs alpha convention, and the
clip-space Y convention combined with the viewport. Each rule is tested on
small fixtures in both directions (a consistent use is clean, a conflicting
one is reported)."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIX = ROOT / "tests" / "fixtures"

from covenant import spirv, shader_meaning as sm  # noqa: E402
from covenant.host_trace import SwapchainInfo  # noqa: E402
from covenant.program import present_encoding  # noqa: E402

HAVE_SLANG = Path(spirv.SLANGC).exists()
HAVE_GLSL = shutil.which(spirv.GLSLANG) is not None
OUT = Path(tempfile.mkdtemp(prefix="cov-ff-"))
_cache = {}


def slang(name):
    if name not in _cache:
        _cache[name] = spirv.compile_slang(FIX / name, OUT / name)
    return _cache[name]


def codes(r):
    return sorted(set(r.codes()))


def swap(color_space, fmt="VK_FORMAT_A2B10G10R10_UNORM_PACK32"):
    return SwapchainInfo("swapchain#1", "swapchain", 3, 3, 2, fmt, (640, 480), ["swap[0]"],
                         color_space)


def blend(src_c, dst_c, src_a=None, dst_a=None):
    f = lambda n: "VK_BLEND_FACTOR_" + n
    return {"enable": True, "src_color": f(src_c), "dst_color": f(dst_c), "color_op": "VK_BLEND_OP_ADD",
            "src_alpha": f(src_a or src_c), "dst_alpha": f(dst_a or dst_c), "alpha_op": "VK_BLEND_OP_ADD"}


SA_ISA = ("SRC_ALPHA", "ONE_MINUS_SRC_ALPHA")
ONE_ISA = ("ONE", "ONE_MINUS_SRC_ALPHA")


class PresentEncoding(unittest.TestCase):
    def test_color_space_table(self):
        self.assertEqual(present_encoding(swap("VK_COLOR_SPACE_HDR10_ST2084_EXT"))[0], "pq")
        self.assertEqual(present_encoding(swap("VK_COLOR_SPACE_SRGB_NONLINEAR_KHR",
                                               "VK_FORMAT_B8G8R8A8_UNORM"))[0], "srgb")
        self.assertEqual(present_encoding(swap("VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT",
                                               "VK_FORMAT_R16G16B16A16_SFLOAT"))[0], "linear")
        self.assertIsNone(present_encoding(swap("VK_COLOR_SPACE_PASS_THROUGH_EXT"))[0])
        # traces without a color space keep the sRGB-display assumption
        self.assertEqual(present_encoding(swap(None))[0], "srgb")


@unittest.skipUnless(HAVE_SLANG, "slangc not installed")
class SwapchainColorSpace(unittest.TestCase):
    def run_out(self, entry, color_space, fmt, binding_fmt="VK_FORMAT_R16G16B16A16_SFLOAT"):
        m = slang("hdr_output.slang")
        sc = swap(color_space, fmt)
        return sm.check([sm.PipelineUse("out", [(m, entry)],
                                        bindings={(0, 0): ("scene", binding_fmt),
                                                  (0, 1): ("video", "VK_FORMAT_A2B10G10R10_UNORM_PACK32")},
                                        attachments={0: ("swap[0]", fmt)},
                                        presented={"swap[0]": present_encoding(sc)})])

    HDR10 = ("VK_COLOR_SPACE_HDR10_ST2084_EXT", "VK_FORMAT_A2B10G10R10_UNORM_PACK32")
    SDR = ("VK_COLOR_SPACE_SRGB_NONLINEAR_KHR", "VK_FORMAT_B8G8R8A8_UNORM")
    SCRGB = ("VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT", "VK_FORMAT_R16G16B16A16_SFLOAT")

    def test_matching_encodings_are_clean(self):
        for entry, (cs, fmt) in (("fsPQ", self.HDR10), ("fsGamma", self.SDR), ("fsLinear", self.SCRGB)):
            r = self.run_out(entry, cs, fmt)
            self.assertTrue(r.ok, f"{entry} on {cs}:\n{r}")

    def test_wrong_encoding_for_the_color_space(self):
        for entry, (cs, fmt) in (("fsGamma", self.HDR10), ("fsLinear", self.HDR10),
                                 ("fsPQ", self.SDR), ("fsGamma", self.SCRGB), ("fsPQ", self.SCRGB)):
            r = self.run_out(entry, cs, fmt)
            self.assertEqual(codes(r), ["meaning.color-mismatch"], f"{entry} on {cs}:\n{r}")

    def test_pq_is_flagged_as_heuristic(self):
        r = self.run_out("fsPQ", *self.SDR)
        self.assertTrue(r.items and all(d.severity == "warning" for d in r.items), str(r))
        self.assertIn("ST 2084", str(r))

    def test_pq_decode(self):
        m = slang("hdr_output.slang")
        use = lambda tag: sm.PipelineUse("video", [(m, "fsVideo")],
                                         bindings={(0, 1): ("video", None)},
                                         attachments={0: ("hdr_rt", "VK_FORMAT_R16G16B16A16_SFLOAT")})
        # the decoded frame lands in a linear target: clean
        rt_linear = sm.HostSeed({"color": "linear"}, "hdr_rt is linear", resource="hdr_rt")
        self.assertTrue(sm.check([use(None)], [rt_linear]).ok)
        # PQ-decoding a texture that holds sRGB code values is a mismatch
        srgb_video = sm.HostSeed({"color": "srgb"}, "the video is sRGB", resource="video")
        r = sm.check([use(None)], [srgb_video])
        self.assertEqual(codes(r), ["meaning.color-mismatch"], str(r))


@unittest.skipUnless(HAVE_SLANG, "slangc not installed")
class BlendAlphaConvention(unittest.TestCase):
    def use(self, entry, bl, target="rt", dst="transparent", fmt="VK_FORMAT_R8G8B8A8_UNORM", name=None):
        m = slang("blend_alpha.slang")
        return sm.PipelineUse(name or entry, [(m, entry)],
                              bindings={(0, 0): ("sprite_png", None), (0, 1): ("plain_png", None),
                                        **({(0, 2): ("rt", None), (0, 3): ("rt", None)}
                                           if entry.startswith("fsLayer") else {})},
                              attachments={0: (target, fmt)},
                              blend={0: bl} if bl else {},
                              dst_alpha={0: dst} if dst else {})

    def test_source_convention_from_color_factors(self):
        self.assertTrue(sm.check([self.use("fsSprite", blend(*SA_ISA, *ONE_ISA))]).ok)
        r = sm.check([self.use("fsSprite", blend(*ONE_ISA))])          # straight source, premul blend
        self.assertEqual(codes(r), ["meaning.convention-mismatch"], str(r))

    def test_in_shader_premultiply(self):
        self.assertTrue(sm.check([self.use("fsPremul", blend(*ONE_ISA))]).ok)
        r = sm.check([self.use("fsPremul", blend(*SA_ISA, *ONE_ISA))])  # premultiplied twice
        self.assertEqual(codes(r), ["meaning.convention-mismatch"], str(r))
        self.assertTrue(all(d.severity == "warning" for d in r.items), str(r))

    def test_convention_crosses_passes(self):
        draw = self.use("fsSprite", blend(*SA_ISA, *ONE_ISA))       # rt now holds premultiplied color
        good = self.use("fsLayer", blend(*ONE_ISA), target="swap", dst="opaque")
        bad = self.use("fsLayer", blend(*SA_ISA), target="swap", dst="opaque")
        self.assertTrue(sm.check([draw, good]).ok)
        r = sm.check([draw, bad])
        self.assertEqual(codes(r), ["meaning.convention-mismatch"], str(r))
        self.assertIn("premultiplied", str(r))
        # a declared straight sampler of that target conflicts too
        r = sm.check([draw, self.use("fsLayerStraight", None, target="swap", dst="opaque")])
        self.assertEqual(codes(r), ["meaning.convention-mismatch"], str(r))

    def test_no_seed_without_known_transparent_destination(self):
        bad = self.use("fsLayer", blend(*SA_ISA), target="swap", dst="opaque")
        for dst in ("opaque", None):            # opaque clear, or loaded contents
            draw = self.use("fsSprite", blend(*SA_ISA, *ONE_ISA), dst=dst)
            self.assertTrue(sm.check([draw, bad]).ok, dst)

    def test_unblended_output_keeps_its_convention(self):
        draw = self.use("fsPremul", None)
        r = sm.check([draw, self.use("fsLayerStraight", None, target="swap", dst="opaque")])
        self.assertEqual(codes(r), ["meaning.convention-mismatch"], str(r))

    def test_alpha_squared(self):
        draw = self.use("fsSprite", blend(*SA_ISA))             # SRC_ALPHA on the alpha channel too
        reader = self.use("fsLayer", blend(*ONE_ISA), target="swap", dst="opaque")
        r = sm.check([draw, reader])
        self.assertEqual(codes(r), ["meaning.convention-mismatch"], str(r))
        self.assertIn("alpha squared", str(r))
        # never sampled afterwards: its alpha is never read
        self.assertTrue(sm.check([draw]).ok)
        # no alpha channel: nothing is stored
        self.assertTrue(sm.check([self.use("fsSprite", blend(*SA_ISA), fmt="VK_FORMAT_B10G11R11_UFLOAT_PACK32"),
                                  reader]).ok)

    def test_other_blends_say_nothing(self):
        add = blend("ONE", "ONE")
        self.assertTrue(sm.check([self.use("fsPremul", add), self.use("fsSprite", add)]).ok)


@unittest.skipUnless(HAVE_SLANG, "slangc not installed")
class ClipY(unittest.TestCase):
    UP = (0, 0, 640, 480)
    FLIP = (0, 480, 640, -480)

    def check(self, entry, vp, module=None):
        m = module or slang("yflip.slang")
        return sm.check([sm.PipelineUse("p", [(m, entry)], bindings={(0, 0): ("camera", None)},
                                        viewport=vp)])

    def test_single_flip_is_clean(self):
        for entry, vp in (("vsPlain", self.FLIP), ("vsNegate", self.UP), ("vsScale", self.UP),
                          ("vsProjVk", self.UP), ("vsProjGl", self.FLIP), ("vsConstFlip", self.UP),
                          ("vsPlain", self.UP), ("vsNegate", None)):
            r = self.check(entry, vp)
            self.assertTrue(r.ok, f"{entry} {vp}:\n{r}")

    def test_double_flip(self):
        for entry, vp in (("vsNegate", self.FLIP), ("vsScale", self.FLIP), ("vsProjVk", self.FLIP),
                          ("vsConstFlip", self.FLIP), ("vsProjVkNegate", self.UP),
                          ("vsProjVkNegate", None)):
            r = self.check(entry, vp)
            self.assertEqual(codes(r), ["meaning.convention-mismatch"], f"{entry} {vp}:\n{r}")

    @unittest.skipUnless(HAVE_GLSL, "glslangValidator not installed")
    def test_glsl_position_y_store(self):
        m = spirv.compile_glsl(FIX / "yflip.vert", OUT / "glsl")
        self.assertTrue(self.check("main", self.UP, m).ok)
        r = self.check("main", self.FLIP, m)
        self.assertEqual(codes(r), ["meaning.convention-mismatch"], str(r))


CXX = shutil.which("clang++-20") or shutil.which("g++")


@unittest.skipUnless(HAVE_SLANG and CXX, "needs slangc and a C++ compiler")
class DriverRecordsFixedFunctionState(unittest.TestCase):
    """model_driver records blend factors, static viewports and the swapchain
    color space; host_trace puts them on SwapchainInfo / DrawEvent."""

    def test_trace_fields(self):
        import os
        import subprocess
        from covenant.host_trace import load_trace
        out = OUT / "ff_state"
        out.mkdir(exist_ok=True)
        spirv.compile_slang(FIX / "yflip.slang", out)
        drv = ROOT / "model_driver" / "build"
        exe = out / "host"
        subprocess.run([CXX, "-std=c++17", "-O0", f"-I{ROOT / 'bench' / 'common'}", f"-I{ROOT / 'covenant' / 'include'}",
                        str(FIX / "ff_state.cpp"),
                        "-o", str(exe), f"-L{drv}", "-lcovvk", f"-Wl,-rpath,{drv}"], check=True)
        trace = out / "trace.jsonl"
        subprocess.run([str(exe), str(out)], check=True, timeout=60,
                       env=dict(os.environ, COVENANT_TRACE=str(trace)))
        m = load_trace(str(trace))
        cs = {s.name: s.color_space for s in m.swapchains.values()}
        self.assertEqual(cs.get("swapchain_hdr10"), "VK_COLOR_SPACE_HDR10_ST2084_EXT", cs)
        (d,) = [d for d in m.draws if m.name(d.pipeline) == "static_flip"]
        self.assertEqual(d.viewport, (0, 480, 640, -480))
        self.assertEqual(d.color_clears.get("layer_rt"), [0, 0, 0, 0])
        bl = m.pipelines[d.pipeline].record["blend"][0]
        self.assertEqual((bl["src_color"], bl["dst_color"], bl["src_alpha"], bl["dst_alpha"]),
                         ("VK_BLEND_FACTOR_SRC_ALPHA", "VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA",
                          "VK_BLEND_FACTOR_ONE", "VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA"))


if __name__ == "__main__":
    unittest.main()
