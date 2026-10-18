"""End-to-end tests of the COVENANT model driver + trace loader.

Builds model_driver/ (libcovvk.so and the triangle example) if needed, runs
the triangle renderer against the model driver (no GPU), loads the trace and
runs the timing checker. Skips when no C++ compiler or glslangValidator."""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from covenant import timing  # noqa: E402
from covenant.host_trace import load_trace, spirv_readonly_bindings  # noqa: E402

MD = ROOT / "model_driver"
BUILD = MD / "build"
LDSO = "/lib64/ld-linux-x86-64.so.2"


def _tools():
    cxx = shutil.which("clang++-20") or shutil.which("g++")
    return cxx and shutil.which("glslangValidator") and shutil.which("make")


def _build():
    r = subprocess.run(["make", "-s", "-C", str(MD), "all", "example"], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("model driver build failed:\n" + r.stdout + r.stderr)


def _run_triangle(trace, *flags, spirv_dir=None):
    exe = BUILD / "triangle"
    env = dict(os.environ, COVENANT_TRACE=str(trace))
    if spirv_dir:
        env["COVENANT_SPIRV_DIR"] = str(spirv_dir)
    args = [str(BUILD / "tri.vert.spv"), str(BUILD / "tri.frag.spv"), *flags]
    try:
        r = subprocess.run([str(exe)] + args, capture_output=True, text=True, env=env, timeout=60)
    except PermissionError:
        # the shared project folder does not keep the exec bit; run through ld.so
        r = subprocess.run([LDSO, str(exe)] + args, capture_output=True, text=True, env=env, timeout=60)
    if r.returncode:
        raise RuntimeError(f"triangle failed ({r.returncode}):\n{r.stdout}{r.stderr}")
    return r


@unittest.skipUnless(_tools(), "needs a C++ compiler, make and glslangValidator")
class ModelDriverTriangle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _build()
        cls.tmp = Path(tempfile.mkdtemp(prefix="covvk-test-"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def load(self, name, *flags):
        trace = self.tmp / f"{name}.jsonl"
        spv = self.tmp / f"{name}-spirv"
        spv.mkdir(exist_ok=True)
        _run_triangle(trace, *flags, spirv_dir=spv)
        return load_trace(str(trace), spirv_dir=str(spv))

    def test_correct_program_has_no_timing_diagnostics(self):
        m = self.load("ok")
        r = timing.check(m.frame)
        self.assertTrue(r.ok, str(r))
        self.assertEqual(m.warnings, [])

    def test_missing_fence_wait_races_on_uniform_buffer(self):
        m = self.load("bug", "--bug-no-fence-wait")
        r = timing.check(m.frame)
        races = [d for d in r.items if d.code == "timing.cross-queue-race"]
        self.assertTrue(races, str(r))
        for d in races:
            self.assertIn("ubo[", d.message)
            self.assertIn("'host'", d.message)
            self.assertIn("'graphics0'", d.message)
        # without the fence wait the command buffers are also re-recorded, and the fences
        # reset, while their earlier submissions may be pending (the in-use rules)
        self.assertIn("timing.cross-queue-race", set(r.codes()), str(r))
        self.assertLessEqual(set(r.codes()), {"timing.cross-queue-race", "timing.command-buffer-in-use",
                                              "timing.fence-in-use", "timing.acquire-semaphore-in-use"}, str(r))

    def test_host_model_contents(self):
        m = self.load("model")
        sc = m.swapchains["swapchain"]
        self.assertEqual((sc.image_count, sc.min_image_count, sc.surface_min_image_count), (3, 3, 2))
        self.assertEqual(m.frame.swapchains["swapchain"], (3, 2))
        self.assertEqual(sc.images, ["swapchain[0]", "swapchain[1]", "swapchain[2]"])
        self.assertTrue(all(m.images[i].swapchain == "swapchain" for i in m.images))
        (p,) = m.pipelines.values()
        self.assertEqual(p.color_formats, [sc.format])
        sha = hashlib.sha1((BUILD / "tri.vert.spv").read_bytes()).hexdigest()
        self.assertIn(("VK_SHADER_STAGE_VERTEX_BIT", sha, "main"), p.stages)
        self.assertEqual(p.set_layouts[0][0]["type"], "VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER")
        self.assertEqual(len(m.draws), 6)
        d = m.draws[0]
        self.assertEqual(d.descriptors[(0, 0)], ("VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER", "ubo[0]", None))
        self.assertEqual(d.attachments[0][:3], ("color", "swapchain[0]", sc.format))
        self.assertEqual(d.vertex_buffers, ["vertices"])
        self.assertEqual({q.split(":")[0] for q in m.frame.queues}, {"host", "graphics0", "present"})
        self.assertTrue(all(q.startswith("present:swapchain[") for q in m.frame.queues
                            if q.startswith("present")))
        uses = [op for op in m.frame.queues["host"] if isinstance(op, timing.Use) and op.flag == "HOST_WRITE"]
        self.assertEqual([u.res for u in uses], ["vertices"] + ["ubo[0]", "ubo[1]"] * 3)
        self.assertTrue(all((u.stage, u.flag) == ("HOST", "HOST_WRITE") for u in uses))

    def test_spirv_dump_and_reflection(self):
        self.load("spirv")
        spv = self.tmp / "spirv-spirv"
        files = sorted(spv.glob("*.spv"))
        self.assertEqual(len(files), 2)
        # tri.vert declares one uniform block at (0, 0); it is not a storage resource
        code = (BUILD / "tri.vert.spv").read_bytes()
        self.assertEqual(spirv_readonly_bindings(code), {(0, 0): False})


if __name__ == "__main__":
    unittest.main()
