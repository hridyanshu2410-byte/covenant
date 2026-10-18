import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from covenant import check_all, structural, timing  # noqa: E402
from covenant.timing import Frame, Use, Barrier, Loop, Acquire, Present  # noqa: E402
from forward import build  # noqa: E402


def run(bugs=()):
    p = build(bugs)
    return check_all(p["pipelines"], p["host_writes"], p["frame"], p["commands"],
                     p["layouts"])


class SeededBugs(unittest.TestCase):
    def test_clean_program_passes(self):
        r = run()
        self.assertTrue(r.ok, str(r))

    def expect(self, bug, code):
        r = run([bug])
        self.assertIn(code, r.codes(), str(r))

    def test_srgb_never_decoded(self):        self.expect("srgb", "meaning.color-mismatch")
    def test_world_vs_view(self):             self.expect("world_view", "meaning.frame-mismatch")
    def test_double_srgb_encode(self):        self.expect("double_encode", "meaning.color-mismatch")
    def test_centimeters_vs_meters(self):     self.expect("units_cm", "meaning.unit-mismatch")
    def test_host_struct_padding(self):       self.expect("host_layout", "structure.host-layout")
    def test_draw_before_bind(self):          self.expect("missing_bind", "structure.missing-binding")
    def test_async_compute_race(self):        self.expect("async_race", "timing.cross-queue-race")
    def test_frames_in_flight_race(self):     self.expect("in_flight", "timing.cross-queue-race")
    def test_missing_render_to_sample(self):  self.expect("no_barrier", "timing.wrong-layout")

    def test_each_bug_reported_by_one_checker_only(self):
        for bug, checker in [("srgb", "meaning"), ("world_view", "meaning"),
                             ("units_cm", "meaning"), ("host_layout", "structure"),
                             ("missing_bind", "structure"), ("async_race", "timing"),
                             ("in_flight", "timing")]:
            r = run([bug])
            self.assertEqual({d.checker for d in r.items}, {checker}, bug)


class NameBasedIO(unittest.TestCase):
    def test_reordered_varyings(self):
        fwd = build()["pipelines"][0]
        # RCGP would pair outputs and inputs by position: swapped, silently
        self.assertEqual(structural.positional_pairs(fwd)[:2],
                         [("world_pos", "normal"), ("normal", "world_pos")])
        self.assertTrue(structural.check_io(fwd, positional=True).ok)
        # COVENANT pairs by name, so the program is simply correct
        self.assertTrue(structural.check_io(fwd).ok)


class Loops(unittest.TestCase):
    def ping_pong(self, close):
        body = [Barrier("blur", "shader_read", "general", "to storage"),
                Use("blur", "write", "general", "blur pass")]
        if close:
            body += [Barrier("blur", "general", "shader_read", "to sampled"),
                     Use("blur", "read", "shader_read", "sample blur")]
        f = Frame(initial_layouts={"blur": "shader_read"})
        f.q("graphics").append(Loop(body, "for each blur iteration"))
        return f

    def test_barrier_loop_accepted_when_fixed_point(self):
        f = self.ping_pong(close=True)
        self.assertTrue(timing.check(f).ok)
        self.assertTrue(timing.rcgp_would_reject(f))   # RCGP's foreach rule

    def test_loop_that_drifts_is_rejected(self):
        r = timing.check(self.ping_pong(close=False))
        self.assertIn("timing.loop-not-fixed-point", r.codes(), str(r))


class Swapchain(unittest.TestCase):
    """Modeled on a freeze of RCGP's samples on a desktop GPU (bench case 16):
    the app acquires more images than it may hold, with an infinite timeout."""
    def frame(self, present_before_next):
        f = Frame(swapchains={"swapchain": (3, 2)})   # may hold 2 at once
        h = f.q("host")
        for i in range(3):
            h.append(Acquire("swapchain", where=f"frame {i}: acquire"))
            if present_before_next or i == 2:
                h.append(Present("swapchain", f"frame {i}: present"))
        return f

    def test_overacquire_rejected(self):
        r = timing.check(self.frame(present_before_next=False))
        self.assertEqual(r.codes(), ["timing.swapchain-overacquire"], str(r))

    def test_paced_frames_accepted(self):
        self.assertTrue(timing.check(self.frame(present_before_next=True)).ok)


class BarrierScopes(unittest.TestCase):
    """Barriers order only what their stage/access masks name."""
    def compute_then_draw(self, dst_stage, dst_access):
        f = Frame()
        g = f.q("graphics")
        g.append(Use("verts", "write", where="skinning (compute)", stage="COMPUTE_SHADER",
                     flag="SHADER_STORAGE_WRITE"))
        g.append(Barrier("verts", where="barrier", src_stages=frozenset({"COMPUTE_SHADER"}),
                         src_access=frozenset({"SHADER_WRITE"}),
                         dst_stages=frozenset({dst_stage}), dst_access=frozenset({dst_access})))
        g.append(Use("verts", "read", where="draw reads vertices", stage="VERTEX_INPUT",
                     flag="VERTEX_ATTRIBUTE_READ"))
        return timing.check(f)

    def test_right_masks(self):
        self.assertTrue(self.compute_then_draw("VERTEX_INPUT", "VERTEX_ATTRIBUTE_READ").ok)

    def test_wrong_masks(self):
        r = self.compute_then_draw("VERTEX_SHADER", "SHADER_READ")
        self.assertEqual(r.codes(), ["timing.barrier-scope"], str(r))

    def test_attachment_writes_in_one_pass_are_ordered(self):
        f = Frame()
        g = f.q("graphics")
        for pass_id, n in ((1, 2), (2, 1)):
            for k in range(n):
                g.append(Use("hdr", "write", where=f"pass {pass_id} draw {k}",
                             stage="COLOR_ATTACHMENT_OUTPUT", flag="COLOR_ATTACHMENT_WRITE",
                             scope=pass_id))
        # draws 0 and 1 share a pass (rasterization order); pass 2 needs a barrier
        r = timing.check(f)
        self.assertEqual(r.codes(), ["timing.missing-barrier"], str(r))
        self.assertIn("pass 2", str(r))

    def test_mip_chain_wrong_src_access(self):
        f = Frame()
        g = f.q("graphics")
        for i in range(1, 3):
            g.append(Use(f"mip{i}", "write", where=f"blit into mip {i}", stage="TRANSFER",
                         flag="TRANSFER_WRITE"))
            g.append(Barrier(f"mip{i}", where=f"barrier after mip {i}",
                             src_stages=frozenset({"TRANSFER"}),
                             src_access=frozenset({"TRANSFER_READ"}),     # should be WRITE
                             dst_stages=frozenset({"TRANSFER"}),
                             dst_access=frozenset({"TRANSFER_READ"})))
            g.append(Use(f"mip{i}", "read", where=f"blit from mip {i}", stage="TRANSFER",
                         flag="TRANSFER_READ"))
        self.assertIn("timing.barrier-scope", timing.check(f).codes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
