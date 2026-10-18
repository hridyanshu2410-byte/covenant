"""Access-scope and same-call rules of covenant.timing, from the two checker
bugs the differential test against the Khronos synchronization validation
(syncval) found (bench/elide/difftest)."""
import unittest

from covenant.diagnostics import Report
from covenant.timing import Barrier, Frame, Use, check

F = frozenset
CA, CAW = "COLOR_ATTACHMENT_OUTPUT", "COLOR_ATTACHMENT_WRITE"


def codes(fr):
    return sorted({d.code for d in check(fr, Report()).items})


class AccessScope(unittest.TestCase):
    def frame(self, dst_stage):
        return Frame(queues={"gfx": [
            Use("hdr", "write", None, "draw", CA, CAW),
            Barrier("hdr", None, None, "barrier", F({CA}), F({CAW}), F({dst_stage}), F({"SHADER_READ"})),
            Use("hdr", "read", None, "tonemap", "FRAGMENT_SHADER", "SHADER_SAMPLED_READ"),
        ]})

    def test_logically_earlier_dst_stage_orders_but_does_not_make_visible(self):
        # VERTEX_SHADER blocks the fragment stage (execution), but the write is
        # made visible only to accesses of stages named in the mask
        self.assertEqual(codes(self.frame("VERTEX_SHADER")), ["timing.barrier-scope"])
        self.assertEqual(codes(self.frame("FRAGMENT_SHADER")), [])


class SameCall(unittest.TestCase):
    def test_structs_of_one_call_do_not_chain(self):
        lt, ltf = F({"LAYOUT_TRANSITION"}), F({"LAYOUT_TRANSITION_WRITE"})
        ops = [
            # the image struct's transition (as elide models it: B1 ; write ; B2)
            Barrier("hdr", "color", "transfer_src", "call", F({CA}), F({CAW}), lt, ltf, call="c", struct=1),
            Use("hdr", "write", None, "call (layout transition)", "LAYOUT_TRANSITION",
                "LAYOUT_TRANSITION_WRITE", call="c", struct=1),
            Barrier("hdr", None, None, "call (after)", lt, ltf, F({"TRANSFER"}), F(), call="c", struct=1),
            # a memory barrier in the same call cannot make that transition visible
            Barrier(None, None, None, "call (memory)", F({"ALL_COMMANDS"}), F({"MEMORY_WRITE"}),
                    F({"TRANSFER"}), F({"TRANSFER_READ"}), call="c", struct=2),
            Use("hdr", "read", None, "copy", "TRANSFER", "TRANSFER_READ"),
        ]
        self.assertEqual(codes(Frame(queues={"gfx": ops}, initial_layouts={"hdr": "color"})),
                         ["timing.barrier-scope"])
        # the same memory barrier in a later call does chain
        ops[3] = Barrier(None, None, None, "next call", F({"ALL_COMMANDS"}), F({"MEMORY_WRITE"}),
                         F({"TRANSFER"}), F({"TRANSFER_READ"}), call="d", struct=1)
        self.assertEqual(codes(Frame(queues={"gfx": ops}, initial_layouts={"hdr": "color"})), [])


if __name__ == "__main__":
    unittest.main()
