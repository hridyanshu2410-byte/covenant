"""elide.py pins the barriers around any command the model does not interpret
: a stub recorded in a command buffer pins that recording's
barriers; one the trace cannot place (no command buffer, or an unmodelled host
call) pins every barrier."""
import json
import os
import tempfile
import unittest

from covenant import elide

HERE = os.path.dirname(os.path.abspath(__file__))
TRACES = os.path.join(HERE, "..", "bench", "elide", "traces")
SRC = os.path.join(TRACES, "particles.jsonl")


@unittest.skipUnless(os.path.exists(SRC), "demonstration traces not present")
class ElideUnmodelled(unittest.TestCase):
    def run_with(self, extra_after_seq, rec):
        with open(SRC) as f:
            recs = [json.loads(x) for x in f if x.strip()]
        out = []
        for r in recs:
            out.append(r)
            if r["seq"] == extra_after_seq:
                out.append(dict(rec, seq=10 ** 6))
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        try:
            el = elide.Elider(f.name, os.path.join(TRACES, "particles-spv"))
            el.run()
            return el
        finally:
            os.unlink(f.name)

    def test_baseline_removes_barriers(self):
        el = elide.Elider(SRC, os.path.join(TRACES, "particles-spv"))
        el.run()
        self.assertTrue(any(not c.pin for c in el.calls))

    def test_stub_in_a_recording_pins_that_recording(self):
        el = self.run_with(105, {"fn": "vkCmdClearAttachments", "cb": "command_buffer#1", "warning": "unimplemented"})
        ordn = el.recording_of_seq[105]
        mine = [c for c in el.calls if c.rec_ord == ordn]
        self.assertTrue(mine)
        for c in mine:
            self.assertIn("vkCmdClearAttachments", c.pin or "", c.seq)
        self.assertTrue(any(not c.pin for c in el.calls if c.rec_ord != ordn))   # other recordings unaffected

    def test_stub_without_command_buffer_pins_everything(self):
        el = self.run_with(105, {"fn": "vkCmdClearAttachments", "warning": "unimplemented"})
        self.assertTrue(el.calls)
        for c in el.calls:
            self.assertIn("cannot place", c.pin or "")

    def test_unmodelled_host_call_pins_everything(self):
        el = self.run_with(105, {"fn": "vkQueueBindSparse", "warning": "unimplemented"})
        for c in el.calls:
            self.assertIn("vkQueueBindSparse", c.pin or "")

    def test_harmless_stub_pins_nothing_extra(self):
        el = self.run_with(105, {"fn": "vkCmdDebugMarkerInsertEXT", "warning": "unimplemented"})
        self.assertTrue(any(not c.pin for c in el.calls))


if __name__ == "__main__":
    unittest.main()
