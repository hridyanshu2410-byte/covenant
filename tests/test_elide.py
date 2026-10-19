"""Tests for covenant.elide (proof-licensed barrier elision) on the recorded
demonstration traces in bench/elide/traces (see bench/elide/README.md)."""
import os
import unittest

from covenant import elide

HERE = os.path.dirname(os.path.abspath(__file__))
TRACES = os.path.join(HERE, "..", "bench", "elide", "traces")


def elider(w):
    return elide.Elider(os.path.join(TRACES, f"{w}.jsonl"), os.path.join(TRACES, f"{w}-spv"))


@unittest.skipUnless(os.path.exists(os.path.join(TRACES, "postchain.jsonl")), "demonstration traces not present")
class ElideTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            elide.VkNames.find_header()
        except FileNotFoundError as ex:     # flag names are read from the Vulkan headers
            raise unittest.SkipTest(str(ex))
        cls.plans = {}
        for w in ("particles", "deferred", "postchain"):
            el = elider(w)
            el.run()
            cls.plans[w] = (el, el.plan_json())

    def test_plan_has_no_new_hazard(self):
        for w, (el, pj) in self.plans.items():
            bad, _, _ = el.new_hazards(el.plan)
            self.assertEqual(bad, [], w)
            self.assertEqual(pj["proof"]["plain_timing_check"]["new"], 0, w)

    def test_removes_and_narrows(self):
        d = self.plans["particles"][1]["summary"]["decisions"]
        self.assertGreaterEqual(d.get("remove", 0), 24)        # 8 of 12 per frame, 3 frames
        d = self.plans["postchain"][1]["summary"]["decisions"]
        self.assertGreaterEqual(d.get("remove", 0), 18)

    def test_transitions_and_host_barriers_never_removed(self):
        for w, (el, pj) in self.plans.items():
            for b in pj["barriers"]:
                for e in b["elements"]:
                    if e["kind"] == "image" and e["layout"].split(" -> ")[0] != e["layout"].split(" -> ")[1]:
                        self.assertNotEqual(e["action"], "remove", (w, b["seq"]))
                    if "HOST" in e["recorded"]["dst_stage"] or "HOST" in e["recorded"]["src_stage"]:
                        self.assertEqual(e["action"], "keep", (w, b["seq"]))

    def test_joint_check_matters(self):
        """postchain: every full barrier is redundant on its own, but removing
        all of them together races; the greedy search keeps every other one."""
        plan, bad = elide.naive_control(elider("postchain"))
        self.assertGreater(len(bad), 0)
        self.assertGreater(len(plan), self.plans["postchain"][1]["summary"]["decisions"]["remove"])

    def test_chain_plan_verified(self):
        for w, (el, pj) in self.plans.items():
            self.assertTrue(pj["chain"], w)
            self.assertIn("re-verified", pj["summary"]["chain_note"])

    def test_signature_format(self):
        names = elide.VkNames()
        rec = {"seq": 1, "fn": "vkCmdPipelineBarrier2", "cb": "command_buffer#1", "dependency_flags": [],
               "memory_barriers": [{"src_stage": ["VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT"],
                                    "src_access": ["VK_ACCESS_2_MEMORY_WRITE_BIT"],
                                    "dst_stage": ["VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT"],
                                    "dst_access": ["VK_ACCESS_2_SHADER_READ_BIT"]}],
               "buffer_barriers": [], "image_barriers": []}
        c = elide.Call(rec, 0, 0)
        self.assertEqual(elide.call_signature(c, names), "2|0|M:65536:65536:2048:32")

    def test_staged_signal_pins_batch(self):
        """A batch that signals a semaphore with a stage mask narrower than
        ALL_COMMANDS is outside the oracle's model: its barriers are pinned."""
        import json
        import tempfile
        src = os.path.join(TRACES, "particles.jsonl")
        recs = [json.loads(l) for l in open(src)]
        sem = {"seq": 0, "fn": "vkCreateSemaphore", "semaphore": "semaphore#99", "type": "binary", "initial_value": 0}
        first = None
        for r in recs:
            if r["fn"] == "vkQueueSubmit" and first is None:
                first = r["seq"]
                r["submits"][0]["signal"] = [{"semaphore": "semaphore#99", "value": 0,
                                              "stage": ["VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT"]}]
        with tempfile.TemporaryDirectory() as d:
            t = os.path.join(d, "t.jsonl")
            with open(t, "w") as f:
                for r in [recs[0], sem] + recs[1:]:
                    f.write(json.dumps(r) + "\n")
            el = elide.Elider(t, os.path.join(TRACES, "particles-spv"))
        pinned = [c for c in el.calls if c.pin and "stage mask" in c.pin]
        self.assertTrue(pinned)
        self.assertTrue(all(first in c.submits for c in pinned))


if __name__ == "__main__":
    unittest.main()
