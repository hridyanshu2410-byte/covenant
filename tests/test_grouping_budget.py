"""Report grouping under a budget. When an oracle run goes over
the hazard-pair or memory budget the result says "incomplete" with the reason, every
diagnostic is still a member of a group, and nothing is reported as complete."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from covenant import report_groups, timing  # noqa: E402
from covenant.timing import Frame, Use  # noqa: E402


def frame(n=6):
    """n buffers, each written twice and read on one queue with no barrier: per buffer the
    epoch mode reports 2 hazard pairs and the pairwise mode 3."""
    f = Frame()
    q = f.q("graphics0")
    for i in range(n):
        for k, acc in enumerate(("write", "write", "read")):
            q.append(Use(f"buffer#{i}", acc, where=f"vkCmdDispatch @seq {10 * i + k} (cb#{i}): set 0 binding 0",
                         stage="COMPUTE_SHADER", flag="SHADER_STORAGE_WRITE" if acc == "write" else "SHADER_STORAGE_READ",
                         cmd=object()))
    return f


class GroupBudget(unittest.TestCase):
    def members(self, res):
        return sum(g["diagnostics"] for g in res["repairs"] + res["other_groups"])

    def test_unbounded_is_complete(self):
        res = report_groups.group(frame(), max_pairs=0, budget_mb=0)
        self.assertEqual(res["status"], "complete")
        self.assertEqual(res["incomplete_reasons"], [])

    def test_pairwise_over_budget_is_incomplete_and_keeps_every_diagnostic(self):
        n = 6
        full = report_groups.group(frame(n), max_pairs=0, budget_mb=0)
        res = report_groups.group(frame(n), max_pairs=2 * n, budget_mb=0)   # epoch fits, pairwise does not
        self.assertEqual(res["status"], "incomplete")
        self.assertIn("pairwise base run", res["incomplete_reasons"][0])
        self.assertEqual(res["diagnostics"], full["diagnostics"])
        self.assertGreaterEqual(self.members(res), res["diagnostics"])
        self.assertEqual(res["repair_groups"], 0)
        self.assertIn("REPORTING INCOMPLETE", report_groups.format_result(res))

    def test_base_over_budget_falls_back_to_message_groups(self):
        res = report_groups.group(frame(), max_pairs=1, budget_mb=0)
        self.assertEqual(res["status"], "incomplete")
        self.assertIn("base run", res["incomplete_reasons"][0])
        rep = timing.check(frame(), window=False)
        self.assertEqual(res["diagnostics"], len(rep.items))
        self.assertEqual(self.members(res), len(rep.items))

    def test_memory_budget(self):
        res = report_groups.group(frame(), max_pairs=0, budget_mb=1)       # any process is over 1 MB
        self.assertEqual(res["status"], "incomplete")
        self.assertIn("MB", " ".join(res["incomplete_reasons"]))
        self.assertEqual(self.members(res), res["diagnostics"])

    def test_cli_writes_incomplete_stub_first(self):
        import json
        import tempfile
        d = Path(tempfile.mkdtemp())
        out = d / "g.json"
        try:
            report_groups.main([str(d / "missing.jsonl"), "--json", str(out)])
        except (OSError, ValueError):
            pass
        self.assertEqual(json.loads(out.read_text())["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
