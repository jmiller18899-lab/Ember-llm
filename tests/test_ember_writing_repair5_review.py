"""CPU checks for the committed WR5 evidence and automatic verdict."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "jobs"))
import ember_writing_repair5_review as R


class WritingRepair5ReviewTests(unittest.TestCase):
    def test_saved_totals_reject_wr5(self):
        result = R.summary()
        self.assertEqual(result["totals"]["before_pass"], 714)
        self.assertEqual(result["totals"]["after_pass"], 711)
        self.assertEqual(result["shortening"]["before"]["verbatim_copy"], 5)
        self.assertEqual(result["shortening"]["after"]["verbatim_copy"], 8)
        self.assertEqual(result["lost"], [
            ("fresh_writing", "fresh-shorten-03"),
            ("promo_v2", "draft-short-00"),
            ("promo_v2", "draft-short-05"),
            ("suite_v3", "drafting-07"),
        ])
        self.assertEqual(result["gained"], [("probes", "probe-context-00")])
        self.assertTrue(result["rejected"])

    def test_copying_replay_worsens_after_step_96(self):
        steps = R.summary()["copying_steps"]
        self.assertEqual(steps["0"]["verbatim_copies"], 5)
        self.assertEqual(steps["96"]["verbatim_copies"], 6)
        self.assertEqual(steps["128"]["verbatim_copies"], 8)

    def test_checksum_manifest_covers_evaluation_files(self):
        folder = ROOT / "reports/evidence/writing-repair5-20261003"
        listed = {line.split()[1] for line in (folder / "SHA256SUMS").read_text().splitlines() if line.strip()}
        for name in ("evidence/final-report.json", "evidence/baseline-744.json",
                     "evidence/candidate-744.json", "evidence/shortening-after.json"):
            self.assertIn("./" + name, listed)


if __name__ == "__main__":
    unittest.main()
