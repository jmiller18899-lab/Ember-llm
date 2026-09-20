import unittest

from jobs.ember_qwen35_4b_evidence_recovery import build_summary


class EvidenceRecoveryTests(unittest.TestCase):
    def test_build_summary_detects_improvement_without_retraining(self):
        base = [
            {"suite": "frozen", "family": "classification", "scoring": "exact", "exact_match": True},
            {"suite": "frozen", "family": "arithmetic", "scoring": "exact", "exact_match": False},
            {"suite": "frozen", "family": "drafting", "scoring": "rubric", "exact_match": None},
        ]
        final = [
            {"suite": "frozen", "family": "classification", "scoring": "exact", "exact_match": True},
            {"suite": "frozen", "family": "arithmetic", "scoring": "exact", "exact_match": True},
            {"suite": "frozen", "family": "drafting", "scoring": "rubric", "exact_match": None},
        ]

        summary = build_summary(base, final)

        self.assertEqual(summary["base_exact"], {"pass": 1, "total": 2})
        self.assertEqual(summary["final_exact"], {"pass": 2, "total": 2})
        self.assertEqual(summary["exact_delta"], 1)
        self.assertEqual(summary["strict_family_regressions"], [])
        self.assertEqual(summary["eval_case_count"], 3)
        self.assertFalse(summary["production_ready"])


if __name__ == "__main__":
    unittest.main()
