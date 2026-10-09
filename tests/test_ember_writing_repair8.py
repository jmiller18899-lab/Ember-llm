"""CPU contracts for WR8 meaning-aware contrastive training."""
import os
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "jobs"))
import ember_writing_repair1_data as D
import ember_writing_repair7_data as W7
import ember_writing_repair8_data as W
import ember_writing_repair8_train as T
import ember_meaning_preservation_v3 as V3


class WritingRepair8Tests(unittest.TestCase):
    def test_composition_and_encoded_budget(self):
        rows = W.build_train_rows(D.retention("train"))
        self.assertEqual(len(rows), 416)
        self.assertEqual(Counter(r["family"] for r in rows),
                         {"shortening": 96, "recipient": 64, "arithmetic": 64,
                          "extraction": 64, "grounding": 48, "clarification": 48, "direct": 32})
        short = [r for r in rows if r["family"] == "shortening"]
        self.assertEqual(len(short), 96)
        self.assertTrue(all(r["loss"] == "contrastive" and r.get("rejected") for r in short))
        encoded = W.encode_pairs(rows)
        self.assertEqual(len(encoded), 512)
        self.assertEqual(sum(r["kind"] == 1 for r in encoded), 96)
        self.assertEqual(sum(r["kind"] == 0 for r in encoded), 416)

    def test_preferred_passes_v3_and_rejected_does_not(self):
        for row in W.writing("train"):
            if row["family"] != "shortening":
                continue
            good = V3.grade_case({"kind": "shortening", "source": row["source"]},
                                 row["answer"], lambda r, o: True)
            bad = V3.grade_case({"kind": "shortening", "source": row["source"]},
                                row["rejected"], lambda r, o: True)
            self.assertEqual(good["status"], "pass", row["id"])
            self.assertNotEqual(bad["status"], "pass", row["id"])
            self.assertNotEqual(W.norm(row["rejected"]), W.norm(row["answer"]))
            self.assertNotEqual(W.norm(row["rejected"]), W.norm(row["source"]) or row["shortening_group"] == "consent_unless")

    def test_force_and_reminder_negatives(self):
        rows = [r for r in W.writing("train") if r["family"] == "shortening"]
        should = next(r for r in rows if r["shortening_group"] == "timing_should")
        self.assertIn(" must ", should["rejected"])
        self.assertIn(" should ", should["answer"])
        reminder = next(r for r in rows if r["shortening_group"] == "reminder")
        self.assertIn("Please remember", reminder["answer"])
        self.assertNotIn("remember", reminder["rejected"].casefold())

    def test_no_overlap_with_wr7_or_cross_split(self):
        train = W.writing("train")
        dev = W.writing("dev")
        history = W7.writing("train") + W7.writing("dev")
        audit = W.audit(train + [dict(id="wr8-retain-x", prompt="retain only", answer="1",
                                      split="train", training_allowed=True, family="direct", loss="sft")],
                        dev, history, [])
        self.assertEqual(audit["exact_prompt_source_overlaps"], 0)
        train_src = {W.norm(r["source"]) for r in train if r.get("source")}
        self.assertFalse(train_src & {W.norm(r["source"]) for r in history if r.get("source")})

    def test_selection_rejects_copy_increase(self):
        baseline = {"step": 0, "copies": 2, "force_fails": 1, "v3_pass": 4}
        snapshots = [
            baseline,
            {"step": 32, "copies": 4, "force_fails": 1, "v3_pass": 8},
            {"step": 64, "copies": 2, "force_fails": 0, "v3_pass": 6},
            {"step": 128, "copies": 3, "force_fails": 0, "v3_pass": 9},
        ]
        self.assertEqual(T.choose_selected(baseline, snapshots), 64)

    def test_selection_falls_back_to_repair2(self):
        baseline = {"step": 0, "copies": 2, "force_fails": 0, "v3_pass": 4}
        snapshots = [baseline, {"step": 128, "copies": 5, "force_fails": 2, "v3_pass": 10}]
        self.assertEqual(T.choose_selected(baseline, snapshots), 0)

    def test_launch_guards(self):
        env = {"GITHUB_REF": "refs/heads/" + T.BRANCH, "GITHUB_RUN_ATTEMPT": "1",
               "WR8_CODE_COMMIT": "a" * 40, "HF_TOKEN": "x"}
        self.assertTrue(T.validate_launch(env, False))
        with self.assertRaises(ValueError):
            T.validate_launch(env, True)
        with self.assertRaises(ValueError):
            T.validate_launch({**env, "GITHUB_REF": "refs/heads/main"}, False)
        local = {"WR8_ALLOW_LOCAL_LAUNCH": "1", "WR8_LOCAL_BRANCH": T.BRANCH,
                 "WR8_CODE_COMMIT": "b" * 40, "HF_TOKEN": "x"}
        self.assertTrue(T.validate_launch(local, False))

    def test_encode_ids_keeps_kind(self):
        row = T.encode_ids([1, 2], [3], 4, 16, 1)
        self.assertEqual(row["kind"], 1)
        self.assertEqual(row["labels"][:2], [-100, -100])
        self.assertEqual(row["input_ids"], [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            T.encode_ids([1], [2], 3, 16, 2)

    def test_v3_is_not_the_frozen_grader(self):
        self.assertEqual(V3.GRADER_VERSION, "meaning-preservation-v3")
        self.assertNotEqual(T.DATA_SHA, T.ENGINE_SHA)
        self.assertFalse(T.AUTO_PROMOTION)
        self.assertEqual(T.SOURCE, "Jmiller18899/ember-qwen3.5-4b-repair2")

    def test_dev_compact_rows_exist_and_are_held_out(self):
        dev = W.writing("dev")
        compact = [r for r in dev if r.get("shortening_group") == "compact_diagnostic"]
        self.assertEqual(len(compact), 8)
        self.assertTrue(all(r["training_allowed"] is False for r in compact))
        self.assertTrue(all(not r["id"].startswith("wr7-") for r in compact))


if __name__ == "__main__":
    unittest.main()
