"""Data-quality and response-persistence contracts before another writing run."""
import copy
import importlib
import importlib.util
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jobs"))
import ember_writing_repair1_data as D
import ember_writing_repair12_data as OLD
import ember_writing_repair12_train as T
import ember_meaning_preservation_v3 as V3


class RevisedWritingData(unittest.TestCase):
    def data(self):
        name = "ember_writing_repair13_data"
        self.assertIsNotNone(importlib.util.find_spec(name), "revised shortening examples are missing")
        return importlib.import_module(name)

    def test_reminder_examples_save_several_words_without_losing_the_cue(self):
        W = self.data()
        rows = W.writing("train")
        self.assertEqual(len(rows), 96)
        for row in rows:
            self.assertGreaterEqual(len(row["source"].split()) - len(row["answer"].split()), 3, row["id"])
            self.assertLessEqual(len(row["answer"]) / len(row["source"]), .90, row["id"])
            self.assertEqual(V3.grade_case({"kind": "shortening", "source": row["source"]}, row["answer"], lambda r, o: True)["status"], "pass", row["id"])
            self.assertIsNone(re.search(r"\b(?:each|every) (?:trainees|bakers|cashiers|porters|custodians|tutors|gardeners|installers|electricians|receptionists|technicians|repairers)\b", row["source"]), row["id"])
            if row["shortening_group"] in ("reminder", "reminder_forget"):
                self.assertTrue(row["answer"].startswith(("Please remember", "Do not forget")))

    def test_contrasts_isolate_a_meaning_error_in_the_short_answer(self):
        W = self.data()
        for row in W.writing("train"):
            self.assertEqual(row["rejected"], row["source"])
            self.assertLess(len(row["overedit"]), len(row["source"]), row["id"])
            bad = V3.grade_case({"kind": "shortening", "source": row["source"]}, row["overedit"], lambda r, o: True)
            self.assertNotEqual(bad["status"], "pass", row["id"])
            if row["shortening_group"] in ("reminder", "reminder_forget"):
                self.assertIn("reminder_dropped", bad["reasons"])
                self.assertEqual(row["overedit"], re.sub(r"^(?:Please remember |Do not forget )", "", row["answer"]))
            if row["shortening_group"] == "timing_should":
                self.assertIn("force_changed:recommendation->requirement", bad["reasons"])

    def test_short_answers_keep_articles_for_singular_countable_objects(self):
        W = self.data()
        for row in W.writing("train") + W.writing("dev"):
            if row["action_verb"] in ("keep", "hold", "bring", "carry", "inspect", "pack"):
                self.assertRegex(row["answer"], rf"\b{row['action_verb']} the ", row["id"])

    def test_other_training_families_remain_exactly_unchanged(self):
        W = self.data()
        parent = OLD.build_train_rows(D.retention("train"))
        new = W.build_train_rows(D.retention("train"))
        self.assertEqual(len(new), 416)
        self.assertEqual([r for r in parent if r["family"] != "shortening"], [r for r in new if r["family"] != "shortening"])
        self.assertEqual(len(OLD.encode_examples(new)), 416)

    def test_development_is_fresh_and_is_not_serializable_as_training(self):
        W = self.data()
        train = W.build_train_rows(D.retention("train"))
        dev = W.writing("dev")
        self.assertEqual(len(dev), 32)
        self.assertTrue(all(r["training_allowed"] is False for r in dev))
        history = OLD.writing("train") + OLD.writing("dev")
        report = W.audit(train, dev, history, [])
        self.assertEqual(report["exact_prompt_source_overlaps"], 0)
        with self.assertRaises(ValueError):
            OLD.sft_bytes([dev[0]])
        changed = copy.deepcopy(train)
        first = next(r for r in changed if r["family"] == "shortening")
        first["source"] = history[0]["source"]
        with self.assertRaises(ValueError):
            W.audit(changed, dev, history, [])
        with self.assertRaises(ValueError):
            W.audit(train, dev, [], [next(r for r in train if r["family"] == "shortening")["source"]])
        leaked_dev = copy.deepcopy(dev)
        leaked_dev[0]["prompt"] = next(r for r in train if r["family"] != "shortening")["prompt"]
        with self.assertRaises(ValueError):
            W.audit(train, leaked_dev, [], [])


class SavedCheckpointResponses(unittest.TestCase):
    def bundle(self, step=13, outputs=None, snapshot=None):
        fn = getattr(T, "checkpoint_response_evidence", None)
        self.assertTrue(callable(fn), "checkpoint response texts are not persisted")
        outputs = outputs if outputs is not None else [dict(id="probe-a", family="shortening", source="Long source.", prompt="Shorten it.", output="  Exact response.\n", training_allowed=False)]
        snapshot = snapshot if snapshot is not None else dict(step=13, v3_pass=0, meaning=dict(rows=[dict(id="probe-a", status="review", reasons=["unverified_paraphrase_or_scope"])]))
        return fn(step, outputs, snapshot, "checkpoints/step-13", "a" * 40)

    def test_raw_response_and_case_grade_are_saved_together(self):
        b = self.bundle()
        self.assertEqual(b["step"], 13)
        self.assertEqual(b["checkpoint_revision"], "a" * 40)
        self.assertEqual(b["responses"][0]["output"], "  Exact response.\n")
        self.assertEqual(b["responses"][0]["semantic_grade"]["status"], "review")
        self.assertTrue(b["responses"][0]["manual_review_pending"])
        self.assertFalse(b["automatic_promotion"])

    def test_wrong_step_and_missing_case_alignment_are_rejected(self):
        with self.assertRaises(ValueError):
            self.bundle(step=26)
        with self.assertRaises(ValueError):
            self.bundle(outputs=[])
        with self.assertRaises(ValueError):
            self.bundle(snapshot=dict(step=13, v3_pass=0, meaning=dict(rows=[dict(id="different", status="review", reasons=[])])))

    def test_response_capture_does_not_rewrite_frozen_grades_or_input(self):
        outputs = [dict(id="probe-a", family="shortening", source="Long source.", output="Short.")]
        snapshot = dict(step=13, v3_pass=0, meaning=dict(rows=[dict(id="probe-a", status="fail", reasons=["reminder_dropped"])]))
        before = copy.deepcopy((outputs, snapshot))
        b = self.bundle(outputs=outputs, snapshot=snapshot)
        self.assertEqual((outputs, snapshot), before)
        self.assertEqual(b["responses"][0]["semantic_grade"]["reasons"], ["reminder_dropped"])
        self.assertEqual(b["snapshot"], snapshot)


if __name__ == "__main__":
    unittest.main()
