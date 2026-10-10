"""CPU contracts for WR10 keep-pass + copy-margin training."""
import math
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "jobs"))
import ember_writing_repair1_data as D
import ember_writing_repair8_data as W8
import ember_writing_repair9_data as W9
import ember_writing_repair10_data as W
import ember_writing_repair10_train as T
import ember_meaning_preservation_v3 as V3


class WritingRepair10Tests(unittest.TestCase):
    def test_composition_and_step_budget(self):
        rows = W.build_train_rows(D.retention("train"))
        self.assertEqual(len(rows), 416)
        self.assertEqual(T.STEPS * T.ACCUM, 416)
        self.assertEqual(T.STEPS, 52)
        self.assertEqual(Counter(r["family"] for r in rows),
                         {"shortening": 96, "recipient": 64, "arithmetic": 64,
                          "extraction": 64, "grounding": 48, "clarification": 48, "direct": 32})
        short = [r for r in rows if r["family"] == "shortening"]
        self.assertTrue(all(r["loss"] == "copy_margin" and r["rejected"] == r["source"] for r in short))
        self.assertTrue(all(r.get("overedit") for r in short))
        encoded = W.encode_examples(rows)
        self.assertEqual(len(encoded), 416)
        self.assertEqual(sum(r["loss_kind"] == 1 for r in encoded), 96)
        self.assertTrue(all(r.get("overedit") for r in encoded if r["loss_kind"] == 1))

    def test_preferred_passes_v3_and_both_rejects_fail(self):
        for row in W.writing("train"):
            if row["family"] != "shortening":
                continue
            good = V3.grade_case({"kind": "shortening", "source": row["source"]},
                                 row["answer"], lambda r, o: True)
            copied = V3.grade_case({"kind": "shortening", "source": row["source"]},
                                   row["rejected"], lambda r, o: True)
            over = V3.grade_case({"kind": "shortening", "source": row["source"]},
                                 row["overedit"], lambda r, o: True)
            self.assertEqual(good["status"], "pass", row["id"])
            self.assertNotEqual(copied["status"], "pass", row["id"])
            self.assertIn("not_shorter", copied["reasons"])
            self.assertNotEqual(over["status"], "pass", row["id"])
            self.assertEqual(row["rejected"], row["source"])
            if row["shortening_group"] in ("reminder", "reminder_forget"):
                self.assertTrue(any(reason.startswith("reminder") for reason in over["reasons"]), row["id"])

    def test_margin_grows_when_the_rejected_text_is_more_likely(self):
        source_preferred = W.copy_margin(2.0, 0.25)
        short_preferred = W.copy_margin(0.25, 2.0)
        self.assertGreater(source_preferred, short_preferred)
        self.assertAlmostEqual(short_preferred, math.log1p(math.exp(-1.75)))
        self.assertAlmostEqual(W.copy_margin(1.0, 1.0), math.log(2))

    def test_no_overlap_with_wr8_wr9_or_benchmark_fourgrams(self):
        import json
        train = W.writing("train")
        dev = W.writing("dev")
        history = W8.writing("train") + W8.writing("dev") + W9.writing("train") + W9.writing("dev")
        audit = W.audit(train, dev, history, [])
        self.assertEqual(audit["exact_prompt_source_overlaps"], 0)
        self.assertEqual(audit["copy_margin_pairs"], 96)
        self.assertEqual(audit["keep_pass_pairs"], 96)
        payload = json.loads((ROOT / "reports/evidence/writing-repair5-20261003/evidence/baseline-744.json").read_text())
        sources = [W.shortening_source(record.get("row") or record) for record in payload["records"]]
        sources = [source for source in sources if source]
        self.assertEqual(len(sources), 30)
        grams = set().union(*(W.fourgrams(source) for source in sources))
        for row in train:
            if row["family"] != "shortening":
                continue
            overlap = W.fourgrams(row["source"]) & grams
            self.assertFalse(overlap, row["id"] + " " + str(overlap))

    def test_selection_keeps_the_copy_floor(self):
        baseline = {"step": 0, "copies": 2, "force_fails": 1, "v3_pass": 4}
        snapshots = [
            baseline,
            {"step": 13, "copies": 4, "force_fails": 1, "v3_pass": 8},
            {"step": 26, "copies": 2, "force_fails": 0, "v3_pass": 6},
            {"step": 52, "copies": 3, "force_fails": 0, "v3_pass": 9},
        ]
        self.assertEqual(T.choose_selected(baseline, snapshots), 26)

    def test_selection_falls_back_to_repair2(self):
        baseline = {"step": 0, "copies": 2, "force_fails": 0, "v3_pass": 4}
        snapshots = [baseline, {"step": 52, "copies": 5, "force_fails": 2, "v3_pass": 10}]
        self.assertEqual(T.choose_selected(baseline, snapshots), 0)

    def test_collate_keeps_source_and_overedit_pairs(self):
        example = {"input_ids": [1, 2], "labels": [-100, 3], "attention_mask": [1, 1], "loss_kind": 1,
                   "copy_input_ids": [1, 4], "copy_labels": [-100, 5], "copy_attention_mask": [1, 1],
                   "overedit_input_ids": [1, 6], "overedit_labels": [-100, 7], "overedit_attention_mask": [1, 1]}
        rows = T.collate_rows([example])
        self.assertEqual(rows["loss_kind"], [1])
        self.assertEqual(rows["copy_input_ids"], [[1, 4]])
        self.assertEqual(rows["overedit_input_ids"], [[1, 6]])
        with self.assertRaises(ValueError):
            T.collate_rows([example, example])
        broken = dict(example, copy_labels=[-100, -100])
        with self.assertRaises(ValueError):
            T.collate_rows([broken])
        broken_over = dict(example, overedit_labels=[-100, -100])
        with self.assertRaises(ValueError):
            T.collate_rows([broken_over])

    def test_data_imports_when_jobs_directory_is_absent(self):
        work = Path(tempfile.mkdtemp(prefix="wr10-import-"))
        saved_path = sys.path[:]
        saved_modules = {
            name: sys.modules.pop(name)
            for name in (
                "ember_meaning_preservation_v3",
                "ember_drafting_repair_candidates_eval",
                "wr10_data",
                "wr10_frozen_engine",
            )
            if name in sys.modules
        }
        sys.path[:] = [
            entry for entry in sys.path
            if not (Path(entry) / "ember_meaning_preservation_v3.py").is_file()
        ]
        try:
            frozen, data, checker = T.load_experiment_modules(
                work,
                (ROOT / "jobs/ember_writing_repair2_train.py").read_bytes(),
                (ROOT / "jobs/ember_writing_repair10_data.py").read_bytes(),
                (ROOT / "jobs/ember_meaning_preservation_v3.py").read_bytes(),
                (ROOT / "jobs/ember_drafting_repair_candidates_eval.py").read_bytes(),
            )
            self.assertEqual(data.VERSION, "ember-writing-repair10-keep-pass-v1")
            self.assertEqual(checker.GRADER_VERSION, "meaning-preservation-v3")
            self.assertTrue(hasattr(frozen, "load_inputs"))
        finally:
            sys.path[:] = saved_path
            for name in (
                "wr10_data", "wr10_frozen_engine", "ember_meaning_preservation_v3",
                "ember_drafting_repair_candidates_eval",
            ):
                sys.modules.pop(name, None)
            sys.modules.update(saved_modules)

    def test_launch_guards_and_pins(self):
        env = {"GITHUB_REF": "refs/heads/" + T.BRANCH, "GITHUB_RUN_ATTEMPT": "1",
               "WR10_CODE_COMMIT": "a" * 40, "HF_TOKEN": "x"}
        self.assertTrue(T.validate_launch(env, False))
        with self.assertRaises(ValueError):
            T.validate_launch(env, True)
        self.assertFalse(T.AUTO_PROMOTION)
        self.assertFalse(T.AUTO_RETRY)
        self.assertEqual(T.SOURCE, "Jmiller18899/ember-qwen3.5-4b-repair2")
        self.assertEqual(T.TIMEOUT, "90m")
        self.assertEqual(T.OUT, "Jmiller18899/ember-qwen3.5-4b-writing-repair10-20261010")
        self.assertEqual(hashlib_data(), T.DATA_SHA)

    def test_pre_optimizer_crash_can_be_replaced(self):
        crashed = ["launch.json", "checkpoint-0/adapter_model.safetensors", "evidence/training-metrics.json"]
        self.assertTrue(T.failed_before_optimizer(crashed))
        self.assertFalse(T.failed_before_optimizer(crashed + ["checkpoints/step-13/adapter_model.safetensors"]))
        self.assertFalse(T.failed_before_optimizer(crashed + ["candidate/adapter_model.safetensors"]))

    def test_dev_compact_rows_are_held_out(self):
        dev = W.writing("dev")
        compact = [r for r in dev if r.get("shortening_group") == "compact_diagnostic"]
        self.assertEqual(len(compact), 8)
        self.assertTrue(all(r["training_allowed"] is False and r["loss"] == "eval" for r in compact))
        self.assertEqual(len(dev), 56)

    def test_starts_from_repair2_not_wr9(self):
        self.assertEqual(T.SOURCE, "Jmiller18899/ember-qwen3.5-4b-repair2")
        self.assertNotIn("writing-repair9", T.SOURCE)
        self.assertTrue(T.OUT.endswith("writing-repair10-20261010"))


def hashlib_data():
    import hashlib
    return hashlib.sha256((ROOT / "jobs/ember_writing_repair10_data.py").read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
