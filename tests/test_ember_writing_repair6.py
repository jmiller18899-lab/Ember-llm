"""CPU contracts for the fixed WR6 hedge-removal experiment."""
import sys, unittest, json, copy
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "jobs"))
try:
    import ember_writing_repair6_data as W
    import ember_writing_repair6_train as T
except ModuleNotFoundError:
    W = T = None


class WritingRepair6Tests(unittest.TestCase):
    def test_next_experiment_is_implemented(self):
        self.assertIsNotNone(W, "WR6 training data and trainer must exist before launch")

    @unittest.skipIf(W is None, "Implementation not created yet")
    def test_training_composition_and_reference_budgets(self):
        import ember_writing_repair1_data as D
        rows = W.build_train_rows(D.retention("train"))
        self.assertEqual(len(rows), 512)
        self.assertEqual(Counter(r["family"] for r in rows),
                         {"shortening": 128, "recipient": 128, "arithmetic": 64,
                          "extraction": 64, "grounding": 48, "clarification": 48, "direct": 32})
        short = [r for r in rows if r["family"] == "shortening"]
        self.assertEqual(Counter(r["shortening_group"] for r in short),
                         {"hedge": 96, "condition": 16, "compact": 16})
        mean = sum(len(r["answer"]) / len(r["source"]) for r in short) / 128
        self.assertGreaterEqual(mean, 0.62)
        self.assertLessEqual(mean, 0.78)
        suffixes = set()
        for r in short:
            self.assertNotEqual(W.norm(r["answer"]), W.norm(r["source"]))
            self.assertLessEqual(len(r["answer"]) / len(r["source"]), 0.8)
            self.assertNotEqual(W.prefix2(r["answer"]), W.prefix2(r["source"]))
            self.assertIn(r["action_verb"], W.norm(r["answer"]).split())
            if r["shortening_group"] == "compact":
                suffixes.add(r["source"].rsplit(",", 1)[-1] if "," in r["source"] else r["structure"].split("{")[-1])
        self.assertGreaterEqual(len({r["structure"] for r in short if r["shortening_group"] == "compact"}), 8)

    @unittest.skipIf(W is None, "Implementation not created yet")
    def test_addressee_pronoun_converts_to_your(self):
        rows = W.writing("train")
        pronouns = [r for r in rows if r.get("message_group") == "addressee_pronoun"]
        self.assertEqual(Counter(r["owner_pronoun"] for r in pronouns), {"her": 16, "his": 16, "their": 16})
        for r in pronouns:
            self.assertIn(r["owner_pronoun"] + " " + r["object"], r["source"])
            self.assertIn("your " + r["object"], r["answer"])
            self.assertTrue(r["prompt"].startswith("Write a quick note to ")
                            or r["prompt"].startswith("Send ")
                            or r["prompt"].startswith("Draft a brief message for ")
                            or r["prompt"].startswith("Write " + r["recipient"]))
        paired = [r for r in rows if r.get("message_group") == "modal_contrast"]
        groups = {}
        for r in paired:
            groups.setdefault(r["pair_id"], []).append(r)
        self.assertEqual(len(groups), 8)
        for pair in groups.values():
            self.assertEqual({r["force"] for r in pair}, {"should", "will"})
            self.assertEqual(len({r["source"].replace("should", "MODAL").replace("will", "MODAL") for r in pair}), 1)

    @unittest.skipIf(W is None, "Implementation not created yet")
    def test_unchanged_or_action_changing_targets_are_rejected(self):
        row = copy.deepcopy(W.writing("train")[0])
        row["answer"] = row["source"]
        with self.assertRaises(ValueError):
            W.validate_reference(row)
        row = copy.deepcopy(next(r for r in W.writing("train") if r["family"] == "shortening"))
        row["answer"] = row["answer"].replace(row["action_verb"], "ensure")
        with self.assertRaises(ValueError):
            W.validate_reference(row)

    @unittest.skipIf(W is None, "Implementation not created yet")
    def test_development_and_final_holdouts_cannot_enter_training(self):
        dev = W.writing("dev")
        self.assertEqual(len(dev), 64)
        with self.assertRaises(ValueError):
            W.sft_bytes(dev)
        with self.assertRaises(ValueError):
            W.writing("final_holdout")
        train = W.writing("train")
        W.audit(train, dev, [], [])
        with self.assertRaises(ValueError):
            W.audit(train, dev, [train[0]], [])
        with self.assertRaises(ValueError):
            W.audit(train, dev, [], [train[0]["source"]])

    @unittest.skipIf(W is None, "Implementation not created yet")
    def test_retention_rows_are_the_existing_training_only_rows(self):
        import ember_writing_repair1_data as D
        original = D.retention("train")
        rows = W.build_train_rows(original)
        retained = [r for r in rows if r["family"] not in ("recipient", "shortening")]
        self.assertEqual(sorted((r["prompt"], r["answer"]) for r in retained),
                         sorted((r["prompt"], r["answer"]) for r in original))

    @unittest.skipIf(W is None, "Implementation not created yet")
    def test_training_sources_do_not_share_benchmark_fourgrams(self):
        records = json.loads((ROOT / "reports/evidence/writing-repair5-20261003/evidence/baseline-744.json").read_text())["records"]
        sources = [W.shortening_source(r["row"]) for r in records if W.shortening_source(r["row"])]
        self.assertEqual(len(sources), 30)
        W.audit(W.writing("train"), W.writing("dev"), [], sources)

    def test_saved_benchmark_shortening_metric_remains_compatible(self):
        records = json.loads((ROOT / "reports/evidence/writing-repair5-20261003/evidence/baseline-744.json").read_text())["records"]
        pairs = [(W.shortening_source(r["row"]), r["output"]) for r in records if W.shortening_source(r["row"])]
        stats = W.length_stats(pairs)
        self.assertEqual(stats["rows"], 30)
        self.assertEqual(stats["verbatim_copy"], 5)
        self.assertEqual(W.length_stats([("Please bring the bag.", "Please bring the bag.")])["not_shorter"], 1)

    @unittest.skipIf(W is None, "Implementation not created yet")
    def test_launch_rejects_retries_and_existing_candidate(self):
        env = {"GITHUB_REF": "refs/heads/" + T.BRANCH, "GITHUB_RUN_ATTEMPT": "1", "WR6_CODE_COMMIT": "a" * 40, "HF_TOKEN": "test"}
        T.validate_launch(env, False)
        with self.assertRaises(ValueError):
            T.validate_launch({**env, "GITHUB_RUN_ATTEMPT": "2"}, False)
        with self.assertRaises(ValueError):
            T.validate_launch(env, True)
        receipt = {"status": "PASS", "output_repo": T.OUT, "source_model": T.SOURCE,
                   "source_revision": T.SOURCE_REV, "training_started": False}
        T.validate_storage([".gitattributes", "storage-preflight.json"], receipt)
        with self.assertRaises(ValueError):
            T.validate_storage(["storage-preflight.json", "candidate/adapter_model.safetensors"], receipt)
        with self.assertRaises(ValueError):
            T.validate_storage(["storage-preflight.json"], {**receipt, "source_revision": "b" * 40})

    @unittest.skipIf(W is None, "Implementation not created yet")
    def test_answer_loss_masks_prompt_and_refuses_truncation(self):
        self.assertEqual(T.encode_ids([1, 2], [3], 4, 4),
                         {"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]})
        with self.assertRaises(ValueError):
            T.encode_ids([1, 2], [3], 4, 3)

    @unittest.skipIf(W is None, "Implementation not created yet")
    def test_data_sha_matches_pinned_trainer_constant(self):
        import hashlib
        sha = hashlib.sha256((ROOT / "jobs/ember_writing_repair6_data.py").read_bytes()).hexdigest()
        self.assertEqual(sha, T.DATA_SHA)


if __name__ == "__main__":
    unittest.main()
