import importlib.util
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "jobs" / "ember_direct_answer_audit.py"
spec = importlib.util.spec_from_file_location("audit", SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

class Tokenizer:
    def decode(self, ids, skip_special_tokens=True):
        pieces = {1: " 21 ", 2: "25", 3: "<|im_end|>", 4: "<think>", 5: "48"}
        return "".join(pieces[i] for i in ids if not (skip_special_tokens and i in {3, 4}))

class GenerationEvidenceTests(unittest.TestCase):
    def test_keeps_raw_tokens_decoding_and_graded_text(self):
        r = audit.generation_record(Tokenizer(), [1, 3], 96, [3])
        required = {"raw_token_ids", "raw_decoded", "decoded_text", "output", "generated_tokens", "stop_reason", "hit_token_limit", "ended_with_eos"}
        self.assertTrue(required <= set(r), required - set(r))
        self.assertEqual(r["raw_token_ids"], [1, 3])
        self.assertEqual(r["raw_decoded"], " 21 <|im_end|>")
        self.assertEqual(r["decoded_text"], " 21 ")
        self.assertEqual(r["output"], "21")

    def test_eos_at_limit_is_not_a_cutoff(self):
        r = audit.generation_record(Tokenizer(), [2, 3], 2, [3])
        self.assertEqual(r.get("stop_reason"), "eos")
        self.assertIs(r.get("hit_token_limit"), True)
        self.assertIs(r.get("ended_with_eos"), True)

    def test_limit_without_eos_is_cutoff(self):
        r = audit.generation_record(Tokenizer(), [2, 5], 2, [3])
        self.assertEqual(r.get("stop_reason"), "length")
        self.assertIs(r.get("ended_with_eos"), False)

    def test_short_stop_without_eos_is_unknown(self):
        r = audit.generation_record(Tokenizer(), [2], 96, [3])
        self.assertEqual(r.get("stop_reason"), "unknown")

    def test_special_tokens_not_silently_lost(self):
        r = audit.generation_record(Tokenizer(), [4, 1, 3], 96, [3])
        self.assertIn("<think>", r.get("raw_decoded", ""))
        self.assertEqual(r["output"], "21")

    def test_invalid_limit_rejected(self):
        with self.assertRaises(ValueError):
            audit.generation_record(Tokenizer(), [1], 0, [3])

    def test_more_tokens_than_limit_rejected(self):
        with self.assertRaises(ValueError):
            audit.generation_record(Tokenizer(), [1, 2], 1, [3])

class PairingTests(unittest.TestCase):
    def row(self, i="x", prompt="Number only."):
        return {"id": i, "prompt": prompt, "family": "arithmetic", "suite": "regression", "scoring": "exact", "answer": "21", "rubric": None, "output": "21", "exact_match": True}

    def test_reorders_by_id_not_array_position(self):
        a, b = self.row("a"), self.row("b")
        pairs = audit.pair_saved([a, b], [b, a])
        self.assertEqual([x["id"] for x, _ in pairs], [y["id"] for _, y in pairs])

    def test_rejects_missing_rows(self):
        with self.assertRaises(ValueError):
            audit.pair_saved([self.row()], [])

    def test_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            audit.pair_saved([self.row(), self.row()], [self.row(), self.row()])

    def test_rejects_prompt_change(self):
        with self.assertRaises(ValueError):
            audit.pair_saved([self.row()], [self.row(prompt="Different question")])

    def test_rejects_expected_answer_change(self):
        a, b = self.row(), self.row()
        b["answer"] = "25"
        with self.assertRaises(ValueError):
            audit.pair_saved([a], [b])

    def test_rejects_bad_saved_score(self):
        a, b = self.row(), self.row()
        b["exact_match"] = False
        with self.assertRaises(ValueError):
            audit.pair_saved([a], [b])

if __name__ == "__main__":
    unittest.main()
