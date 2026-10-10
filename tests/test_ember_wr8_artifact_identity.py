"""Exercise the saved-weight/evaluated-score boundary without a GPU."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jobs'))
import ember_writing_repair8_train as T


class ArtifactIdentityTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(getattr(T, "evaluation_artifact", None)), "Selected weights lack an explicit identity")
        self.assertTrue(callable(getattr(T, "evaluate_selected", None)), "Evaluation still uses final training step")

    def test_selected_64_points_to_its_bytes_not_final_128(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            selected = work / 'training/checkpoint-64'
            selected.mkdir(parents=True)
            (selected / 'adapter_model.safetensors').write_bytes(b'selected-64')
            (work / 'candidate').mkdir()
            (work / 'candidate/adapter_model.safetensors').write_bytes(b'final-128')
            artifact = T.evaluation_artifact(work, 64, 128, 'a' * 40)
            self.assertEqual(artifact['step'], 64)
            self.assertEqual(artifact['subfolder'], 'checkpoints/step-64')
            self.assertEqual(artifact['revision'], 'a' * 40)
            self.assertEqual(artifact['adapter_sha256'], hashlib.sha256(b'selected-64').hexdigest())

    def test_final_and_baseline_selections_have_unambiguous_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            for step, local, remote in [(0, 'checkpoint-0', 'checkpoint-0'), (128, 'candidate', 'candidate')]:
                folder = work / local
                folder.mkdir()
                (folder / 'adapter_model.safetensors').write_bytes(str(step).encode())
                artifact = T.evaluation_artifact(work, step, 128, 'b' * 40)
                self.assertEqual(artifact['step'], step)
                self.assertEqual(artifact['subfolder'], remote)

    def test_missing_selected_weights_fail_before_publishing_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                T.evaluation_artifact(Path(tmp), 64, 128, 'c' * 40)

    def test_selected_step_is_used_for_meaning_and_benchmark_artifacts(self):
        artifact = {'step': 64, 'subfolder': 'checkpoints/step-64', 'revision': 'd' * 40}
        uploads = {}
        rows = [{'id': 'fresh-only', 'output': 'short'}]
        scores = {'fresh': {'pass': 1}}
        def meaning(step, outputs):
            return {'step': step, 'count': len(outputs)}
        dev, after, records, snapshot = T.evaluate_selected(
            artifact, lambda: rows, meaning, lambda: (scores, rows), uploads.__setitem__)
        self.assertEqual(snapshot['step'], 64)
        self.assertEqual(uploads['evidence/candidate-744.json']['evaluated_checkpoint'], artifact)
        self.assertEqual(uploads['evidence/meaning-after.json']['evaluated_checkpoint'], artifact)
        self.assertEqual(uploads['evidence/evaluation-artifact.json'], artifact)
        self.assertEqual(uploads['evidence/dev-after.json'], rows)
        self.assertEqual(after, scores)


if __name__ == '__main__':
    unittest.main()
