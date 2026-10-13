"""Contracts for the controlled WR13 run and its WR11 comparison."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'jobs'))
import ember_writing_repair12_train as PARENT
import ember_writing_repair13_data as W


def snapshot(step, statuses, reasons=None, copies=None):
    rows = [dict(id=str(i), status=s, reasons=(reasons or {}).get(i, [])) for i, s in enumerate(statuses)]
    return dict(step=step, copies=len(copies or []), copy_ids=copies or [],
                force_fails=sum(any(r.startswith('force_changed') for r in x['reasons']) for x in rows),
                v3_pass=statuses.count('pass'), meaning=dict(rows=rows))


class WR13Contracts(unittest.TestCase):
    def retention(self):
        # The historical WR1 release tests intentionally patch the shared base
        # vocabulary. WR13 must use the original frozen retention fixture.
        spec = importlib.util.spec_from_file_location('wr13_original_retention_test', ROOT/'jobs/ember_writing_repair1_data.py')
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        return original.retention('train')

    def trainer(self):
        name = 'ember_writing_repair13_train'
        self.assertIsNotNone(importlib.util.find_spec(name), 'WR13 trainer is missing')
        return importlib.import_module(name)

    def test_configuration_retains_the_approved_schedule(self):
        T = self.trainer()
        self.assertEqual((T.LR, T.STEPS, T.ACCUM, T.MAX_LEN, T.SEED, T.SEMANTIC_WEIGHT),
                         (PARENT.LR, PARENT.STEPS, PARENT.ACCUM, PARENT.MAX_LEN, PARENT.SEED, PARENT.SEMANTIC_WEIGHT))
        self.assertNotEqual(T.OUT, PARENT.OUT)
        self.assertFalse(T.AUTO_RETRY or T.AUTO_PROMOTION)

    def test_training_payload_is_the_exact_approved_preparation(self):
        T = self.trainer()
        rows = W.build_train_rows(self.retention())
        raw = T.validate_training_payload(rows)
        self.assertEqual(len(raw.splitlines()), 416)
        altered = copy.deepcopy(rows)
        altered[0]['answer'] += ' altered'
        with self.assertRaises(ValueError):
            T.validate_training_payload(altered)

    def test_standalone_loader_stages_the_shared_parent_before_new_data(self):
        T = self.trainer()
        with tempfile.TemporaryDirectory() as folder:
            frozen, data, checker = T.load_experiment_modules(Path(folder),
                (ROOT/'jobs/ember_writing_repair2_train.py').read_bytes(),
                (ROOT/'jobs/ember_writing_repair13_data.py').read_bytes(),
                (ROOT/'jobs/ember_meaning_preservation_v3.py').read_bytes(),
                (ROOT/'jobs/ember_drafting_repair_candidates_eval.py').read_bytes())
            rows = data.build_train_rows(self.retention())
            self.assertEqual(len(data.encode_examples(rows)), 416)
            self.assertEqual(checker.GRADER_VERSION, 'meaning-preservation-v3')

    def test_target_gain_cannot_lose_a_wr11_pass(self):
        T = self.trainer()
        base = snapshot(0, ['pass', 'fail', 'fail'], {1:['reminder_dropped']})
        wr11 = snapshot(0, ['pass', 'fail', 'pass'], {1:['reminder_dropped']})
        loses_wr11 = snapshot(13, ['pass', 'pass', 'fail'])
        fixed = snapshot(26, ['pass', 'pass', 'pass'])
        self.assertEqual(T.choose_selected(base, [base, loses_wr11], wr11), 0)
        self.assertEqual(T.choose_selected(base, [base, loses_wr11, fixed], wr11), 26)

    def test_uploaded_trainer_validates_payload_without_repository_imports(self):
        # Jobs uploads only the trainer. An isolated process must stage and
        # validate real data without finding sibling modules in the repository.
        with tempfile.TemporaryDirectory() as folder:
            uploaded = Path(folder) / 'trainer.py'
            uploaded.write_bytes((ROOT/'jobs/ember_writing_repair13_train.py').read_bytes())
            script = '''
import importlib.util, pathlib, sys
root, uploaded = map(pathlib.Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location('uploaded_trainer', uploaded)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
# Replace only remote source transport with the corresponding real fixtures.
trainer.read_source = lambda rel, *args: (root/rel).read_bytes()
work = uploaded.parent/'staged'
work.mkdir()
engine, data, checker = trainer.load_experiment_modules(work,
    (root/'jobs/ember_writing_repair2_train.py').read_bytes(),
    (root/'jobs/ember_writing_repair13_data.py').read_bytes(),
    (root/'jobs/ember_meaning_preservation_v3.py').read_bytes(),
    (root/'jobs/ember_drafting_repair_candidates_eval.py').read_bytes())
retention = trainer.load_module('original_retention',
    (root/'jobs/ember_writing_repair1_data.py').read_bytes(), work)
rows = data.build_train_rows(retention.retention('train'))
payload = trainer.validate_training_payload(rows)
assert len(payload.splitlines()) == 416
print('ISOLATED_WR13_PAYLOAD_PASS')
'''
            result = subprocess.run([sys.executable, '-I', '-c', script, str(ROOT), str(uploaded)],
                                    cwd=folder, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('ISOLATED_WR13_PAYLOAD_PASS', result.stdout)

    def test_wr11_controls_are_checked_case_by_case_for_new_critical_errors(self):
        T = self.trainer()
        base = snapshot(0, ['fail', 'fail'], {0:['reminder_dropped'],1:['force_changed:recommendation->requirement']})
        wr11 = snapshot(0, ['fail', 'fail'], {0:['reminder_dropped'],1:['not_shorter']})
        candidate = snapshot(13, ['pass', 'fail'], {1:['force_changed:recommendation->requirement']})
        self.assertEqual(T.choose_selected(base, [base, candidate], wr11), 0)

    def test_private_storage_reservation_refuses_existing_training(self):
        T = self.trainer()
        receipt = dict(status='PASS', output_repo=T.OUT, source_model=T.SOURCE,
                       source_revision=T.SOURCE_REV, training_started=False)
        T.validate_storage(['.gitattributes', 'storage-preflight.json'], receipt)
        with self.assertRaises(ValueError):
            T.validate_storage(['storage-preflight.json', 'launch.json'], receipt)

    def test_launch_accepts_only_the_new_pinned_branch(self):
        T = self.trainer()
        env = dict(WR13_CODE_COMMIT='a'*40, HF_TOKEN='test-only', GITHUB_REF='refs/heads/'+T.BRANCH, GITHUB_RUN_ATTEMPT='1')
        self.assertTrue(T.validate_launch(env, False))
        with self.assertRaises(ValueError):
            T.validate_launch(dict(env, GITHUB_REF='refs/heads/'+PARENT.BRANCH), False)
        with self.assertRaises(ValueError):
            T.validate_launch(env, True)

    def test_benchmark_controls_require_the_recorded_scores_and_case_coverage(self):
        T = self.trainer()
        baseline = [dict(suite='suite', id=str(i), ok=i<714) for i in range(744)]
        wr11 = [dict(r, ok=i<715) for i, r in enumerate(baseline)]
        T.validate_benchmark_controls(baseline, wr11)
        with self.assertRaises(ValueError):
            T.validate_benchmark_controls(baseline, baseline)
        with self.assertRaises(ValueError):
            T.validate_benchmark_controls(baseline, wr11[:-1]+[wr11[0]])

    def test_qualification_detects_lost_wr11_pass_even_with_an_equal_total(self):
        T = self.trainer()
        baseline = [dict(suite='suite', id=str(i), ok=i<714) for i in range(744)]
        wr11 = [dict(r, ok=i<715) for i, r in enumerate(baseline)]
        candidate = [dict(r, ok=i<714 or i==715) for i, r in enumerate(baseline)]
        q = T.qualify_benchmark(candidate, baseline, wr11, selected_step=13, benchmark_copies=0)
        self.assertFalse(q['qualifies'])
        self.assertIn(('suite','714'), [tuple(x) for x in q['lost_wr11_passes']])
        self.assertTrue(T.qualify_benchmark(wr11, baseline, wr11, selected_step=13, benchmark_copies=0)['qualifies'])
        self.assertFalse(T.qualify_benchmark(wr11, baseline, wr11, selected_step=0, benchmark_copies=0)['qualifies'])
        self.assertFalse(T.qualify_benchmark(wr11, baseline, wr11, selected_step=13, benchmark_copies=6)['qualifies'])


if __name__ == '__main__':
    unittest.main()
