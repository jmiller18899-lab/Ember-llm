"""WR3 contracts: corrupted splits, copying or changed force must fail."""
import copy
import importlib.util
from pathlib import Path
import re
import unittest
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / 'jobs' / (name + '.py')
    assert path.is_file(), 'WR3 implementation is missing: ' + path.name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WritingData(unittest.TestCase):
    def setUp(self):
        self.d = load('ember_writing_repair3_data')

    def test_composition(self):
        self.assertEqual(Counter(r['family'] for r in self.d.writing('train')),
                         {'shortening': 160, 'recipient': 96})
        self.assertEqual(Counter(r['family'] for r in self.d.writing('dev')),
                         {'shortening': 48, 'recipient': 48})

    def test_fresh_development_has_no_prompt_source_or_structure_overlap(self):
        a, b = self.d.writing('train'), self.d.writing('dev')
        for field in ('id', 'prompt', 'source', 'structure'):
            left = {self.d.norm(r[field]) for r in a if r.get(field)}
            right = {self.d.norm(r[field]) for r in b if r.get(field)}
            self.assertFalse(left & right, field)

    def test_all_ids_and_prompts_unique(self):
        for split in ('train', 'dev'):
            rows = self.d.writing(split)
            for field in ('id', 'prompt'):
                self.assertEqual(len(rows), len({self.d.norm(r[field]) for r in rows}))

    def test_substantive_shortening_and_protected_facts(self):
        for split in ('train', 'dev'):
            for r in self.d.writing(split):
                self.d.validate_reference(r)
                if r['family'] == 'shortening':
                    self.assertLessEqual(len(r['answer'].split()), .8 * len(r['source'].split()), r['id'])
                    for fact in r['protected']:
                        self.assertIn(fact.casefold(), r['answer'].casefold())
                    self.assertEqual(re.findall(r'\d+', r['source']), re.findall(r'\d+', r['answer']))

    def test_copy_is_rejected(self):
        r = copy.deepcopy(next(r for r in self.d.writing('train') if r['family'] == 'shortening'))
        r['answer'] = r['source']
        with self.assertRaises(ValueError):
            self.d.validate_reference(r)

    def test_should_to_must_is_rejected(self):
        r = copy.deepcopy(next(r for r in self.d.writing('train') if 'should' in r['protected']))
        r['answer'] = r['answer'].replace('should', 'must')
        with self.assertRaises(ValueError):
            self.d.validate_reference(r)

    def test_owner_and_recipient_are_not_interchangeable(self):
        rows = [r for r in self.d.writing('train') if r['family'] == 'recipient']
        self.assertEqual(Counter(r['relation'] for r in rows), {'recipient': 48, 'third_party': 48})
        for r in rows:
            self.assertTrue(r['answer'].startswith('Hi ' + r['recipient'] + ', '))
            expected = 'your ' + r['object'] if r['relation'] == 'recipient' else r['owner'] + "'s " + r['object']
            self.assertIn(expected, r['answer'])
            self.assertNotIn('your ' + r['object'], r['answer']) if r['relation'] == 'third_party' else None

    def test_wrong_owner_is_rejected(self):
        r = copy.deepcopy(next(r for r in self.d.writing('train') if r.get('relation') == 'recipient'))
        r['answer'] = r['answer'].replace('your ', 'her ')
        with self.assertRaises(ValueError):
            self.d.validate_reference(r)

    def test_no_final_holdout_api(self):
        for split in ('writing_holdout', 'model_holdout', 'test', ''):
            with self.assertRaises(ValueError):
                self.d.writing(split)

    def test_sft_rejects_dev_or_duplicate_rows(self):
        rows = self.d.writing('train')
        with self.assertRaises(ValueError):
            self.d.sft_bytes(rows + self.d.writing('dev'))
        with self.assertRaises(ValueError):
            self.d.sft_bytes(rows + [rows[0]])

    def test_sft_has_only_training_fields(self):
        import json
        rows = self.d.writing('train')
        raw = self.d.sft_bytes(rows)
        self.assertEqual(raw, self.d.sft_bytes(rows))
        for line in raw.splitlines():
            self.assertEqual(set(json.loads(line)), {'id', 'prompt', 'answer'})

    def test_leakage_audit_rejects_collision(self):
        rows = self.d.writing('train')
        with self.assertRaises(ValueError):
            self.d.audit(rows, [], [{'prompt': rows[0]['prompt']}])

    def test_retention_only_accepts_training_rows(self):
        with self.assertRaises(ValueError):
            self.d.build_train_rows([{'id': 'dev-0', 'split': 'dev', 'training_allowed': False}])


class TrainingContracts(unittest.TestCase):
    def setUp(self):
        self.t = load('ember_writing_repair3_train')

    def test_fixed_budget(self):
        self.assertEqual(self.t.FLAVOR, 'l4x1')
        self.assertEqual(self.t.TIMEOUT, '90m')
        self.assertEqual(self.t.LR, 7.5e-7)
        self.assertEqual(self.t.STEPS, 128)
        self.assertFalse(self.t.AUTO_PROMOTION)
        self.assertFalse(self.t.AUTO_RETRY)

    def test_source_not_latest_candidate(self):
        self.assertEqual(self.t.SOURCE, 'Jmiller18899/ember-qwen3.5-4b-repair2')
        self.assertEqual(self.t.SOURCE_REV, 'daf938bba5d4e6b650ec9d34a2d3ac56706cf549')
        self.assertNotEqual(self.t.OUT, self.t.SOURCE)

    def test_launch_rejects_duplicate_or_wrong_context(self):
        valid = {'GITHUB_REF': 'refs/heads/' + self.t.BRANCH, 'GITHUB_RUN_ATTEMPT': '1',
                 'WR3_CODE_COMMIT': 'a' * 40, 'HF_TOKEN': 'test-not-a-secret'}
        self.t.validate_launch(valid, False)
        with self.assertRaises(ValueError):
            self.t.validate_launch(valid, True)
        for field, value in [('GITHUB_REF', 'refs/heads/main'), ('GITHUB_RUN_ATTEMPT', '2'),
                             ('WR3_CODE_COMMIT', 'main'), ('HF_TOKEN', '')]:
            bad = dict(valid); bad[field] = value
            with self.assertRaises(ValueError):
                self.t.validate_launch(bad, False)

    def test_hash_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            self.t.verify(b'changed file', '0' * 64)

    def test_length_limit_never_silently_truncates(self):
        self.assertEqual(self.t.encode_ids([1, 2], [3], 4, 4),
                         {'input_ids': [1, 2, 3, 4], 'labels': [-100, -100, 3, 4]})
        with self.assertRaises(ValueError):
            self.t.encode_ids([1, 2], [3], 4, 3)


if __name__ == '__main__':
    unittest.main()
