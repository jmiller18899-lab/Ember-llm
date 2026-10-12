"""Check fresh prompt pairs and detect per-case losses hidden by equal totals."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'reports/probes/wr11_reminder_preflight.py'


class WR11PreflightTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.is_file(), 'WR11 has no fresh paired diagnostic')
        spec = importlib.util.spec_from_file_location('wr11_probe', MODULE)
        self.p = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.p)

    def payload(self):
        return {
            'probe_sha256': self.p.probe_sha256(),
            'checkpoint': {'repo_id': 'test/synthetic-fixture', 'revision': 'a' * 40,
                           'subfolder': 'candidate', 'adapter_sha256': 'b' * 64},
            'generation': {'do_sample': False, 'max_new_tokens': 96, 'enable_thinking': False},
            'records': [{'id': r['id'], 'output': r['reference'], 'system_prompt': 'Synthetic test only.'}
                        for r in self.p.cases()],
        }

    def test_pairs_share_source_and_reference_and_never_enter_training(self):
        rows = self.p.cases()
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({r['source_id'] for r in rows}), 12)
        for source_id in {r['source_id'] for r in rows}:
            pair = [r for r in rows if r['source_id'] == source_id]
            self.assertEqual({r['wrapper'] for r in pair}, {'neutral', 'explicit'})
            self.assertEqual(len({r['source'] for r in pair}), 1)
            self.assertEqual(len({r['reference'] for r in pair}), 1)
            self.assertTrue(all(r['training_allowed'] is False for r in pair))
        self.assertTrue(all('reference' not in r for r in self.p.prompts()))

    def test_references_pass_and_reminder_deletions_fail(self):
        for row in self.p.cases():
            self.assertEqual(self.p.grade(row, row['reference'])['status'], 'pass', row['id'])
            if row['family'] != 'control':
                grade = self.p.grade(row, row['reminder_dropped'])
                self.assertEqual(grade['status'], 'fail', row['id'])
                self.assertIn('reminder_dropped', grade['reasons'])

    def test_all_correct_has_no_regressions_or_prompt_sensitivity(self):
        result = self.p.compare(self.payload(), self.payload())
        self.assertEqual(result['baseline']['pass'], 24)
        self.assertEqual(result['candidate']['pass'], 24)
        self.assertEqual(result['regressions'], [])
        self.assertEqual(result['candidate']['prompt_sensitive_sources'], [])
        self.assertFalse(result['production_ready'])
        self.assertFalse(result['training_recommendation_established'])

    def test_equal_totals_do_not_hide_a_lost_reminder(self):
        baseline, candidate = self.payload(), self.payload()
        rows = self.p.cases()
        baseline['records'][2]['output'] = rows[2]['source']
        candidate['records'][0]['output'] = rows[0]['reminder_dropped']
        result = self.p.compare(baseline, candidate)
        self.assertEqual(result['net_pass_change'], 0)
        self.assertEqual(result['regressions'], [rows[0]['id']])
        self.assertEqual(result['improvements'], [rows[2]['id']])
        self.assertEqual(result['candidate']['prompt_sensitive_sources'], [rows[0]['source_id']])
        self.assertFalse(result['preserves_baseline_passes'])

    def test_missing_duplicate_and_unknown_prediction_ids_are_rejected(self):
        for mode in ('missing', 'duplicate', 'unknown'):
            payload = self.payload()
            if mode == 'missing':
                payload['records'].pop()
            elif mode == 'duplicate':
                payload['records'].append(copy.deepcopy(payload['records'][0]))
            else:
                payload['records'][0]['id'] = 'unknown'
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.p.compare(self.payload(), payload)

    def test_wrong_probe_or_unpinned_weights_are_rejected(self):
        for mode in ('probe', 'revision', 'digest'):
            payload = self.payload()
            if mode == 'probe':
                payload['probe_sha256'] = 'c' * 64
            elif mode == 'revision':
                payload['checkpoint']['revision'] = 'main'
            else:
                payload['checkpoint']['adapter_sha256'] = 'unknown'
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.p.compare(self.payload(), payload)

    def test_different_inference_settings_or_system_prompts_are_rejected(self):
        for mode in ('settings', 'system'):
            payload = self.payload()
            if mode == 'settings':
                payload['generation']['max_new_tokens'] = 10
            else:
                payload['records'][0]['system_prompt'] = 'Different policy'
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.p.compare(self.payload(), payload)

    def test_fresh_sources_do_not_overlap_observed_history(self):
        report = self.p.audit(ROOT)
        self.assertEqual(report['exact_prompt_source_overlaps'], 0)
        self.assertEqual(report['benchmark_records_checked'], 744)
        self.assertGreater(report['historical_rows_checked'], 0)
        self.assertFalse(report['final_holdouts_loaded'])


if __name__ == '__main__':
    unittest.main()
