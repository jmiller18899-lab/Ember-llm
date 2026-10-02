"""Integrity and summary contracts for the committed WR4 evidence and manual review."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'reports/evidence/writing-repair4-20260930'


def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'jobs'/(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WR4Review(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review=load('ember_writing_repair4_review')
        cls.summary=cls.review.summarize()

    def test_evidence_matches_manifest(self):
        for line in (EVIDENCE/'SHA256SUMS').read_text().splitlines():
            digest,name=line.split('  ',1)
            self.assertEqual(hashlib.sha256((EVIDENCE/name).read_bytes()).hexdigest(),digest,name)

    def test_saved_diagnostics_match_regenerated_dev_split(self):
        regenerated={r['id']:(r['prompt'],r['source'],r['answer']) for r in load('ember_writing_repair4_data').writing('dev')}
        for name in ('dev-before.json','dev-after.json'):
            saved={r['id']:(r['prompt'],r['source'],r['answer']) for r in self.review.load(name)}
            self.assertEqual(saved,regenerated)

    def test_benchmark_totals_match_saved_report(self):
        report=self.review.load('final-report.json')
        for label,expected in (('before',(714,26,4)),('after',(713,27,4))):
            totals=tuple(sum(s[k] for s in report[label].values()) for k in ('pass','fail','review'))
            self.assertEqual(totals,expected)

    def test_manual_review_shows_no_diagnostic_gain(self):
        s=self.summary
        self.assertEqual((s['changed'],s['greeting_only_changes']),(23,10))
        self.assertEqual(s['families']['before'],{'shortening':{'pass':17,'fail':15},'recipient':{'pass':28,'fail':1,'review':3}})
        self.assertEqual(s['families']['after'],{'shortening':{'pass':17,'fail':15},'recipient':{'pass':27,'fail':1,'review':4}})
        self.assertEqual(s['transitions'],{'pass->pass':11,'fail->fail':9,'fail->pass':1,'pass->fail':1,'pass->review':1})

    def test_shortening_regression_is_source_copying(self):
        s=self.summary
        self.assertEqual([s['dev_shortening_length'][k]['verbatim_copy'] for k in ('before','after')],[1,4])
        bench=s['benchmark_shortening_length']
        self.assertEqual([bench[k]['rows'] for k in ('before','after')],[30,30])
        self.assertEqual([bench[k]['verbatim_copy'] for k in ('before','after')],[5,8])
        self.assertEqual(set(bench['after']['verbatim_ids'])-set(bench['before']['verbatim_ids']),
                         {'promo_v2/draft-short-00','promo_v2/draft-short-05','fresh_writing/fresh-shorten-03'})


if __name__=='__main__':
    unittest.main()
