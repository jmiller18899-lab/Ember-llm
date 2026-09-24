"""Tests for dataset construction and audit, never training examples."""
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'jobs/ember_writing_repair1_release.py'
if not PATH.exists(): PATH = ROOT / 'jobs/ember_writing_repair1_data.py'
sys.path.insert(0,str(ROOT/'jobs'))

class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = None
        if PATH.exists():
            spec = importlib.util.spec_from_file_location('wr1_data', PATH)
            cls.module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = cls.module
            spec.loader.exec_module(cls.module)

    def data(self):
        self.assertIsNotNone(self.module, 'Curriculum builder must exist before this contract can pass')
        return self.module.build_splits()

    def test_exact_split_counts(self):
        self.assertEqual({k: len(v) for k,v in self.data().items()},
            {'train':512, 'dev':64, 'writing_holdout':96, 'model_holdout':96})

    def test_training_recipe(self):
        self.assertEqual(Counter(r['family'] for r in self.data()['train']),
            {'shortening':160, 'recipient':96, 'arithmetic':64, 'extraction':64,
             'grounding':48, 'clarification':48, 'direct':32})

    def test_balanced_writing_holdout(self):
        self.assertEqual(Counter(r['family'] for r in self.data()['writing_holdout']),
                         {'shortening':48, 'recipient':48})

    def test_deterministic_generation(self):
        self.assertEqual(self.data(), self.data())

    def test_all_references_have_independent_checks(self):
        for split, rows in self.data().items():
            for r in rows:
                with self.subTest(id=r['id']):
                    self.assertEqual(self.module.validate_reference(r), [])
                    self.assertTrue(r['rubric'])
                    self.assertEqual(r['training_allowed'], split=='train')

    def test_audit_positive_package(self):
        d=self.data()
        result = self.module.audit_splits(d)
        self.assertTrue(result['passed'], result)

    def test_duplicate_id_rejected(self):
        d=self.data(); d['train'][1]['id']=d['train'][0]['id']
        self.assertFalse(self.module.audit_splits(d)['passed'])

    def test_cross_split_prompt_leak_rejected(self):
        d=self.data(); d['dev'][0]['prompt']=d['train'][0]['prompt']
        self.assertFalse(self.module.audit_splits(d)['passed'])

    def test_cross_split_template_leak_rejected(self):
        d=self.data(); d['dev'][0]['template_id']=d['train'][0]['template_id']
        self.assertFalse(self.module.audit_splits(d)['passed'])

    def test_reference_mutations_rejected(self):
        d=self.data()
        for family in ['shortening','recipient','arithmetic','extraction','grounding','clarification','direct']:
            r=copy.deepcopy(next(x for x in d['train'] if x['family']==family))
            r['answer']='I completed and sent everything. 999999'
            with self.subTest(family=family):
                self.assertTrue(self.module.validate_reference(r))

    def test_request_cannot_become_requirement(self):
        r=copy.deepcopy(next(r for r in self.data()['train'] if r['family']=='shortening' and r['rubric'].get('force')=='request' and ', please ' in r['answer']))
        r['answer']=r['answer'].replace(', please ', ' must ')
        self.assertTrue(self.module.validate_reference(r))

    def test_other_owner_cannot_become_recipient(self):
        r=copy.deepcopy(next(r for r in self.data()['train'] if r['family']=='recipient' and r['rubric']['owner_relation']=='third_party'))
        r['answer']=r['answer'].replace(r['rubric']['owner']+"'s", 'your')
        self.assertTrue(self.module.validate_reference(r))

    def test_sender_stays_first_person(self):
        r=copy.deepcopy(next(r for r in self.data()['train'] if r['family']=='recipient'))
        r['answer']=r['answer'].replace('I ', 'You ')
        self.assertTrue(self.module.validate_reference(r))

    def test_shortening_cannot_drop_fact(self):
        r=copy.deepcopy(next(r for r in self.data()['train'] if r['family']=='shortening' and 'exactly ' in r['answer']))
        r['answer']=r['answer'].replace('exactly ', '')
        self.assertTrue(self.module.validate_reference(r))

    def test_holdout_cannot_be_loaded_for_training(self):
        d=self.data()
        self.assertEqual(len(self.module.training_rows(d['train'])),512)
        with self.assertRaises(ValueError): self.module.training_rows(d['writing_holdout'])
        with self.assertRaises(ValueError): self.module.training_rows(d['train']+d['dev'])

    def test_export_contains_no_rejected_answers_in_training(self):
        d=self.data()
        for r in self.module.training_rows(d['train']):
            self.assertEqual(set(r),{'id','prompt','answer'})
            self.assertIsInstance(r['answer'],str)
            self.assertNotIn('rejected',r)

    def test_benchmark_overlap_is_rejected(self):
        d=self.data(); p=d['train'][0]['prompt']
        self.assertFalse(self.module.audit_splits(d, external_prompts=[p])['passed'])

    def test_complete_export_and_checksums(self):
        d=self.data()
        with tempfile.TemporaryDirectory() as tmp:
            manifest=self.module.write_package(Path(tmp),d)
            for name,entry in manifest['files'].items():
                p=Path(tmp)/name
                self.assertTrue(p.is_file())
                self.assertEqual(self.module.sha256(p.read_bytes()),entry['sha256'])

    def test_actual_structure_overlap_rejected_after_slot_renaming(self):
        d=self.data()
        r=copy.deepcopy(next(r for r in d['train'] if r['family']=='shortening'))
        r['id']='deliberate-dev-structure-leak'; r['split']='dev'; r['training_allowed']=False
        r['template_id']='dev/new-looking-id'; r['structural_signature']='fabricated-unrelated-signature'
        r['contrast_group']='dev/new-group'
        # Same structure with a different entity must not bypass a content audit.
        r['prompt']=r['prompt'].replace('All ', 'All  ').replace('all ', 'all  ')
        d['dev'][0]=r
        self.assertFalse(self.module.audit_splits(d)['passed'])

    def test_arithmetic_against_hand_calculated_controls(self):
        self.data()
        r={'a':13,'b':5,'c':3,'total':65,'start_minutes':1425,'duration':35}
        controls={'add_sub':'15','mult_add':'68','mult_sub':'62','divide_add':'8',
                  'double_sum':'78','perimeter':'36','signed':'-11','time':'00:20'}
        for operation, answer in controls.items():
            with self.subTest(operation=operation):
                self.assertEqual(self.module.arithmetic_expected(dict(r,operation=operation)),answer)

    def test_source_and_template_not_random_split(self):
        d=self.data()
        signatures={s:{self.module.prompt_skeleton(r.get('source',r['prompt'])) for r in rows} for s,rows in d.items()}
        for a in d:
            for b in d:
                if a<b: self.assertFalse(signatures[a]&signatures[b],(a,b,signatures[a]&signatures[b]))

    def test_holdout_classifier_covers_both_classes(self):
        rows=self.data()['model_holdout']
        self.assertEqual({r['answer'] for r in rows if r['rubric']['task']=='classification'},{'HIGH','LOW'})

    def test_plural_group_references_are_grammatical(self):
        d=self.data()
        for r in d['dev']+d['writing_holdout']:
            if r['family']=='shortening':
                self.assertNotRegex(r['answer'],r'\b(?:editors|inspectors|weavers|ceramists|librarians|cartographers|restorers|ushers|docents|cataloguers) member\b')

    def test_holdout_ownership_prompt_does_not_use_wrong_article(self):
        d=self.data()
        for r in d['writing_holdout']:
            self.assertNotIn('a ocarina',r['prompt'])
            self.assertNotIn('a abacus',r['prompt'])

if __name__=='__main__': unittest.main()
