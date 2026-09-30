"""Data isolation and semantic-corruption contracts for the WR4 experiment."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import unittest
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]


def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'jobs'/(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WritingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w=load('ember_writing_repair4_data')
        cls.d=load('ember_writing_repair1_data')
        cls.prior=load('ember_writing_repair3_data')
        cls.train=cls.w.build_train_rows(cls.d.retention('train'))
        cls.dev=cls.w.writing('dev')

    def test_balanced_composition_and_preserved_retention(self):
        self.assertEqual(Counter(r['family'] for r in self.train),
            {'shortening':128,'recipient':128,'arithmetic':64,'extraction':64,
             'grounding':48,'clarification':48,'direct':32})
        self.assertEqual(Counter(r['family'] for r in self.dev),{'shortening':32,'recipient':32})
        source={r['id']:(r['prompt'],r['answer']) for r in self.d.retention('train')}
        kept={r['id'].removeprefix('wr4-retain-'):(r['prompt'],r['answer'])
              for r in self.train if r['id'].startswith('wr4-retain-')}
        self.assertEqual(kept,source)

    def test_train_dev_and_observed_writing_are_separate(self):
        historical=self.d.writing('train')+self.d.writing('dev')+self.prior.writing('train')+self.prior.writing('dev')
        self.assertEqual(self.w.audit(self.train,self.dev,historical)['exact_prompt_source_overlaps'],0)
        with self.assertRaises(ValueError):
            self.w.audit(self.train,self.dev,[next(r for r in self.train if r['family']=='shortening')])

    def test_compact_references_do_real_shortening_and_preserve_anchors(self):
        for r in self.train+self.dev:
            if r['family']!='shortening':
                continue
            with self.subTest(id=r['id']):
                self.w.validate_reference(r)
                self.assertLess(len(r['answer'].split()),len(r['source'].split()))
                self.assertLess(len(r['answer']),len(r['source']))
                self.assertEqual(re.findall(r'\d+',r['answer']),re.findall(r'\d+',r['source']))

    def test_copying_and_loss_of_time_are_rejected(self):
        r=copy.deepcopy(next(r for r in self.train if r['family']=='shortening' and re.search(r'\d',r['answer'])))
        r['answer']=r['source']
        with self.assertRaises(ValueError):
            self.w.validate_reference(r)
        r=copy.deepcopy(next(r for r in self.train if r['family']=='shortening' and re.search(r'\d',r['answer'])))
        r['answer']=re.sub(r'\d','',r['answer'])
        with self.assertRaises(ValueError):
            self.w.validate_reference(r)

    def test_each_modal_has_both_owner_relations_and_named_pronoun_forms(self):
        modal=[r for r in self.train if r.get('force')]
        self.assertEqual(len(modal),64)
        for force in ('can','should','might','will'):
            self.assertEqual(Counter((r['relation'],r['representation']) for r in modal if r['force']==force),
                             {('recipient','named'):4,('recipient','pronoun'):4,
                              ('third_party','named'):4,('third_party','pronoun'):4})
        own=[r for r in modal if r['relation']=='recipient' and r['representation']=='pronoun']
        for pronoun in ('her','his','their'):
            self.assertTrue(any(pronoun+' '+r['object'] in r['prompt'] for r in own),pronoun)

    def test_named_and_pronoun_minimal_pairs_have_identical_targets(self):
        modal=[r for r in self.train if r.get('force')]
        keyed={r['id']:r for r in modal}
        for r in modal:
            if r['representation']=='named':
                partner=keyed[r['id'].removesuffix('named')+'pronoun']
                self.assertEqual(r['answer'],partner['answer'])

    def test_can_should_might_cannot_be_strengthened_to_will(self):
        for force in ('can','should','might'):
            r=copy.deepcopy(next(r for r in self.train if r.get('force')==force))
            r['answer']=r['answer'].replace(force,'will')
            with self.subTest(force=force),self.assertRaises(ValueError):
                self.w.validate_reference(r)

    def test_recipient_perspective_and_third_party_ownership_are_protected(self):
        for relation in ('recipient','third_party'):
            r=copy.deepcopy(next(r for r in self.train if r.get('relation')==relation))
            ownership='your '+r['object'] if relation=='recipient' else r['owner']+"'s "+r['object']
            r['answer']=r['answer'].replace(ownership,'their '+r['object'])
            with self.subTest(relation=relation),self.assertRaises(ValueError):
                self.w.validate_reference(r)

    def test_speaker_negation_cannot_be_broadened_to_a_global_claim(self):
        for split in (self.train,self.dev):
            r=copy.deepcopy(next(r for r in split if r.get('negative_scope')=='speaker'))
            r['answer']=r['answer'].replace(r['negative_claim'],'it has never been moved')
            with self.assertRaises(ValueError):
                self.w.validate_reference(r)
        self.assertEqual(Counter(r.get('negative_scope') for r in self.train if r.get('negative_scope')),
                         {'speaker':32,'global':32})

    def test_export_cannot_accept_dev_duplicates_or_final_holdouts(self):
        for rows in (self.train+self.dev,self.train+[self.train[0]]):
            with self.assertRaises(ValueError):
                self.w.sft_bytes(rows)
        with self.assertRaises(ValueError):
            self.w.build_train_rows(self.d.retention('dev'))
        for split in ('test','writing_holdout','model_holdout',''):
            with self.assertRaises(ValueError):
                self.w.writing(split)
        raw=self.w.sft_bytes(self.train)
        self.assertEqual(raw,self.w.sft_bytes(self.train))
        self.assertEqual(len(raw.splitlines()),512)
        self.assertTrue(all(set(json.loads(line))=={'id','prompt','answer'} for line in raw.splitlines()))


class LaunchContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t=load('ember_writing_repair4_train')

    def test_fixed_reference_hyperparameters_and_source_checksums(self):
        self.assertEqual((self.t.SOURCE,self.t.SOURCE_REV),
            ('Jmiller18899/ember-qwen3.5-4b-repair2','daf938bba5d4e6b650ec9d34a2d3ac56706cf549'))
        self.assertNotEqual(self.t.OUT,self.t.SOURCE)
        self.assertEqual((self.t.LR,self.t.STEPS,self.t.ACCUM,self.t.MAX_LEN,self.t.SEED),
                         (7.5e-7,128,4,384,431))
        self.assertEqual((self.t.FLAVOR,self.t.TIMEOUT),('l4x1','90m'))
        self.assertFalse(self.t.AUTO_PROMOTION)
        self.assertFalse(self.t.AUTO_RETRY)
        self.assertEqual(hashlib.sha256((ROOT/'jobs/ember_writing_repair4_data.py').read_bytes()).hexdigest(),self.t.DATA_SHA)
        self.assertEqual(hashlib.sha256((ROOT/'jobs/ember_writing_repair2_train.py').read_bytes()).hexdigest(),self.t.ENGINE_SHA)

    def test_repeated_or_wrong_branch_launch_cannot_duplicate_a_paid_run(self):
        env={'GITHUB_REF':'refs/heads/'+self.t.BRANCH,'GITHUB_RUN_ATTEMPT':'1',
             'WR4_CODE_COMMIT':'a'*40,'HF_TOKEN':'test-not-a-secret'}
        self.t.validate_launch(env,False)
        with self.assertRaises(ValueError):
            self.t.validate_launch(env,True)
        for key,value in [('GITHUB_REF','refs/heads/main'),('GITHUB_RUN_ATTEMPT','2'),('HF_TOKEN','')]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.t.validate_launch(dict(env,**{key:value}),False)


if __name__=='__main__':
    unittest.main()
