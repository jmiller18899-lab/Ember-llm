"""Training isolation, loss masking, persistence and comparison contracts."""
import importlib.util
import json
import hashlib
import math
import sys
import types
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'jobs/ember_writing_repair1_train.py'
if PATH.exists():
    spec=importlib.util.spec_from_file_location('wr1_train_tested',PATH)
    W=importlib.util.module_from_spec(spec);spec.loader.exec_module(W)
else:
    W=types.SimpleNamespace()

class TrainingContracts(unittest.TestCase):
    def api(self, name):
        self.assertTrue(callable(getattr(W,name,None)),f'Missing training contract: {name}')
        return getattr(W,name)
    def rows(self):
        return [{'id':f'wr1-train-example-{i:03}', 'prompt':f'Question {i}', 'answer':str(i)} for i in range(512)]
    def test_correct_training_rows(self):
        self.assertEqual(len(self.api('validate_rows')(self.rows())),512)
    def test_holdout_row_rejected(self):
        r=self.rows();r[0]['id']='wr1-writing_holdout-0'
        with self.assertRaises(ValueError):self.api('validate_rows')(r)
    def test_duplicate_rejected(self):
        r=self.rows();r[1]=dict(r[0])
        with self.assertRaises(ValueError):self.api('validate_rows')(r)
    def test_metadata_not_training(self):
        r=self.rows();r[0]['rubric']={'answer':'leak'}
        with self.assertRaises(ValueError):self.api('validate_rows')(r)
    def test_wrong_count(self):
        with self.assertRaises(ValueError):self.api('validate_rows')(self.rows()[:-1])
    def test_empty_answer_rejected(self):
        r=self.rows();r[0]['answer']=' '
        with self.assertRaises(ValueError):self.api('validate_rows')(r)
    def test_digest_mismatch_rejected(self):
        with self.assertRaises(ValueError):self.api('verify_bytes')(b'bad','0'*64)
    def test_digest_matches(self):
        raw=b'content'; self.assertEqual(self.api('verify_bytes')(raw,hashlib.sha256(raw).hexdigest()),raw)
    def test_answer_only_mask_and_end_token(self):
        x=self.api('encode_ids')([1,2,3],[4,5],99,256)
        self.assertEqual(x,{'input_ids':[1,2,3,4,5,99],'labels':[-100,-100,-100,4,5,99]})
    def test_no_silent_truncation(self):
        with self.assertRaises(ValueError):self.api('encode_ids')([1]*255,[4],99,256)
    def test_missing_end_token_rejected(self):
        with self.assertRaises(ValueError):self.api('encode_ids')([1],[2],None,256)
    def test_padding_mask(self):
        x=self.api('pad_batch')([{'input_ids':[1,2,99],'labels':[-100,2,99]},{'input_ids':[1,3,4,99],'labels':[-100,3,4,99]}],99)
        self.assertEqual(x['input_ids'][0],[1,2,99,99])
        self.assertEqual(x['labels'][0],[-100,2,99,-100])
        self.assertEqual(x['attention_mask'][0],[1,1,1,0])
    def test_empty_batch_rejected(self):
        with self.assertRaises(ValueError):self.api('pad_batch')([],99)
    def test_original_adapter_cannot_be_output(self):
        with self.assertRaises(ValueError):self.api('validate_output_repo')('Jmiller18899/ember-qwen3.5-4b-repair2')
    def test_only_agreed_destination_allowed(self):
        self.api('validate_output_repo')('Jmiller18899/ember-qwen3.5-4b-writing-repair1-20260924')
    def test_one_epoch_is_128_updates(self):
        self.assertEqual(self.api('expected_steps')(),128)
    def test_completed_training_requires_changed_adapter(self):
        with self.assertRaises(ValueError):self.api('verify_training')(128,0.1,'same','same')
    def test_completed_training_requires_finite_loss(self):
        with self.assertRaises(ValueError):self.api('verify_training')(128,float('nan'),'before','after')
    def test_completed_training_requires_all_updates(self):
        with self.assertRaises(ValueError):self.api('verify_training')(127,0.1,'before','after')
    def test_training_complete(self):
        self.assertTrue(self.api('verify_training')(128,0.1,'before','after'))
    def test_no_regressions_hidden_by_equal_totals(self):
        f=self.api('compare_records')
        before=[{'suite':'s','id':'a','ok':True},{'suite':'s','id':'b','ok':False}]
        after=[{'suite':'s','id':'a','ok':False},{'suite':'s','id':'b','ok':True}]
        result=f(before,after)
        self.assertEqual(result['regressions'],[['s','a']])
        self.assertFalse(result['automatic_promotion'])
    def test_comparison_missing_rows_rejected(self):
        f=self.api('compare_records')
        with self.assertRaises(ValueError):f([{'suite':'s','id':'a','ok':True}],[])
    def test_comparison_duplicate_rows_rejected(self):
        f=self.api('compare_records');r={'suite':'s','id':'a','ok':True}
        with self.assertRaises(ValueError):f([r,r],[r])
    def test_launch_scope_excludes_holdouts(self):
        x=self.api('run_spec')()
        self.assertEqual(x['training_rows'],512)
        self.assertFalse(x['final_holdouts_evaluated'])
        self.assertFalse(x['automatic_promotion'])
        self.assertEqual(x['epochs'],1)
        self.assertEqual(x['learning_rate'],1.5e-6)

if __name__=='__main__':unittest.main()
