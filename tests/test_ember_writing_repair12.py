"""WR12 semantic-anchor alignment and target-improvement safeguards."""
import importlib.util
import math
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'jobs'))
import ember_writing_repair1_data as D
import ember_writing_repair11_data as OLD
import ember_writing_repair11_train as OLD_T
import ember_writing_repair12_data as W
import ember_writing_repair12_train as T

def snapshot(step, statuses, reasons=None, copies=None):
    rows = [dict(id=str(i), status=s, reasons=(reasons or {}).get(i, [])) for i,s in enumerate(statuses)]
    return dict(step=step, copies=len(copies or []), copy_ids=copies or [],
                force_fails=sum(any(r.startswith('force_changed') for r in x['reasons']) for x in rows),
                v3_pass=statuses.count('pass'), meaning=dict(rows=rows))

class WR12Contracts(unittest.TestCase):
    def test_offsets_weight_whole_reminder_and_force_tokens(self):
        self.assertTrue(callable(getattr(T, 'anchor_mask', None)), 'semantic token weighting is missing')
        answer='Please remember all cadets must stay until Friday.'
        offsets=[m.span() for m in re.finditer(r'\S+',answer)]
        mask=T.anchor_mask(answer, offsets)
        self.assertEqual(mask,[1,1,0,0,1,0,1,0])
        self.assertEqual(T.anchor_mask('Do not forget Mira might wait.',[(0,2),(3,6),(7,13),(14,18),(19,24),(25,30)]),[1,1,1,0,1,0])

    def test_mask_rejects_missing_anchor_offsets(self):
        self.assertTrue(callable(getattr(T, 'anchor_mask', None)))
        with self.assertRaises(ValueError): T.anchor_mask('Please remember the note.',[(0,6),(16,25)])

    def test_mask_excludes_prompt_and_end_token(self):
        self.assertTrue(callable(getattr(T,'encode_anchor_mask',None)))
        self.assertEqual(T.encode_anchor_mask(3,[1,0,1]),[0,0,0,1,0,1,0])

    def test_loss_only_ablation_preserves_all_training_text(self):
        old=OLD.build_train_rows(D.retention('train'));new=W.build_train_rows(D.retention('train'))
        self.assertEqual([{k:v for k,v in r.items() if k!='id'} for r in old],
                         [{k:v for k,v in r.items() if k!='id'} for r in new])
        self.assertEqual((T.LR,T.STEPS,T.ACCUM,T.SEED),(OLD_T.LR,OLD_T.STEPS,OLD_T.ACCUM,OLD_T.SEED))

    def test_fresh_development_never_reuses_observed_sources(self):
        before={r['source'] for r in OLD.writing('dev')}
        after=W.writing('dev');self.assertEqual(len(after),56)
        self.assertFalse(before & {r['source'] for r in after})

    def test_unrelated_gain_cannot_select_checkpoint(self):
        base=snapshot(0,['pass','fail','fail'],{1:['reminder_dropped']})
        other=snapshot(13,['pass','fail','pass'],{1:['reminder_dropped']})
        self.assertEqual(T.choose_selected(base,[base,other]),0)

    def test_target_gain_is_eligible_with_pass_retention(self):
        base=snapshot(0,['pass','fail','fail'],{1:['reminder_dropped']})
        fixed=snapshot(13,['pass','pass','fail'])
        self.assertEqual(T.choose_selected(base,[base,fixed]),13)

    def test_target_gain_cannot_hide_new_copy_or_force_error(self):
        base=snapshot(0,['pass','fail','fail','fail'],{1:['reminder_dropped'],2:['not_shorter']},['2'])
        copied=snapshot(13,['pass','pass','fail','fail'],{3:['not_shorter']},['3'])
        force=snapshot(26,['pass','pass','fail','fail'],{3:['force_changed:recommendation->requirement']})
        self.assertEqual(T.choose_selected(base,[base,copied,force]),0)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch tests run in the CPU validation job')
    def test_anchor_loss_uses_causal_shift_and_only_selected_tokens(self):
        import torch
        self.assertTrue(callable(getattr(T,'semantic_nll',None)))
        logits=torch.tensor([[[0.,0.,0.,0.],[0.,3.,0.,0.],[0.,0.,3.,0.],[0.,0.,0.,3.]]],requires_grad=True)
        labels=torch.tensor([[-100,1,2,3]])
        mask=torch.tensor([[0,1,0,0]])
        loss=T.semantic_nll(logits,labels,mask)
        self.assertAlmostEqual(loss.item(),math.log(4),places=5)
        loss.backward()
        self.assertGreater(logits.grad[0,0].abs().sum().item(),0)
        self.assertEqual(logits.grad[0,1:].abs().sum().item(),0)
        with self.assertRaises(ValueError): T.semantic_nll(logits,labels,torch.tensor([[1,0,0,0]]))

if __name__=='__main__': unittest.main()
