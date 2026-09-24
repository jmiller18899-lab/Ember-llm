import copy
import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jobs'))
import ember_writing_repair1_curriculum as c

class CurriculumTests(unittest.TestCase):
    def setUp(self):
        self.parts = c.build()
    def test_exact_split_sizes(self):
        self.assertEqual({k:len(v) for k,v in self.parts.items()}, {'train':512,'dev':64,'test':96})
    def test_training_mix(self):
        self.assertEqual(Counter(r['task'] for r in self.parts['train']), {'shortening':160,'perspective':96,'retention':256})
    def test_evaluation_balance(self):
        for split, n in [('dev',32),('test',48)]:
            self.assertEqual(Counter(r['task'] for r in self.parts[split]), {'shortening':n,'perspective':n})
    def test_references_pass_audit(self):
        self.assertTrue(c.audit(self.parts)['passed'])
    def test_deterministic(self):
        self.assertEqual(self.parts, c.build())
    def test_no_exact_cross_split_prompt_overlap(self):
        seen=set()
        for rows in self.parts.values():
            for row in rows:
                p=c.normalized(row['prompt'])
                self.assertNotIn(p,seen); seen.add(p)
    def test_blueprints_are_split_disjoint(self):
        sets=[{r['blueprint'] for r in self.parts[s]} for s in ('train','dev','test')]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    def test_ownership_balance(self):
        for split,n in [('train',48),('dev',16),('test',24)]:
            rows=[r for r in self.parts[split] if r['task']=='perspective']
            self.assertEqual(Counter(r['meaning']['owner_role'] for r in rows), {'recipient':n,'third_party':n})
    def test_every_shortening_is_actually_shorter(self):
        for rows in self.parts.values():
            for r in rows:
                if r['task']=='shortening': self.assertLess(len(r['answer']),len(r['source']),r['id'])
    def test_meaning_change_rejected(self):
        r=next(r for r in self.parts['train'] if r['task']=='shortening' and r['meaning']['force']=='request')
        r['answer']=r['answer'].replace(', please ', ' must ')
        self.assertFalse(c.audit(self.parts)['passed'])
    def test_wrong_owner_rejected(self):
        r=next(r for r in self.parts['train'] if r['task']=='perspective' and r['meaning']['owner_role']=='third_party')
        r['answer']=r['answer'].replace(r['meaning']['owner']+"'s", 'your')
        self.assertFalse(c.audit(self.parts)['passed'])
    def test_arithmetic_error_rejected(self):
        r=next(r for r in self.parts['train'] if r.get('family')=='arithmetic')
        r['answer']=str(int(r['answer'])+1)
        self.assertFalse(c.audit(self.parts)['passed'])
    def test_duplicate_rejected(self):
        self.parts['train'][1]=copy.deepcopy(self.parts['train'][0])
        self.assertFalse(c.audit(self.parts)['passed'])
    def test_external_contamination_rejected(self):
        bad=[{'prompt':self.parts['train'][0]['prompt']}]
        self.assertFalse(c.audit(self.parts, bad)['passed'])
    def test_holdout_not_in_training_view(self):
        view=c.training_view(self.parts)
        self.assertEqual(len(view),512)
        self.assertTrue(all(set(r)=={'id','prompt','answer'} for r in view))
        self.assertFalse({r['id'] for r in view}&{r['id'] for r in self.parts['test']})

class SemanticMutationTests(unittest.TestCase):
    def setUp(self): self.parts=c.build()
    def test_invented_extra_instruction_rejected(self):
        r=next(r for r in self.parts['train'] if r['task']=='shortening')
        r['answer']+=' Also call the manager.'
        self.assertFalse(c.audit(self.parts)['passed'])
    def test_double_negation_on_prohibition_rejected(self):
        r=next(r for r in self.parts['train'] if r['task']=='shortening' and r['meaning']['force']=='prohibition')
        r['answer']=r['answer'].replace('must not ', 'must not not ')
        self.assertFalse(c.audit(self.parts)['passed'])
    def test_source_qualifier_tampering_rejected(self):
        r=next(r for r in self.parts['train'] if r['task']=='shortening' and 'exactly' in r['source'])
        r['source']=r['source'].replace('exactly', 'at least')
        self.assertFalse(c.audit(self.parts)['passed'])
    def test_optional_not_confused_with_prohibition(self):
        r=next(r for r in self.parts['test'] if r['task']=='shortening' and r['meaning']['force']=='no_obligation')
        r['answer']=r['answer'].replace('need not', 'must not')
        self.assertFalse(c.audit(self.parts)['passed'])
    def test_change_time_in_perspective_rejected(self):
        r=next(r for r in self.parts['train'] if r['task']=='perspective')
        r['answer']=r['answer'].replace(r['meaning']['time'], 'next year')
        self.assertFalse(c.audit(self.parts)['passed'])
    def test_uncertain_ability_is_not_certain_ability(self):
        r=next(r for r in self.parts['train'] if r['task']=='perspective' and r['meaning']['owner_role']=='recipient' and r['meaning']['uncertain'])
        self.assertIn('might be able to return',r['answer'])
        r['answer']=r['answer'].replace('might be able to', 'can')
        self.assertFalse(c.audit(self.parts)['passed'])
    def test_negation_cannot_be_added_to_message(self):
        r=next(r for r in self.parts['train'] if r['task']=='perspective' and 'I can return' in r['answer'])
        r['answer']=r['answer'].replace('I can return','I can not return')
        self.assertFalse(c.audit(self.parts)['passed'])
    def test_added_message_claim_rejected(self):
        r=next(r for r in self.parts['train'] if r['task']=='perspective')
        r['answer']+=' The item is damaged.'
        self.assertFalse(c.audit(self.parts)['passed'])

if __name__=='__main__': unittest.main()
