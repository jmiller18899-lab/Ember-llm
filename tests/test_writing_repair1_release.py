import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'jobs'))
import ember_writing_repair1_release as r

class ReleaseTests(unittest.TestCase):
    def setUp(self): self.p=r.build()
    def test_extraction_state_is_not_always_closed(self):
        states={x['answer'] for x in self.p['train'] if x['family']=='extraction' and x['oracle']['key']=='state'}
        self.assertEqual(states,{'queued','open','closed'})
    def test_grounding_values_vary_for_each_topic(self):
        keys={x['oracle']['key'] for x in self.p['train'] if x['family']=='grounding'}
        for key in keys:
            known={x['answer'] for x in self.p['train'] if x['family']=='grounding' and x['oracle']['key']==key and key in x['oracle']['record']}
            self.assertEqual(len(known),3,key)
    def test_same_possession_has_different_times(self):
        objects={x['meaning']['object'] for x in self.p['train'] if x['task']=='perspective'}
        for obj in objects:
            times={x['meaning']['time'] for x in self.p['train'] if x['task']=='perspective' and x['meaning']['object']==obj}
            self.assertGreater(len(times),1,obj)
    def test_threshold_includes_equality(self):
        equality=[x for x in self.p['train'] if x.get('oracle',{}).get('op')=='threshold' and x['oracle']['value']==x['oracle']['limit']]
        self.assertEqual(len(equality),1)
        self.assertEqual(equality[0]['answer'],'LOW')
    def test_all_final_records_pass_reference_audit(self):
        self.assertTrue(r.audit(self.p)['passed'])
    def test_no_shortening_targets_changed(self):
        before={x['id']:x for rs in r.seed.build().values() for x in rs if x['task']=='shortening'}
        after={x['id']:x for rs in self.p.values() for x in rs if x['task']=='shortening'}
        self.assertEqual(before,after)
    def test_deterministic(self): self.assertEqual(self.p,r.build())
    def test_training_view_is_training_only(self):
        view=r.training_view(self.p)
        self.assertEqual(len(view),512)
        self.assertTrue(all(set(x)=={'id','prompt','answer'} for x in view))

if __name__=='__main__': unittest.main()
