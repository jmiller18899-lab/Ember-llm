import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jobs'))
import ember_drafting_repair_candidates_eval as w

class WritingPolicyScope(unittest.TestCase):
    def test_message_requests(self):
        for p in ['Text Imani that I have her coat.', 'Tell Mateo that his tablet is here.', 'Message Seren: I found their watch.', 'Draft a reply to Zora saying the parcel is ready.', 'Write a note to Ravi: Juno has his ticket.', 'Let Lior know I can return their book tomorrow.']:
            with self.subTest(prompt=p): self.assertTrue(w.perspective_gate(p))
    def test_nonmessage_requests(self):
        for p in ['Calculate 17 + 6 - 9.', 'Return only the word Text.', 'Explain how to draft a message.', 'owner=Zora; item=coat. Whose item is it?', 'Translate this: "Text Zora that I have her coat."', 'Shorten this: Text Zora that I have her coat.']:
            with self.subTest(prompt=p): self.assertFalse(w.perspective_gate(p))
    def test_supplied_shortening(self):
        for p in ['Shorten this: Please bring the blue folder on Tuesday.', 'Make this shorter without losing facts: All guests must gather at gate 6 by noon.', 'Condense this but keep the deadline: "No one may enter before 8."', 'Please shorten this: The desk closes at five every evening.']:
            with self.subTest(prompt=p): self.assertTrue(w.shortening_gate(p))
    def test_missing_or_embedded_shortening(self):
        for p in ['Shorten this.', 'Can you make it shorter?', 'Summarize my notes.', 'Explain how shortening works.', 'Return this exactly: Shorten this: hello world.', 'owner=Kai; note=shorten this; date=Tuesday. What day?', 'Make this shorter:']:
            with self.subTest(prompt=p): self.assertFalse(w.shortening_gate(p))
    def test_augmentation_is_scoped(self):
        system='frozen-system'
        self.assertEqual(w.augment('Compute 7+4.',system,True,True),system)
        self.assertEqual(w.augment('Shorten this.',system,True,True,missing=True),system)
        self.assertEqual(w.augment('Text Zora that her bag is ready.',system,False,False),system)
        self.assertIn('his',w.augment('Text Zora that her bag is ready.',system,True,False))
        self.assertIn('shorter',w.augment('Shorten this: Please bring your bag tomorrow.',system,False,True))
    def test_fresh_case_count_and_uniqueness(self):
        rows=w.fresh_cases()
        self.assertEqual(len(rows),24)
        self.assertEqual(len({r['prompt'] for r in rows}),24)
    def test_third_party_ownership_not_reassigned(self):
        row=w.fresh_cases()[12]
        self.assertTrue(w.fresh_score(row,'Hi Tamsin, Orin left his camera in the studio.'))
        self.assertFalse(w.fresh_score(row,'Hi Tamsin, Orin left your camera in the studio.'))
    def test_shortening_rejects_copy_or_lost_negation(self):
        row=w.fresh_cases()[19]
        self.assertFalse(w.fresh_score(row,row['source']))
        self.assertFalse(w.fresh_score(row,'Visitors, enter the north gallery before noon Friday.'))
        self.assertTrue(w.fresh_score(row,'Visitors must not enter the north gallery before noon Friday.'))
    def test_recipient_and_action_honesty(self):
        row=w.fresh_cases()[0]
        self.assertTrue(w.fresh_score(row,'Hi Imani, I have your coat and can return it Sunday.'))
        self.assertFalse(w.fresh_score(row,'Hi Imani, I have her coat and can return it Sunday.'))
        self.assertFalse(w.fresh_score(row,'I sent Imani a message about your coat on Sunday.'))

if __name__=='__main__': unittest.main()
