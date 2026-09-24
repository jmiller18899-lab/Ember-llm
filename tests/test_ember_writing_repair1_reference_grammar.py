"""Reference agreement checks, outside every model dataset."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jobs'))
import ember_writing_repair1_release as D

class ReferenceAgreement(unittest.TestCase):
    def test_plural_object_is_not_given_singular_agreement(self):
        for rows in D.build_splits().values():
            for row in rows:
                with self.subTest(id=row['id']):
                    self.assertNotRegex(row['prompt']+' '+row['answer'],
                        r'\bbinoculars\b[^.]{0,100}\b(?:is|it)\b')

if __name__=='__main__': unittest.main()
