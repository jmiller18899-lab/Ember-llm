"""Behavioral regressions for grader v2; these are tests, NOT training data."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jobs'))
import ember_drafting_repair_candidates_eval as w


def shortening(source, facts=()):
    return {'id': 'test', 'family': 'drafting', 'kind': 'shortening',
            'scoring': 'rubric', 'prompt': 'Shorten this: ' + source,
            'source': source, 'require': list(facts), 'negative_any': []}


CASES = [
    ('students_must_false_pass',
     'All students are kindly asked to leave exactly three folders on the blue desk by 4 PM.',
     'All students must leave exactly three folders on the blue desk by 4 PM.', False),
    ('drivers_must_false_pass',
     'We would appreciate it if all drivers could wait at the west entrance after 7:45 PM on Thursday.',
     'All drivers must wait at the west entrance after 7:45 PM on Thursday.', False),
    ('volunteers_must_false_pass',
     'All volunteers are kindly asked to gather by the loading dock before noon on Saturday.',
     'All volunteers must gather at the loading dock before noon Saturday.', False),
    ('valid_polite_request',
     'All students are kindly asked to leave exactly three folders on the blue desk by 4 PM.',
     'All students, please leave exactly three folders on the blue desk by 4 PM.', True),
    ('valid_appreciate_to_please',
     'We would appreciate it if all drivers could wait at the west entrance after 7:45 PM on Thursday.',
     'All drivers, please wait at the west entrance after 7:45 PM Thursday.', True),
    ('required_to_optional', 'Please remember that staff must wear badges inside the lab.',
     'Staff may wear badges inside the lab.', False),
    ('must_is_valid_when_in_source', 'Please remember that staff must wear badges inside the lab.',
     'Staff must wear badges inside the lab.', True),
    ('permission_to_requirement', 'Please remember that visitors are allowed to use the lift.',
     'Visitors must use the lift.', False),
    ('optional_to_required', 'Please remember that attendance at the briefing is optional.',
     'Attendance at the briefing is mandatory.', False),
    ('recommendation_to_required', 'Please remember that guests should arrive before noon.',
     'Guests must arrive before noon.', False),
    ('recommendation_preserved', 'Please remember that guests should arrive before noon.',
     'Guests should arrive before noon.', True),
    ('uncertainty_to_certainty', 'Please note that the parcel might arrive on Friday.',
     'The parcel will arrive Friday.', False),
    ('uncertainty_preserved', 'Please note that the parcel might arrive on Friday.',
     'The parcel might arrive Friday.', True),
    ('negation_dropped', 'Please remember that visitors must not enter the north gallery before noon on Friday.',
     'Visitors must enter the north gallery before noon Friday.', False),
    ('negation_added', 'Please remember that visitors must enter the north gallery before noon on Friday.',
     'Visitors must not enter the north gallery before noon Friday.', False),
    ('negation_preserved', 'Please remember that visitors must not enter the north gallery before noon on Friday.',
     'Visitors must not enter the north gallery before noon Friday.', True),
    ('no_obligation_is_not_prohibition', 'Please note that staff are not required to attend the briefing.',
     'Staff must not attend the briefing.', False),
    ('no_obligation_preserved', 'Please note that staff are not required to attend the briefing.',
     'Staff are not required to attend the briefing.', True),
    ('negation_substring_notebook', 'Please remember that staff must not move the notebook.',
     'Staff must move the notebook.', False),
    ('condition_dropped', 'Please note that visitors may enter only if the guide is present.',
     'Visitors may enter.', False),
    ('condition_preserved', 'Please note that visitors may enter only if the guide is present.',
     'Visitors may enter only if the guide is present.', True),
    ('condition_reversed', 'Please note that visitors may enter unless the gate is closed.',
     'Visitors may enter if the gate is closed.', False),
    ('exact_quantity_dropped', 'All students are kindly asked to leave exactly three folders on the blue desk by 4 PM.',
     'All students, please leave three folders on the blue desk by 4 PM.', False),
    ('quantity_changed', 'Please remember that guests must bring three forms to the office.',
     'Guests must bring four forms to the office.', False),
    ('before_to_after', 'Please remember that visitors must enter the north gallery before noon on Friday.',
     'Visitors must enter the north gallery after noon Friday.', False),
    ('deadline_changed_with_original_in_parentheses', 'Please remember that visitors must arrive at the gate by 8:30 AM.',
     'Visitors must arrive by 9 at the gate (8:30 AM).', False),
    ('number_substring', 'Please remember that staff must gather in room 12 before noon.',
     'Staff must gather in room 120 before noon.', False),
    ('role_reversal', 'Please note that Lena lends Omar the blue folder on Friday.',
     'Omar lends Lena the blue folder Friday.', False),
    ('unrecognized_new_fact', 'Please remember that staff must gather at the gate before noon.',
     'Staff must gather at the locked gate before noon.', False),
    ('lost_group', 'Please make sure that all technicians meet at the east loading bay before 10:20 AM on Wednesday.',
     'Please meet at the east loading bay before 10:20 AM Wednesday.', False),
    ('lost_am_pm', 'Please make sure that all technicians meet at the east loading bay before 10:20 AM on Wednesday.',
     'Please ensure all technicians meet at the east loading bay before 10:20 Wednesday.', False),
    ('valid_negated_subject_paraphrase', 'It is important that nobody move the red crates from storage room 12 before Saturday.',
     'No one move the red crates from storage room 12 before Saturday.', True),
    ('valid_plain_fact', 'Please note that the library closes at noon on Friday.',
     'The library closes at noon Friday.', True),
    ('advice_not_disguised_as_request', 'Please remember that guests should arrive before noon.',
     'Please require guests to arrive before noon.', False),
]


class MeaningGrading(unittest.TestCase):
    def test_identical_text_still_fails_shortening(self):
        row=shortening('All staff must arrive before noon.')
        self.assertFalse(w.fresh_score(row,row['source']))

    def test_recorded_fresh_rows_are_rejected(self):
        rows=w.fresh_cases()
        self.assertFalse(w.fresh_score(rows[20],CASES[0][2]))
        self.assertFalse(w.fresh_score(rows[22],CASES[1][2]))

    def test_main_suite_and_original_drafting_path_are_covered(self):
        # Without integration this reproduces the old shared scorer's false pass.
        source=CASES[2][1]; output=CASES[2][2]
        row=shortening(source,['loading dock','noon','Saturday'])
        row['kind']='shorten'; row['facts']=row.pop('require')
        legacy=lambda r,o: all(f.lower() in o.lower() for f in r['facts']) and len(o)<len(r['source'])
        self.assertTrue(legacy(row,output))
        grade=getattr(w,'grade_case',None)
        self.assertTrue(callable(grade),'one grading hook must protect every shortening suite')
        self.assertFalse(grade(row,output,legacy)['passed'])

    def test_non_shortening_grades_do_not_change(self):
        grade=getattr(w,'grade_case',None)
        self.assertTrue(callable(grade),'missing shared grading hook')
        for kind in ('recipient','thirdparty','message','arithmetic'):
            row={'kind':kind,'scoring':'exact','answer':'17'}
            self.assertTrue(grade(row,'17',lambda r,o: o==r['answer'])['passed'])
            self.assertFalse(grade(row,'18',lambda r,o: o==r['answer'])['passed'])

    def test_unrecognized_paraphrase_is_review_not_proven_pass(self):
        grade=getattr(w,'grade_case',None)
        self.assertTrue(callable(grade),'missing review status')
        row=shortening('Please remember that staff must assemble in the foyer before noon.')
        result=grade(row,'Staff must meet in the foyer before noon.',lambda r,o:True)
        self.assertFalse(result['passed'])
        self.assertEqual(result['status'],'review')
        self.assertTrue(result['reasons'])

    def test_legacy_failure_cannot_be_overruled(self):
        grade=getattr(w,'grade_case',None)
        self.assertTrue(callable(grade),'missing shared grading hook')
        row=shortening('Please note that the office closes at noon.')
        self.assertFalse(grade(row,'The office closes at noon.',lambda r,o:False)['passed'])

    def test_inputs_are_not_mutated(self):
        import copy
        row=shortening(CASES[0][1]); before=copy.deepcopy(row)
        w.fresh_score(row,CASES[0][2])
        self.assertEqual(row,before)


def make_case(source,output,expected):
    def test(self): self.assertEqual(w.fresh_score(shortening(source),output),expected)
    return test
for name,source,output,expected in CASES:
    setattr(MeaningGrading,'test_'+name,make_case(source,output,expected))


class GraderIntegrity(unittest.TestCase):
    def test_likely_cannot_become_unlikely(self):
        self.assertFalse(w.fresh_score(shortening('Please note that the delivery is likely to arrive Friday.'),
                                       'The delivery is unlikely to arrive Friday.'))

    def test_probability_strength_is_not_erased(self):
        self.assertFalse(w.fresh_score(shortening('Please note that the delivery might arrive Friday.'),
                                       'The delivery probably arrives Friday.'))

    def test_negative_amount_cannot_become_positive(self):
        self.assertFalse(w.fresh_score(shortening('Please note that the recorded balance is -12 units.'),
                                       'The recorded balance is +12 units.'))

    def test_currency_cannot_change(self):
        self.assertFalse(w.fresh_score(shortening('Please note that the price is $5 per item.'),
                                       'The price is €5 per item.'))

    def test_percent_cannot_disappear(self):
        self.assertFalse(w.fresh_score(shortening('Please note that the value is 5% of the total.'),
                                       'The value is 5 of the total.'))

    def test_result_preserves_legacy_grade(self):
        result=w.grade_case(shortening(CASES[0][1]),CASES[0][2],lambda r,o:True)
        self.assertTrue(result['legacy_passed'])
        self.assertFalse(result['passed'])
        self.assertTrue(any('force_changed' in reason for reason in result['reasons']))

    def test_empty_output_is_not_a_pass(self):
        for output in ('', '  ', None, 17):
            self.assertFalse(w.grade_case(shortening('Please note that the office is open.'),output,lambda r,o:True)['passed'])

    def test_missing_source_is_not_a_pass(self):
        row=shortening('source'); row.pop('source')
        self.assertFalse(w.grade_case(row,'Anything',lambda r,o:True)['passed'])

    def test_cpu_replay_rejects_recorded_false_passes(self):
        records=[{'config':'shortening_only','suite':'fresh_writing',
                  'row':shortening(CASES[0][1]),'output':CASES[0][2],'legacy_ok':True}]
        result=w.rescore_records(records)
        self.assertFalse(result['records'][0]['ok'])
        self.assertFalse(result['model_inference_performed'])
        self.assertIn('supplied records only', result['coverage'])

    def test_cpu_replay_rejects_missing_legacy_grade(self):
        with self.assertRaises(ValueError):
            w.rescore_records([{'output':'Something'}])

    def test_cpu_replay_rejects_duplicate_rows(self):
        row={'config':'x','suite':'test','row':shortening(CASES[0][1]),
             'output':CASES[0][2],'legacy_ok':True}
        with self.assertRaises(ValueError): w.rescore_records([row,row])

    def test_cpu_cli_does_not_import_model_or_fetch_frozen_code(self):
        import contextlib, io, json, tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'records.json'
            path.write_text(json.dumps([{'config':'x','suite':'test','row':shortening(CASES[0][1]),
                'output':CASES[0][2],'legacy_ok':True}]))
            with patch.object(sys,'argv',['evaluator','--rescore',str(path)]), \
                 patch.object(w,'load_frozen',side_effect=AssertionError('must remain offline')), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                w.main()
            self.assertFalse(json.loads(output.getvalue())['records'][0]['ok'])

    def test_comparison_uses_corrected_not_historical_thresholds(self):
        import copy
        legacy={'v3_baseline':{s:{'score':[n,n]} for s,n in w.EXPECTED.items()}}
        baseline={'suite_v3':{'score':[180,192],'families':{'drafting':[8,20]}}}
        candidate={'suite_v3':{'score':[181,192],'families':{'drafting':[9,20]}}}
        outputs={'v3_baseline':{('suite_v3',0):{'id':'case','ok':False}},
                 'candidate':{('suite_v3',0):{'id':'case','ok':True}}}
        reproduced,verdict=w.score_comparison({'v3_baseline':baseline,'candidate':candidate},outputs,legacy,{},[])
        self.assertTrue(all(reproduced.values()))
        self.assertTrue(verdict['candidate']['eligible_for_manual_review'])
        # A legacy +1 that disappears after regrading is NOT an improvement.
        candidate=copy.deepcopy(baseline)
        _,verdict=w.score_comparison({'v3_baseline':baseline,'candidate':candidate},outputs,legacy,{},[])
        self.assertFalse(verdict['candidate']['eligible_for_manual_review'])
        self.assertIn('suite_v3_above_corrected_baseline',verdict['candidate']['failed_checks'])

    def test_corrected_comparison_still_blocks_regression(self):
        legacy={'v3_baseline':{s:{'score':[n,n]} for s,n in w.EXPECTED.items()}}
        baseline={'suite_v3':{'score':[180,192],'families':{'drafting':[8,20]}}}
        candidate={'suite_v3':{'score':[182,192],'families':{'drafting':[10,20]}}}
        outputs={'v3_baseline':{('suite_v3',0):{'id':'regression','ok':True}},
                 'candidate':{('suite_v3',0):{'id':'regression','ok':False}}}
        _,verdict=w.score_comparison({'v3_baseline':baseline,'candidate':candidate},outputs,legacy,{},[])
        self.assertFalse(verdict['candidate']['eligible_for_manual_review'])
        self.assertIn('no_individual_regressions',verdict['candidate']['failed_checks'])

if __name__=='__main__': unittest.main()
