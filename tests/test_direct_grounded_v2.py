import copy
import json
from pathlib import Path
import pytest
from direct_answers.grounded_v2 import score,validate_data
ROOT=Path(__file__).resolve().parents[1]


def data():return json.loads((ROOT/'direct_answers/grounded-v2-data.json').read_text())
def tools():return json.loads((ROOT/'tool_assistant/data/router-training.json').read_text())['cases']


def test_every_reference_passes_and_data_is_disjoint():
    d=data();assert validate_data(d,tools())
    for split in ('development','confirmation'):
        assert all(score(r['answer'],r) for r in d[split])


@pytest.mark.parametrize('answer',[
    'Thank you, Calculator, for reviewing my code.',
    'Thank you for deleting my code.',
    'Thank you for reviewing my code. I already deployed it.',
    '',
])
def test_invented_recipient_wrong_action_or_extra_claim_fail(answer):
    assert not score(answer,{'answer':'Thank you for reviewing my code.'})


@pytest.mark.parametrize('answer',[
    'Niko reviewed 82 playlist records in 17 minutes.',
    'Niko reviewed -83 playlist records in 17 minutes.',
    'Niko reviewed - 83 playlist records in 17 minutes.',
    'Zelda reviewed 83 playlist records in 17 minutes.',
    'Niko reviewed 83 invoice records in 17 minutes.',
    'Niko did not review 83 playlist records in 17 minutes.',
])
def test_names_numbers_subject_and_negation_must_match(answer):
    assert not score(answer,{'answer':'Niko reviewed 83 playlist records in 17 minutes.'})


def test_capitalization_and_final_punctuation_do_not_change_content():
    assert score('THANK YOU, NORA, FOR CHECKING THE INVOICE!',{'answer':'Thank you, Nora, for checking the invoice.'})
    assert not score('Latency fell from 30.5 to 12.5 ms.',{'answer':'Latency fell from 305 to 125 ms.'})


def test_valid_paraphrase_can_fail_conservative_reference_metric():
    assert not score('I appreciate your code review.',{'answer':'Thank you for reviewing my code.'})


def test_confirmation_cannot_leak_into_training():
    d=data();d['train'][-1]=copy.deepcopy(d['confirmation'][0])
    with pytest.raises(ValueError,match='overlap'):validate_data(d,tools())


def test_historical_replay_cannot_silently_change():
    d=data();d['train'][0]['answer']='changed'
    with pytest.raises(ValueError,match='replay'):validate_data(d,tools())
