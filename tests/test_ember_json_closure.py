import json

import pytest
import torch

from jobs import ember_json_closure as closure


class Pieces:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids):
        return ''.join(self.pieces[i] for i in ids)


@pytest.mark.parametrize('value', ['K8J3', 'a"b', 'a\\b', 'a\nb', ''])
def test_closing_quote_alignment_handles_escaped_content(value):
    text = '<|tool|>\n' + json.dumps({'arguments': {'query': value}, 'name': 'web_search'})
    tokenizer = Pieces(list(text))
    ids = list(range(len(text)))
    pos = closure.closure_position(tokenizer, ids)
    before = text[:pos]
    assert before.endswith(json.dumps(value)[:-1])
    assert text[pos:].startswith('"}')


def test_closing_punctuation_can_share_a_token():
    tokenizer = Pieces(['<|tool|>\n{"arguments":{"query":"', 'AB12', '"},"', 'name":"web_search"}'])
    assert closure.closure_position(tokenizer, [0,1,2,3]) == 2


def test_token_spanning_value_and_closing_quote_is_rejected():
    tokenizer = Pieces(['<|tool|>\n{"arguments":{"query":"', 'AB', '12"},"', 'name":"web_search"}'])
    with pytest.raises(ValueError, match='value characters'):
        closure.closure_position(tokenizer, [0,1,2,3])


def test_actual_optimizer_delta_is_projected_without_erasing_allowed_direction():
    model = torch.nn.Linear(3,1,bias=False)
    before = closure.diag.flat_parameters(model,torch)
    raw = torch.tensor([.01,-.02,.03])
    basis = [torch.tensor([1.,0.,0.]),torch.tensor([0.,1.,0.])]
    stats = closure.apply_projected_proposal(model,before,raw,basis,torch)
    actual = closure.diag.flat_parameters(model,torch)-before
    torch.testing.assert_close(actual[:2],torch.zeros(2),rtol=0,atol=0)
    assert actual[2] != 0
    assert stats['retained_norm_fraction'] < 1
    assert stats['actual_residual_fraction'] <= closure.MAX_RESIDUAL


@pytest.mark.parametrize('raw', [torch.zeros(3),torch.tensor([float('nan'),0.,0.])])
def test_invalid_proposals_fail_closed(raw):
    model = torch.nn.Linear(3,1,bias=False)
    with pytest.raises(ValueError,match='finite nonzero'):
        closure.apply_projected_proposal(model,closure.diag.flat_parameters(model,torch),raw,[],torch)


def test_fresh_cohorts_never_reuse_prior_values_or_each_other():
    used = closure.diag.historical_values()
    used.update(json.loads(closure.PRIOR_VALUES.read_text())['excluded_values'])
    original = set(used)
    values = [closure.fresh_value(phase,variant,index,used)
              for phase in ['anchors','holdout'] for variant in (0,1) for index in range(48)]
    assert len(values)==len(set(values))
    assert not set(values) & original
    assert not set(values) & set(closure.data.copy_data.HELD_OUT_VALUES)


def test_fresh_holdout_failure_cannot_pass_the_endpoint():
    updates = [{'projection':{'actual_l2':.01,'actual_residual_fraction':0.}} for _ in range(40)]
    full = {'checks':{'placement_learning':True,'copy':True,'familiar_floors':True}}
    fresh = {'anchors':{'all_retained':True},'holdout':{'all_retained':False}}
    checks = closure.final_checks(full,fresh,[{'source_token_still_top1':True}],updates)
    assert not checks['fresh_holdout_retained']
    assert not all(checks.values())
    fresh['holdout']['all_retained']=True
    checks = closure.final_checks(full,fresh,[{'source_token_still_top1':False}],updates)
    assert not checks['protected_closing_tokens_still_top1']
