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


def test_norm_matched_control_keeps_the_basis_component_the_projection_removes():
    model = torch.nn.Linear(3,1,bias=False)
    before = closure.diag.flat_parameters(model,torch)
    raw = torch.tensor([.01,-.02,.03])
    basis = [torch.tensor([1.,0.,0.]),torch.tensor([0.,1.,0.])]
    stats = closure.apply_norm_matched_proposal(model,before,raw,basis,torch)
    actual = closure.diag.flat_parameters(model,torch)-before
    torch.testing.assert_close(actual/actual.norm(),raw/raw.norm(),rtol=1e-5,atol=1e-6)
    assert actual[0] != 0 and actual[1] != 0
    projected = closure.apply_projected_proposal(torch.nn.Linear(3,1,bias=False),before,raw,basis,torch)
    assert abs(stats['actual_l2']-projected['projected_l2']) <= 1e-5*float(raw.norm())
    assert abs(stats['retained_norm_fraction']-projected['retained_norm_fraction']) <= 1e-9
    # the discriminating quantity: the projection keeps none of the basis component,
    # the control keeps all of it, and cosine cannot tell the two apart at this scale
    assert projected['actual_residual_fraction'] <= closure.MAX_RESIDUAL
    assert stats['actual_residual_fraction'] >= .5*stats['expected_residual_fraction'] > 0
    assert stats['actual_residual_fraction'] > 100*projected['actual_residual_fraction']


@pytest.mark.parametrize('raw', [torch.zeros(3),torch.tensor([float('nan'),0.,0.])])
def test_norm_matched_control_fails_closed_on_invalid_proposals(raw):
    model = torch.nn.Linear(3,1,bias=False)
    with pytest.raises(ValueError,match='finite nonzero'):
        closure.apply_norm_matched_proposal(model,closure.diag.flat_parameters(model,torch),raw,[],torch)


def test_control_endpoint_check_rejects_a_run_that_altered_direction_or_magnitude():
    full = {'checks':{'placement_learning':True,'copy':True,'familiar_floors':True}}
    fresh = {'anchors':{'all_retained':True},'holdout':{'all_retained':True}}
    margins = [{'source_token_still_top1':True}]
    good = [{'projection':{'actual_l2':.01,'direction_cosine':1.,'realized_norm_fraction':.995,
                           'retained_norm_fraction':.995,'actual_residual_fraction':.0958,
                           'expected_residual_fraction':.0999}} for _ in range(40)]
    checks = closure.final_checks(full,fresh,margins,good,closure.NORM_MATCHED_MODE)
    assert checks['all_40_nonzero_norm_matched_updates']
    assert 'all_40_nonzero_projected_updates' not in checks
    turned = [dict(u,projection=dict(u['projection'],actual_residual_fraction=1e-6)) for u in good]
    assert not closure.final_checks(full,fresh,margins,turned,closure.NORM_MATCHED_MODE)['all_40_nonzero_norm_matched_updates']
    shrunk = [dict(u,projection=dict(u['projection'],realized_norm_fraction=.5)) for u in good]
    assert not closure.final_checks(full,fresh,margins,shrunk,closure.NORM_MATCHED_MODE)['all_40_nonzero_norm_matched_updates']
    assert len(closure.final_checks(full,fresh,margins,good[:39],closure.NORM_MATCHED_MODE)) == len(checks)


def test_control_survives_the_float32_rounding_this_learning_rate_produces():
    """A tiny update on float32 parameters loses about a percent to assignment
    rounding, in both arms. The endpoint gate must tolerate that and still reject
    an update whose basis component was removed."""
    torch.manual_seed(0)
    model = torch.nn.Linear(500,500,bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.empty(500,500).uniform_(-1,1))
    before = closure.diag.flat_parameters(model,torch)
    basis = []
    for _ in range(8):
        v = torch.randn(250_000)
        for q in basis:
            v -= torch.dot(q,v)*q
        basis.append(v/v.norm())
    raw = torch.randn(250_000)
    raw = raw/raw.norm() + .0958*basis[0]
    raw = raw/raw.norm()*4.69e-4
    control = closure.apply_norm_matched_proposal(model,before,raw,basis,torch)
    projected = closure.apply_projected_proposal(torch.nn.Linear(500,500,bias=False),before,raw,basis,torch)
    assert abs(control['realized_norm_fraction']-control['retained_norm_fraction']) <= closure.MAX_NORM_MATCH_ERROR*control['retained_norm_fraction']
    assert control['actual_residual_fraction'] >= closure.MIN_RETAINED_BASIS_FRACTION*control['expected_residual_fraction']
    assert control['actual_residual_fraction'] > 100*projected['actual_residual_fraction']
    assert abs(control['direction_cosine']-projected['direction_cosine']) < .02
