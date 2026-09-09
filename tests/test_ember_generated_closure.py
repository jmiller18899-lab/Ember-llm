import json
import pytest
import torch
from jobs import ember_generated_closure as g

class Pieces:
    def __init__(self, pieces):
        self.pieces = pieces
    def decode(self, ids):
        return ''.join(self.pieces[i] for i in ids)
    def encode(self, text):
        assert text == 'prompt'
        return [0]

PREFIX = '<|tool|>\n{"arguments":{"query":"'

@pytest.mark.parametrize('body', ['AB12','a\\"b','a\\\\b','a\\nb',''])
@pytest.mark.parametrize('ending,kind', [('"},"name":"web_search"}', 'closed'), ('\n<|endoftext|>','unclosed'),('<|endoftext|>','unclosed')])
def test_value_boundary_preserves_escaped_argument(body, ending, kind):
    text = PREFIX + body + ending
    assert g.generated_value_end(text) == (len(PREFIX+body),kind)

@pytest.mark.parametrize('text', [PREFIX+'cutoff', PREFIX+'a\\q\n', PREFIX+'a\\\n', '{"query":"AB"}', PREFIX+'a\t\n'])
def test_unsafe_or_unknown_boundary_rejected(text):
    with pytest.raises(ValueError):
        g.generated_value_end(text)

@pytest.mark.parametrize('end', ['\n<|endoftext|>', '"},"name":"web_search"}'])
def test_only_punctuation_labels_after_actual_wrong_argument(end):
    t=Pieces(['prompt',PREFIX,'original filler','"},"','name','":"web_search"}','WRONG',end])
    row={'id':'train','target':'RIGHT','prompt':'prompt','source_ids':[1,2,3,4,5],'closing_position':2}
    (x,y), audit = g.structural_example(t,row,[1,6,7])
    assert x == [0,1,6]
    assert y == [-100,-100,3]
    assert audit['suffix_text']=='"},"'
    assert audit['value_changed_from_source']
    assert audit['prefix_ids']==[1,6]
    assert 'WRONG' in audit['completion']
    # An explicit wrong value is conditioning context only, never a target label.
    assert 6 not in y


def test_value_closure_token_straddle_is_rejected():
    t=Pieces(['prompt',PREFIX,'original','"},"','name','":"web_search"}','WRONG\n'])
    row={'id':'train','target':'RIGHT','prompt':'prompt','source_ids':[1,2,3,4,5],'closing_position':2}
    with pytest.raises(ValueError,match='straddles'):
        g.structural_example(t,row,[1,6])


def test_fresh_values_exclude_previous_experiments_and_each_other():
    used=g.diag.historical_values() | set(json.loads(g.PRIOR_VALUES.read_text())['excluded_values'])
    before=set(used)
    values=[g.fresh_value(p,v,i,used) for p in ['anchors','holdout'] for v in [0,1] for i in range(48)]
    assert len(set(values))==len(values)
    assert not set(values)&before


def test_holdout_or_existing_guard_failure_blocks_endpoint():
    fresh={'anchors':{'all_retained':True},'holdout':{'all_retained':False}}
    updates=[{'actual_l2':.01} for _ in range(40)]
    assert not all(g.final_checks({'checks':{'copy':True}},fresh,updates).values())
    fresh['holdout']['all_retained']=True
    assert not all(g.final_checks({'checks':{'copy':False}},fresh,updates).values())
    assert all(g.final_checks({'checks':{'copy':True}},fresh,updates).values())


def test_structural_batch_loss_has_no_gradient_on_argument_prediction():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits=torch.nn.Parameter(torch.zeros(1,3,8))
        def forward(self,x):
            return self.logits,None
    model=Model()
    g.objectives.batch_loss(model,torch,[([0,1,6],[-100,-100,3])],[0]).backward()
    assert torch.count_nonzero(model.logits.grad[:,:2])==0
    assert torch.count_nonzero(model.logits.grad[:,2])>0
