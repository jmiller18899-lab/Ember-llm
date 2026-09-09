import copy
import json
import hashlib
import pytest
import torch
from jobs import ember_training_balance as b


def test_snapshot_is_pinned_training_only():
    raw=b.SNAPSHOT.read_bytes();d=json.loads(raw)
    assert hashlib.sha256(raw).hexdigest()==b.SNAPSHOT_SHA
    assert 'development' not in d and 'holdout' not in d and 'cohorts' not in d
    train_ids={x['id'] for x in d['anchors']}
    assert len(train_ids)==8
    assert all(r['id'] in train_ids for f in d['expected_refreshes'] for r in f['rows'])


def test_cosine_distinguishes_conflict_alignment_and_zero():
    a=torch.tensor([1.,0.])
    assert b.cosine(a,-a,torch)==-1
    assert b.cosine(a,a,torch)==1
    assert b.cosine(a,torch.zeros(2),torch) is None


def test_shadow_trials_preserve_live_history_and_match_real_adamw(monkeypatch):
    model=torch.nn.Linear(2,1,bias=False)
    with torch.no_grad():model.weight.copy_(torch.tensor([[.2,-.3]]))
    cfg={'control_learning_rate':.001,'weight_decay':.01,'gradient_clip':.25,
         'placement_loss_weight':.2,'tool_kl_loss_weight':.55,'copy_kl_loss_weight':.25}
    opt=b.new_optimizer(model,torch,cfg)
    model.weight.square().sum().backward();opt.step();opt.zero_grad(set_to_none=True)
    def components(m,*args):
        return {'placement':lambda:(m.weight-1).square().sum(),
                'tool_kl':lambda:.01*m.weight.square().sum(),
                'copy_kl':lambda:.02*m.weight.square().sum(),
                'closing':lambda:(m.weight+1).square().sum()}
    monkeypatch.setattr(b,'component_losses',components)
    state=copy.deepcopy(opt.state_dict());initial=copy.deepcopy(model.state_dict())
    row=b.inspect_state(model,opt,None,None,torch,cfg,[],{'tools':[],'copies':[]},{},[],13)
    assert row['live_model_unchanged'] and row['live_optimizer_unchanged']
    assert row['placement_closing_cosine']<0
    actual=torch.nn.Linear(2,1,bias=False);actual.load_state_dict(initial)
    ao=b.new_optimizer(actual,torch,cfg,state)
    f=components(actual)
    total=.2*f['placement']()+.55*f['tool_kl']()+.25*f['copy_kl']()+.2*f['closing']()
    total.backward();torch.nn.utils.clip_grad_norm_(actual.parameters(),.25);ao.step()
    with torch.no_grad():expected={k:float(fn()) for k,fn in f.items()}
    trial=row['trials'][-1]
    for k in expected:assert trial['after'][k]==pytest.approx(expected[k],abs=1e-6)
    assert b.optimizer_digest(opt)==b.optimizer_digest(b.new_optimizer(model,torch,cfg,state))


def observations(placement=.095,closing=.001):
    return [{'step':s,'trials':[{'weight':w,'loss_decrease':{'placement':.1 if w==0 else placement,'closing':closing},'after':{'tool_kl':0.,'copy_kl':0.}} for w in b.WEIGHTS]} for s in sorted(b.STEPS)]


def test_recommendation_requires_all_states_and_both_losses():
    assert b.recommend(observations())['suggested_weight_for_next_full_test']==.025
    assert b.recommend(observations()[:3])['suggested_weight_for_next_full_test'] is None
    assert b.recommend(observations(placement=.08))['suggested_weight_for_next_full_test'] is None
    assert b.recommend(observations(closing=-.001))['suggested_weight_for_next_full_test'] is None
    assert not b.recommend(observations())['validated_training_balance']


def test_direction_measurements_remain_accurate_for_millions_of_tiny_updates():
    a=torch.full((3000000,),1e-7,dtype=torch.float32)
    a[::7]=-1e-7
    assert b.cosine(a,a,torch)==pytest.approx(1.,abs=1e-12)
    assert b.stable_norm(a,torch)==pytest.approx(float(a.double().norm()),rel=1e-10)
