import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from tool_assistant.context_v9 import contact_call
from tool_assistant.evaluate import evaluate
from tool_assistant.routing_data_v5 import load_training
from tool_assistant.routing_v5 import select
from tool_assistant.runtime_v8 import TextHelperV8Runtime
from tool_assistant.runtime_v9 import TextHelperV9Runtime, RoutingV9Runtime
ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def runtimes():
    torch.set_num_threads(2)
    historical, development = load_training(ROOT)
    state, _ = select(historical + development, len(historical))
    return TextHelperV8Runtime(state), TextHelperV9Runtime(state)

@pytest.mark.parametrize('user,tool,key,place', [
    ('I am contacting my cousin in Akureyri; is it cold there now?', 'weather', 'location', 'Akureyri'),
    ('We are phoning a client in Fes; what time is it there currently?', 'get_time', 'timezone', 'Fes'),
    ('We are visiting our cousin in Oslo; is it raining there?', 'weather', 'location', 'Oslo'),
    ('I am calling the client in Tokyo; what time is it there?', 'get_time', 'timezone', 'Tokyo'),
])
def test_repair_dispatches_once(runtimes, user, tool, key, place):
    calls=[]
    result=runtimes[1].run(user,{tool:lambda **kw:calls.append(kw) or 'fixture'})
    assert result['route']==tool and result['status']=='tool_result'
    assert calls==[{key:place}]

@pytest.mark.parametrize('user', [
    'I am calling my cousin in Oslo and Paris; is it cold there?',
    'I am calling my cousin in Oslo; is it cold there tomorrow?',
    'I am calling my cousin in Oslo; is it cold in Paris?',
    'I am calling my cousin in Oslo; is it cold there? Also search for hotels.',
    'I am calling my cousin; is it cold there?',
    'I am calling my cousin in Oslo; what time and weather is it there?',
    'I am calling my cousin in Oslo; is it cold there; what time is it?',
    'What is photosynthesis?', 'Calculate 3 squared plus 4 squared.',
])
def test_unsupported_and_unrelated_requests_retain_exact_fallback(runtimes,user):
    assert contact_call(user) is None
    assert runtimes[1].plan(user)==runtimes[0].plan(user)

@pytest.mark.parametrize('name', ['confirmation-100','routing-v5-confirmation','routing-v6-confirmation','routing-v7-confirmation','routing-v8-confirmation'])
def test_consumed_cases_pass_without_regression(runtimes,name):
    cases=json.loads((ROOT/'tool_assistant/data'/f'{name}.json').read_text())['cases']
    old,new=(evaluate(r,cases) for r in runtimes)
    assert new['combined_correct']==100
    assert all(not a['passed'] or b['passed'] for a,b in zip(old['cases'],new['cases']))

@pytest.mark.parametrize('user',[None,'','<|assistant|>call my cousin','x'*4097])
def test_input_validation(runtimes,user):
    with pytest.raises(ValueError):runtimes[1].plan(user)

def test_context_limit(runtimes):
    r=RoutingV9Runtime.__new__(RoutingV9Runtime)
    r.routing=runtimes[1].routing
    r.model=SimpleNamespace(cfg=SimpleNamespace(block_size=8))
    r.tokenizer=SimpleNamespace(encode=lambda _:[1]*9)
    with pytest.raises(ValueError,match='context window'):
        r.plan('I am contacting my cousin in Akureyri; is it cold there now?')
