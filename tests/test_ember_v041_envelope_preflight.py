from __future__ import annotations
import copy, json
import pytest
from jobs import ember_envelope_preflight_v041 as p
CFG=json.loads(p.DEFAULT_CONFIG.read_text())

def test_config_is_cpu_only_and_keeps_floors():
    c=p.load_config(p.DEFAULT_CONFIG)
    assert c['version']=='0.0.41'
    assert c['historical_floor']=={'short_code':8,'long_code':7,'url':7,'path':9,'mixed':9}
    assert c['training_authorized'] is c['gpu_training_authorized'] is c['production_authorized'] is False

def test_subtype_resolver_matches_all_heldout_weak_values():
    for kind,value,_ in p.copy_data.DIAGNOSTICS:
        if kind in p.CALIBRATION_KINDS:
            v=p.subtype(kind,value)
            assert 0 <= v < p.copy_data.VARIANTS[kind]

def test_calibration_values_cover_every_structural_variant_and_are_disjoint():
    vals=p.calibration_values(CFG)
    assert set(vals)=={(k,v) for k in p.CALIBRATION_KINDS for v in range(p.copy_data.VARIANTS[k])}
    flat=[x for xs in vals.values() for x in xs]
    assert len(flat)==len(set(flat))
    assert not set(flat)&set(p.copy_data.HELD_OUT_VALUES)
    assert all(len(xs)==6 for xs in vals.values())

def _row(kind,variant,candidate,ok):
    return {'kind':kind,'subtype':variant,'candidate':candidate,'score':{'envelope_json_valid':ok,'tool_name_correct':ok}}

def test_selector_keeps_baseline_without_two_case_margin():
    rows=[]
    for kind in p.CALIBRATION_KINDS:
      for v in range(p.copy_data.VARIANTS[kind]):
       for i in range(6):
        rows += [_row(kind,v,'baseline',i<4),_row(kind,v,'typed_exact',i<5),_row(kind,v,'query_literal',i<3)]
    _,sel=p.select(rows,CFG)
    assert set(sel.values())=={'baseline'}

def test_selector_allows_clear_subtype_specific_win():
    rows=[]
    target=('short_code',0)
    for kind in p.CALIBRATION_KINDS:
      for v in range(p.copy_data.VARIANTS[kind]):
       for i in range(6):
        if (kind,v)==target:
            rows += [_row(kind,v,'baseline',i<2),_row(kind,v,'typed_exact',i<4),_row(kind,v,'query_literal',i<1)]
        else:
            rows += [_row(kind,v,'baseline',i<5),_row(kind,v,'typed_exact',i<5),_row(kind,v,'query_literal',i<4)]
    _,sel=p.select(rows,CFG)
    assert sel[target]=='typed_exact'
    assert all(c=='baseline' for k,c in sel.items() if k!=target)

def test_final_cases_choose_by_structural_subtype():
    selected={(k,v):'baseline' for k in p.CALIBRATION_KINDS for v in range(p.copy_data.VARIANTS[k])}
    cases=p.build_final_cases(CFG,selected)
    assert len(cases)==90
    for c in cases:
        if c['kind'] in p.CALIBRATION_KINDS:
            assert c['subtype']==p.subtype(c['kind'],c['target'])
            assert c['variant'].endswith(':baseline')

def test_regression_gate_rejects_any_floor_loss():
    pk={k:{'envelope_json_valid':10} for k in p.copy_data.KINDS}
    assert p.regression_gate(pk,CFG)['passed']
    pk['path']['envelope_json_valid']=8
    assert not p.regression_gate(pk,CFG)['passed']

def test_training_cannot_be_enabled(tmp_path,monkeypatch):
    c=copy.deepcopy(CFG); c['gpu_training_authorized']=True
    path=tmp_path/'c.json'; path.write_text(json.dumps(c)); monkeypatch.setattr(p,'DEFAULT_CONFIG',path)
    with pytest.raises(ValueError,match='CPU preflight-only'): p.load_config(path)
