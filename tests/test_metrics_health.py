import hashlib
import json
from pathlib import Path
import shutil

import pytest
from jobs.ember_metrics_health import ROOT, build


@pytest.fixture
def evidence(tmp_path):
    (tmp_path / 'config').mkdir()
    (tmp_path / 'reports').mkdir()
    shutil.copy(ROOT / 'config/ember_metrics_sources.json', tmp_path / 'config')
    registry = json.loads((tmp_path / 'config/ember_metrics_sources.json').read_text())
    for spec in registry.values():
        shutil.copy(ROOT / spec['path'], tmp_path / spec['path'])
    return tmp_path


def mutate(root, key, change, rehash=True):
    path = root / 'config/ember_metrics_sources.json'
    registry = json.loads(path.read_text())
    target = root / registry[key]['path']
    data = json.loads(target.read_text())
    change(data)
    target.write_text(json.dumps(data))
    if rehash:
        registry[key]['sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
        path.write_text(json.dumps(registry))


def test_archived_scores_keep_failed_gates_and_unmeasured_health():
    score = build()
    assert score == json.loads((ROOT / 'reports/ember-non-prediction-scorecard.json').read_text())
    assert score['paired_routing']['full']['repaired'] == 20
    assert score['paired_routing']['full']['regressed'] == 0
    assert score['routing_confirmation_passed']
    assert not score['direct_answers']['learning_gate_passed']
    assert not score['production_ready']
    assert score['live_service_smoke']['current_availability'] == 'not_retested'


@pytest.mark.parametrize('key,change,match', [
    ('routing', lambda d: d['results']['full']['contact_v9'].__setitem__('combined_correct', 99), 'aggregate'),
    ('routing', lambda d: d.__setitem__('strict_pass', False), 'gate'),
    ('routing', lambda d: d['results']['full']['frozen_v8']['cases'][0].__setitem__('user', 'different cohort'), 'Changed request'),
    ('routing', lambda d: d['results']['full']['contact_v9']['cases'].append(d['results']['full']['contact_v9']['cases'][0]), 'Duplicate'),
    ('direct', lambda d: d.__setitem__('development_progress_gate', True), 'gate'),
    ('direct', lambda d: d.__setitem__('frozen_parameter_sha256_after', 'changed'), 'Protected'),
    ('live', lambda d: d['first_attempt_counts'].__setitem__('PASS', 52), 'aggregate'),
])
def test_reject_misleading_evidence_even_after_registry_update(evidence, key, change, match):
    mutate(evidence, key, change)
    with pytest.raises(ValueError, match=match):
        build(evidence)


def test_reject_unregistered_evidence_change(evidence):
    mutate(evidence, 'direct', lambda d: d.__setitem__('selected_step', 40), rehash=False)
    with pytest.raises(ValueError, match='hash mismatch'):
        build(evidence)


def test_assisted_flow_does_not_relabel_original_failures():
    live = build()['live_service_smoke']
    assert live['scope'] == 'service_v10_with_explicit_country_reply'
    assert live['scope_gate_passed'] is True
    assert live['final']['PASS'] == 52 and live['first_attempt']['PASS'] == 49
    assert live['original_unqualified_contract_counts']['FAIL'] == 2


def test_original_live_counts_cannot_be_rewritten(evidence):
    mutate(evidence, 'live', lambda d: d['original_v9_contract']['counts'].__setitem__('FAIL', 0))
    with pytest.raises(ValueError, match='Original live aggregate mismatch'):
        build(evidence)


def test_grounded_result_keeps_classification_separate_from_writing():
    answer = build()['direct_grounded_v2']
    assert answer['learning_gate_passed'] and answer['confirmation_consumed']
    assert not answer['confirmation_gate_passed']
    assert answer['confirmation']['passed'] == 6
    assert answer['confirmation']['by_family']['label'] == {'passed': 6, 'total': 6}
    for family in ('thanks', 'rewrite', 'facts'):
        assert answer['confirmation']['by_family'][family] == {'passed': 0, 'total': 6}


@pytest.mark.parametrize('change,match', [
    (lambda d: d['confirmation'].__setitem__('passed', 24), 'aggregate'),
    (lambda d: d.__setitem__('confirmation_gate_passed', True), 'confirmation gate'),
    (lambda d: d.__setitem__('confirmation_consumed', False), 'consumption'),
    (lambda d: d['confirmation']['rows'][-1].__setitem__('text', 'invented answer'), 'raw answer'),
    (lambda d: d.__setitem__('frozen_parameter_sha256_after', 'changed'), 'Protected'),
])
def test_grounded_report_rejects_false_success(evidence, change, match):
    mutate(evidence, 'direct_grounded_v2', change)
    with pytest.raises(ValueError, match=match):
        build(evidence)
