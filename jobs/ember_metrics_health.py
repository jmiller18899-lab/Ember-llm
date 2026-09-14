"""Recount archived non-prediction evidence; never treat this as a live evaluation."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def counts(rows, flag):
    require(bool(rows), 'Empty evidence cohort')
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate evidence IDs')
    require(all(type(r[flag]) is bool for r in rows), 'Non-boolean outcome')
    return sum(r[flag] for r in rows)


def routing(result):
    rows = result['cases']
    measured = {
        'combined_correct': counts(rows, 'passed'), 'combined_total': len(rows),
        'routing_correct': counts(rows, 'route_ok'), 'routing_total': len(rows),
        'exact_arguments_correct': counts([r for r in rows if r['route'] != 'direct'], 'arguments_ok'),
        'arguments_total': sum(r['route'] != 'direct' for r in rows),
    }
    for row in rows:
        expected = row['route_ok'] and (row['route'] == 'direct' or
                    (row['arguments_ok'] and row['fixture_dispatch_ok']))
        require(row['passed'] == expected, 'Inconsistent combined case outcome')
    require(all(result[k] == v for k, v in measured.items()), 'Routing aggregate mismatch')
    measured['failures'] = [r['id'] for r in rows if not r['passed']]
    return measured


def build(root=ROOT):
    registry = json.loads((root / 'config/ember_metrics_sources.json').read_text())
    sources, evidence = {}, {}
    for key, spec in registry.items():
        raw = (root / spec['path']).read_bytes()
        require(sha256(raw).hexdigest() == spec['sha256'], f'{key}: evidence hash mismatch')
        evidence[key] = json.loads(raw)
        sources[key] = {**spec, 'measured_at': evidence[key]['created_at']}
    route, live, direct = (evidence[k] for k in ('routing', 'live', 'direct'))
    comparisons = {}
    for precision in ('full', 'int4'):
        pair = route['results'][precision]
        baseline = pair[registry['routing'].get('baseline_arm', 'frozen_v7')]
        candidate = pair[registry['routing'].get('candidate_arm', 'definition_v8')]
        before, after = routing(baseline), routing(candidate)
        old = {r['id']: r for r in baseline['cases']}
        new = {r['id']: r for r in candidate['cases']}
        require(old.keys() == new.keys(), 'Unpaired cohorts')
        require(all(all(old[k].get(field) == new[k].get(field) for field in ('user', 'route', 'arguments'))
                    for k in old), 'Changed request or expected answer')
        comparisons[precision] = {
            'baseline': before, 'candidate': after,
            'repaired': sum(not old[k]['passed'] and new[k]['passed'] for k in old),
            'regressed': sum(old[k]['passed'] and not new[k]['passed'] for k in old),
        }
    strict = all(v['candidate']['combined_correct'] == v['candidate']['combined_total'] for v in comparisons.values())
    require(route['strict_pass'] is strict, 'Routing gate mismatch')
    parser = route['parser_regressions']
    require(counts(parser['cases'], 'passed') == parser['passed'], 'Parser aggregate mismatch')
    require(len(parser['cases']) == parser['total'], 'Parser denominator mismatch')
    live_rows = [row for rows in live['results'].values() for row in rows]
    for rows in live['results'].values():
        require(len({r['id'] for r in rows}) == len(rows), 'Duplicate live IDs')
    for field, aggregate in [('outcome', 'counts'), ('first_attempt_outcome', 'first_attempt_counts')]:
        measured = Counter(r[field] for r in live_rows)
        require(bool(live_rows) and set(measured) <= {'PASS', 'FAIL', 'BLOCKED'}, 'Invalid live outcome')
        require({k: measured[k] for k in ('PASS', 'FAIL', 'BLOCKED')} == live[aggregate], 'Live aggregate mismatch')
    if 'expected_checks' in registry['live']:
        require(len(live_rows) == registry['live']['expected_checks'], 'Live cohort size mismatch')
    original_counts = None
    if 'original_v9_contract' in live:
        original = live['original_v9_contract']
        original_rows = [r for rows in original['results'].values() for r in rows]
        require(bool(original_rows), 'Empty original live cohort')
        measured = Counter(r['outcome'] for r in original_rows)
        require(set(measured) <= {'PASS', 'FAIL', 'BLOCKED'}, 'Invalid original live outcome')
        original_counts = {k: measured[k] for k in ('PASS', 'FAIL', 'BLOCKED')}
        require(original_counts == original['counts'], 'Original live aggregate mismatch')
    answer = {}
    for key in ('baseline_quality', 'candidate_quality'):
        data = direct[key]
        require(counts(data['rows'], 'passed') == data['passed'], 'Answer aggregate mismatch')
        require(counts(data['rows'], 'legacy_passed') == data['legacy_passed'], 'Legacy aggregate mismatch')
        require(len(data['rows']) == data['total'], 'Answer denominator mismatch')
        answer[key] = {k: data[k] for k in ('passed', 'legacy_passed', 'total')}
    require([r['id'] for r in direct['baseline_quality']['rows']] ==
            [r['id'] for r in direct['candidate_quality']['rows']], 'Unpaired answer cohorts')
    progress = (direct['selected_step'] > 0 and
                direct['selected_development_loss'] <= direct['baseline_development_loss'] * 0.9 and
                direct['candidate_quality']['passed'] >= direct['baseline_quality']['passed'] + 3)
    require(direct['development_progress_gate'] is progress, 'Answer learning gate mismatch')
    protected = (direct['frozen_parameter_sha256_before'] == direct['frozen_parameter_sha256_after']
                 and direct['block04_features_identical'] and not direct['router_heads_changed'])
    require(protected, 'Protected routing parameters changed during answer trial')
    return {
        'schema_version': 1, 'scope': 'archived_non_prediction_evidence_recount',
        'sources': sources, 'paired_routing': comparisons,
        'parser_supplied_routes': {'passed': parser['passed'], 'total': parser['total']},
        'live_service_smoke': {'final': live['counts'], 'first_attempt': live['first_attempt_counts'],
                               'current_availability': 'not_retested',
                               'scope': live.get('scope', 'single_turn_service_smoke'),
                               'scope_gate_passed': all(r['outcome'] == 'PASS' for r in live_rows),
                               'original_unqualified_contract_counts': original_counts},
        'direct_answers': {**answer,
            'development_loss_before': direct['baseline_development_loss'],
            'development_loss_after': direct['selected_development_loss'],
            'learning_gate_passed': direct['development_progress_gate'],
            'fresh_confirmation_consumed': direct['confirmation_consumed'],
            'protected_routing_parameters_unchanged': protected,
            'limitation': 'Lexical passes can contain invented entities; not broad answer quality.'},
        'routing_confirmation_passed': route['strict_pass'],
        'production_ready': False,
        'unmeasured': ['current live uptime', 'broad generated-answer quality',
                       'native INT4 inference speed', 'production latency and cost'],
        'next_actions': ['Review the latest failed confirmation cases before the next frozen candidate.',
                         'Improve grounded direct-answer content before consuming fresh confirmation.',
                         'Run a new live smoke before release; archived PASS is not current health.'],
        'interpretation': 'Helper progress is separate from LLM learning. Different request suites are not a trend. CI success verifies evidence consistency, not assistant readiness.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    result = build()
    if args.check:
        require(json.loads(args.check.read_text()) == result, 'Scorecard stale: regenerate from registered evidence')
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
