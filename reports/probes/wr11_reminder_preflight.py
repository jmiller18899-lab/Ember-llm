"""CPU-only preparation for WR11: paired prompts and saved-output comparison.

No model is loaded, no training is performed and no checkpoint is promoted.
All cases are development-only. A probe pass is not a promotion decision.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'jobs'))
import ember_meaning_preservation_v3 as V3

VERSION = 'wr11-reminder-preflight-20261012-v1'
WRAPPERS = {
    'neutral': 'Shorten this note: {source}',
    'explicit': 'Shorten this note. Preserve reminders, force and every fact: {source}',
}
# Newly authored diagnostic sources. These are never training examples.
SOURCES = [
    ('remember-01', 'remember', 'Please remember that Mira should inspect the brass compass before 4:20 PM.'),
    ('remember-02', 'remember', 'Please remember that all tailors must collect exactly 14 pins after sunrise.'),
    ('remember-03', 'remember', "Please remember that Dalia's green lantern might flicker until midnight."),
    ('remember-04', 'remember', 'Please remember that you may lift the cedar lid only after 9:15 AM.'),
    ('forget-01', 'forget', 'Do not forget that Felix should examine the silver hinge before 2:40 PM.'),
    ('forget-02', 'forget', 'Do not forget that every baker must label exactly 19 jars before Tuesday.'),
    ('forget-03', 'forget', "Do not forget that Nolan's blue parcel might reach the pier after dawn."),
    ('forget-04', 'forget', 'Do not forget that you may move the wicker crate only if the curator has given consent.'),
    ('control-01', 'control', 'Please note that the potters should polish the granite vase before 3:10 PM.'),
    ('control-02', 'control', 'Please note that all weavers must count exactly 17 bobbins after lunch.'),
    ('control-03', 'control', "Please note that Selma's red parcel might reach the studio on Sunday."),
    ('control-04', 'control', 'Please note that you may open the maple shutter only after 5:45 PM.'),
]


def cases():
    rows = []
    for source_id, family, source in SOURCES:
        if family == 'control':
            reference = source.removeprefix('Please note that ')
            reference = reference[0].upper() + reference[1:]
            dropped = None
        else:
            reference = source.replace(' that ', ' ', 1)
            dropped = re.sub(r'^(Please remember |Do not forget )', '', reference)
            dropped = dropped[0].upper() + dropped[1:]
        for wrapper, template in WRAPPERS.items():
            rows.append(dict(id=f'wr11-probe-{source_id}-{wrapper}', source_id=source_id,
                             family=family, source=source, reference=reference,
                             reminder_dropped=dropped, wrapper=wrapper,
                             prompt=template.format(source=source),
                             split='development', training_allowed=False))
    return rows


def prompts():
    return [{k: v for k, v in r.items() if k not in ('reference', 'reminder_dropped')}
            for r in cases()]


def probe_sha256():
    raw = json.dumps({'version': VERSION, 'prompts': prompts()}, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def grade(row, output):
    return V3.grade_case({'kind': 'shortening', 'source': row['source']}, output, lambda r, o: True)


def audit(root=ROOT):
    """Exact overlap only, using previously observed inputs, not final holdouts."""
    historical = []
    for version in range(3, 11):
        module = importlib.import_module(f'ember_writing_repair{version}_data')
        historical += module.writing('train') + module.writing('dev')
    path = root / 'reports/evidence/writing-repair5-20261003/evidence/baseline-744.json'
    benchmark = json.loads(path.read_text())['records']
    observed = historical + [r['row'] for r in benchmark]
    norm = lambda s: ' '.join(re.findall(r'\w+', s.casefold()))
    known = {norm(r[k]) for r in observed for k in ('source', 'prompt') if r.get(k)}
    overlaps = [r['id'] for r in cases() if any(norm(r[k]) in known for k in ('source', 'prompt'))]
    if overlaps:
        raise ValueError('Observed input overlap: ' + ', '.join(overlaps))
    return {'exact_prompt_source_overlaps': 0, 'benchmark_records_checked': len(benchmark),
            'historical_rows_checked': len(historical), 'final_holdouts_loaded': False,
            'scope': 'Exact source/prompt matches against WR3-WR10 and observed 744 benchmark only; not a semantic overlap audit.'}


def validated(payload):
    if payload.get('probe_sha256') != probe_sha256():
        raise ValueError('Outputs do not match this exact probe')
    checkpoint = payload.get('checkpoint', {})
    for key, length in (('revision', 40), ('adapter_sha256', 64)):
        if not re.fullmatch(r'[0-9a-f]{' + str(length) + '}', str(checkpoint.get(key, ''))):
            raise ValueError('Checkpoint requires an immutable ' + key)
    if not all(isinstance(checkpoint.get(k), str) and checkpoint[k].strip() for k in ('repo_id', 'subfolder')):
        raise ValueError('Checkpoint repository and subfolder are required')
    generation = payload.get('generation')
    if (not isinstance(generation, dict) or generation.get('do_sample') is not False
            or generation.get('enable_thinking') is not False
            or type(generation.get('max_new_tokens')) is not int or generation['max_new_tokens'] <= 0):
        raise ValueError('Explicit deterministic generation settings are required')
    outputs = {}
    for row in payload.get('records', []):
        if row.get('id') in outputs:
            raise ValueError('Duplicate prediction id')
        if not isinstance(row.get('output'), str) or not isinstance(row.get('system_prompt'), str) or not row['system_prompt'].strip():
            raise ValueError('Output text and exact system prompt are required')
        outputs[row.get('id')] = row
    if set(outputs) != {r['id'] for r in cases()}:
        raise ValueError('Predictions must cover every probe id exactly once')
    return outputs


def summarize(outputs):
    rows = []
    counts = Counter()
    for case in cases():
        output = outputs[case['id']]['output']
        g = grade(case, output)
        counts[g['status']] += 1
        rows.append(dict(id=case['id'], source_id=case['source_id'], wrapper=case['wrapper'],
                         family=case['family'], output=output, **g))
    sensitive = []
    for source_id, _, _ in SOURCES:
        pair = [r for r in rows if r['source_id'] == source_id]
        if pair[0]['status'] != pair[1]['status']:
            sensitive.append(source_id)
    return {**{key: counts[key] for key in ('pass', 'fail', 'review')},
            'reminder_drops': sum('reminder_dropped' in r['reasons'] for r in rows),
            'prompt_sensitive_sources': sensitive,
            'by_wrapper': {w: dict(Counter(r['status'] for r in rows if r['wrapper'] == w)) for w in WRAPPERS},
            'rows': rows}


def compare(baseline, candidate):
    b, c = validated(baseline), validated(candidate)
    if baseline['generation'] != candidate['generation']:
        raise ValueError('Generation settings differ')
    if any(b[k]['system_prompt'] != c[k]['system_prompt'] for k in b):
        raise ValueError('System prompts differ between checkpoints')
    before, after = summarize(b), summarize(c)
    old = {r['id']: r['status'] for r in before['rows']}
    regressions = [r['id'] for r in after['rows'] if old[r['id']] == 'pass' and r['status'] != 'pass']
    improvements = [r['id'] for r in after['rows'] if old[r['id']] != 'pass' and r['status'] == 'pass']
    return {'version': VERSION, 'probe_sha256': probe_sha256(), 'grader_version': V3.GRADER_VERSION,
            'baseline_checkpoint': baseline['checkpoint'], 'candidate_checkpoint': candidate['checkpoint'],
            'baseline': before, 'candidate': after, 'net_pass_change': after['pass'] - before['pass'],
            'regressions': regressions, 'improvements': improvements,
            'preserves_baseline_passes': not regressions, 'production_ready': False,
            'training_recommendation_established': False,
            'limitations': ['Development diagnostic only; not a benchmark score.',
                            'Paired prompts share 12 sources; the 24 outputs are not independent cases.',
                            'The v3 checker can abstain; review is never counted as pass.',
                            'Checkpoint hashes are supplied by the producer, not independently verified here.',
                            'No inference, optimization or promotion is performed by this comparison.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, help='Export diagnostic prompts with no answer keys')
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.export:
        if args.baseline or args.candidate or args.output:
            parser.error('--export cannot be combined with comparison options')
        result = {'version': VERSION, 'probe_sha256': probe_sha256(), 'records': prompts(), 'audit': audit()}
        args.export.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'prompts': len(result['records']), 'audit': result['audit']}))
    elif args.baseline and args.candidate and args.output:
        result = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: result[k] for k in ('regressions', 'improvements', 'net_pass_change')}))
    else:
        parser.error('Use --export FILE or --baseline FILE --candidate FILE --output FILE')


if __name__ == '__main__':
    main()
