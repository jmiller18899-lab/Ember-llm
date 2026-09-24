"""Finalized Writing Repair 1 export. Data generation only; never trains a model.

Keep the original authored catalog unchanged. This release removes accidental
entity/value correlations discovered during its audit. Use THIS entry point.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import ember_writing_repair1_curriculum as seed

VERSION = 'writing-repair1-curriculum-v1.1'
CONTEXTS, NAMES, OWNERS = seed.CONTEXTS, seed.NAMES, seed.OWNERS
OBJECTS, WHEN = seed.OBJECTS, seed.WHEN
normalized = seed.normalized
training_view = seed.training_view

GROUNDING_VALUES = {
    'delivery date': ('Thursday', 'Saturday', 'Monday'),
    'arrival time': ('3:42 PM', '9:18 AM', '6:54 PM'),
    'completion date': ('Monday', 'Wednesday', 'Friday'),
    'finish time': ('11:28 AM', '2:46 PM', '8:32 AM'),
    'decision': ('approved', 'rejected', 'deferred'),
    'resolution': ('replacement issued', 'refund issued', 'repair completed'),
    'confirmation status': ('confirmed', 'declined', 'pending'),
    'shipping date': ('Wednesday', 'Friday', 'Sunday'),
}


def build():
    parts = seed.build()
    for split, rows in parts.items():
        for row in rows:
            if row['task'] == 'perspective':
                t, j = map(int, row['id'].rsplit('-', 2)[1:])
                m = row['meaning']
                old_object, old_time = m['object'], m['time']
                new_object = OBJECTS[split][(j + 3*t) % len(OBJECTS[split])]
                new_time = WHEN[split][(j + 2*t) % len(WHEN[split])]
                for key in ('prompt', 'answer'):
                    row[key] = row[key].replace(old_object, new_object).replace(old_time, new_time)
                row['require'] = [x.replace(old_object, new_object).replace(old_time, new_time)
                                  for x in row['require']]
                row['object'] = m['object'] = new_object
                m['time'] = new_time
                continue
            if row['task'] != 'retention':
                continue
            index = int(row['id'].rsplit('-', 1)[1])
            oracle = row['oracle']
            if row['family'] == 'extraction':
                i = index - 64
                record = oracle['record']
                record['state'] = ('queued', 'open', 'closed')[(i//3) % 3]
                key = oracle['key']
                if i % 2:
                    row['prompt'] = 'Record: ' + '; '.join(k+'='+v for k,v in record.items()) + f'. Return only the {key} value.'
                else:
                    row['prompt'] = json.dumps(record, sort_keys=True) + f'\nExtract {key}. Output the value alone.'
                row['answer'] = record[key]
            elif row['family'] == 'grounding' and oracle['key'] in oracle['record']:
                i = index - 144
                old = row['answer']
                new = GROUNDING_VALUES[oracle['key']][(i//8)//2]
                old_fact = oracle['key'] + ': ' + old
                new_fact = oracle['key'] + ': ' + new
                if row['prompt'].count(old_fact) != 1:
                    raise ValueError('Grounding source identity mismatch: ' + row['id'])
                row['prompt'] = row['prompt'].replace(old_fact, new_fact)
                oracle['record'][oracle['key']] = row['answer'] = new
            elif oracle['op'] == 'threshold' and index == 232:
                old = oracle['value']
                oracle['value'] = oracle['limit']
                row['prompt'] = row['prompt'].replace(f'The reading is {old}.', f"The reading is {oracle['value']}.")
                row['answer'] = 'LOW'
    return parts


def audit(parts, external=()):
    report = seed.audit(parts, external)
    report['version'] = VERSION
    report['finalized_entrypoint'] = 'jobs/ember_writing_repair1_release.py'
    return report


def write_package(outdir: Path):
    parts = build()
    report = audit(parts)
    if not report['passed']:
        raise ValueError(json.dumps(report['errors']))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    files = {}
    exports = [('train', training_view(parts)), ('train_metadata', parts['train']),
               ('dev', parts['dev']), ('test', parts['test'])]
    for name, rows in exports:
        raw = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in rows).encode('utf-8')
        path = outdir / (name+'.jsonl')
        path.write_bytes(raw)
        files[path.name] = {'rows': len(rows), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    manifest = {
        'version': VERSION, 'seed': seed.SEED, 'files': files,
        'finalized_entrypoint': 'jobs/ember_writing_repair1_release.py',
        'catalog_entrypoint_is_not_final_export': 'jobs/ember_writing_repair1_curriculum.py',
        'base_model': 'Qwen/Qwen3.5-4B', 'base_revision': seed.BASE_REV,
        'adapter': 'Jmiller18899/ember-qwen3.5-4b-repair2', 'adapter_revision': seed.MODEL_REV,
        'baseline_commit': seed.BASELINE_COMMIT, 'grader_commit': seed.GRADER_COMMIT,
        'training_input': 'train.jsonl', 'auxiliary_metadata_not_additional_training_rows': 'train_metadata.jsonl',
        'evaluation_only': ['dev.jsonl', 'test.jsonl'],
        'final_test_policy': 'Exclude test.jsonl from training, prompt tuning and checkpoint selection. Evaluate once after candidate selection.',
        'retention_provenance': '256 new verified rehearsal rows, not literal replay from previous training',
        'data_author_review': 'Semantic transformation patterns reviewed by the author; per-row ledger/oracle checks; not independent adjudication',
        'synthetic_template_based': True, 'independent_human_review': False,
        'ember_model_inference_performed': False, 'pretraining_exposure_unknown': True,
        'frozen_grader_unchanged': True, 'automatic_new_writing_evaluation_ready': False,
        'new_test_scoring': 'Use a separately declared blind semantic-review protocol; the frozen conservative grader alone is not suitable.',
        'training_started': False, 'automatic_promotion': False,
    }
    (outdir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    (outdir/'local_audit.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return manifest, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/writing_repair1'))
    args = parser.parse_args()
    manifest, report = write_package(args.output)
    print(json.dumps({'manifest': manifest, 'audit': report}, indent=2))

if __name__ == '__main__':
    main()
