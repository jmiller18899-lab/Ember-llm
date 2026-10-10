"""Offline first-divergence loss-mask experiment; no inference or training.

Uses the pinned Qwen tokenizer. Fresh examples are diagnostic only and never
fed to WR8/WR9. A successful mask test does not establish a model improvement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE_REV = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
# source, preferred, rejected; all newly authored for this diagnostic.
CASES = [
 ('reminder', 'Please remember that the basalt badge must stay in drawer 83 until Tuesday.',
  'Please remember the basalt badge must stay in drawer 83 until Tuesday.',
  'The basalt badge must stay in drawer 83 until Tuesday.'),
 ('copy', 'Please remember that the amber lens must stay in cabinet 62 until Thursday.',
  'Please remember the amber lens must stay in cabinet 62 until Thursday.',
  'Please remember that the amber lens must stay in cabinet 62 until Thursday.'),
 ('advice', 'Please note that the surveyors should examine the ochre beacon prior to 6:35 PM.',
  'The surveyors should examine the ochre beacon before 6:35 PM.',
  'The surveyors must examine the ochre beacon before 6:35 PM.'),
 ('count', 'Please note that the stewards must retain exactly 13 wedges in bay 46 until Sunday.',
  'The stewards must retain exactly 13 wedges in bay 46 until Sunday.',
  'The stewards must retain 13 wedges in bay 46 until Sunday.'),
 ('ownership', "Please return Yara's ceramic prism to bay 46 prior to Sunday.",
  "Please return Yara's ceramic prism to bay 46 before Sunday.",
  'Please return your ceramic prism to bay 46 before Sunday.'),
 ('condition', 'Please note that you may use the copper sextant only if the conservator has given consent.',
  'You may use the copper sextant only if the conservator has consented.',
  'You may use the copper sextant if the conservator has consented.'),
 ('uncertainty', 'Please note that the russet parcel might arrive on Saturday.',
  'The russet parcel might arrive Saturday.',
  'The russet parcel will arrive Saturday.'),
 ('timing', 'Please note that you may uncover the indigo mural only after 8:25 PM at the pavilion.',
  'You may uncover the indigo mural only after 8:25 PM at the pavilion.',
  'You may uncover the indigo mural after 8:25 PM at the pavilion.'),
]


def first_divergence_mask(preferred, rejected):
    """Penalize the first wrong continuation under an identical prefix only."""
    for i, (good, bad) in enumerate(zip(preferred, rejected)):
        if good != bad:
            return [j == i for j in range(len(rejected))]
    raise ValueError('Require distinct token sequences including an end token')


def run(tokenizer_path):
    from tokenizers import Tokenizer
    sys.path.insert(0, str(ROOT / 'jobs'))
    import ember_meaning_preservation_v3 as grader
    import ember_writing_repair8_data as data
    tok = Tokenizer.from_file(str(tokenizer_path))
    end = tok.token_to_id('<|im_end|>')
    if end is None:
        raise ValueError('Missing Qwen end token')
    # Audit only already-observed benchmark inputs, never reserved final holdouts.
    benchmark = json.loads((ROOT / 'reports/evidence/writing-repair5-20261003/evidence/baseline-744.json').read_text())
    observed = [r['row'] for r in benchmark['records']] + data.writing('train') + data.writing('dev')
    known = {data.norm(r.get(k, '')) for r in observed for k in ('source', 'prompt') if r.get(k)}
    results = []
    for name, source, good, bad in CASES:
        assert data.norm(source) not in known, name + ': observed source overlap'
        row = {'kind': 'shortening', 'source': source}
        good_grade = grader.grade_case(row, good, lambda r, o: True)
        bad_grade = grader.grade_case(row, bad, lambda r, o: True)
        assert good_grade['status'] == 'pass', (name, good_grade)
        assert bad_grade['status'] != 'pass', (name, bad_grade)
        preferred = tok.encode(good, add_special_tokens=False).ids + [end]
        rejected = tok.encode(bad, add_special_tokens=False).ids + [end]
        mask = first_divergence_mask(preferred, rejected)
        divergence = mask.index(True)
        # Positions before this are identical tokens under identical contexts;
        # the current full-sequence negative loss penalizes all of them.
        assert not any(mask[:divergence]) and sum(mask) == 1
        assert preferred[divergence] != rejected[divergence]
        results.append({'id': name, 'training_allowed': False, 'source': source,
            'preferred': good, 'rejected': bad,
            'preferred_characters': len(good), 'source_characters': len(source),
            'preferred_grade': good_grade, 'rejected_grade': bad_grade,
            'negative_tokens_current': len(rejected), 'negative_tokens_proposed': sum(mask),
            'shared_prefix_penalties_current': divergence, 'shared_prefix_penalties_proposed': 0,
            'first_wrong_token': tok.id_to_token(rejected[divergence]),
            'first_preferred_token': tok.id_to_token(preferred[divergence])})
    return {'scope': 'Offline tokenizer/mask/checker probe only; no model inference or optimizer updates',
        'tokenizer_repo': 'Qwen/Qwen3.5-4B', 'tokenizer_revision': BASE_REV,
        'tokenizer_sha256': hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest(),
        'benchmark_training_overlap': 0, 'audit_scope': 'Exact source/prompt match against observed 744 inputs and WR8 train/dev only',
        'final_holdouts_loaded': False, 'production_ready': False, 'automatic_promotion': False,
        'rows': results,
        'summary': {'pairs': len(results), 'preferred_v3_pass': len(results),
            'rejected_v3_nonpass': len(results),
            'current_shared_prefix_penalties': sum(r['shared_prefix_penalties_current'] for r in results),
            'proposed_shared_prefix_penalties': 0,
            'current_negative_tokens': sum(r['negative_tokens_current'] for r in results),
            'proposed_negative_tokens': sum(r['negative_tokens_proposed'] for r in results)},
        'limitation': 'Removing a loss conflict is not proof of improved outputs. First-divergence-only loss may be too sparse; validate on fresh model generations before any WR9 launch.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = run(args.tokenizer)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary']))
