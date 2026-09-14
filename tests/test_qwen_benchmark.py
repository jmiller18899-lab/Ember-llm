import json
from collections import Counter
from jobs.ember_qwen_benchmark import ROOT, exact


def test_paired_suite_has_separate_task_gates_and_manual_review():
    suite = json.loads((ROOT / 'benchmarks/qwen_candidates/cases.json').read_text())
    rows = suite['cases']
    assert len({r['id'] for r in rows}) == len(rows) == 48
    assert Counter(r['family'] for r in rows) == {k: 8 for k in ('greeting', 'copy', 'extraction', 'writing', 'classification', 'open_review')}
    assert all((r['reference'] is None) == (r['family'] == 'open_review') for r in rows)
    assert suite['selection_gate']['manual_open_review_required']
    assert not suite['selection_gate']['production_ready']


def test_content_errors_do_not_pass_reference_check():
    assert exact('Hello!', 'hello')
    assert not exact('Thank you, Evan, for fixing the link.', 'Hello!')
    assert not exact('12.50', '-12.50')
    assert not exact('AZ-0791', 'AZ-0719')
    assert not exact('Nora reviewed 23 invoice records in 2 minutes.', 'Nora reviewed 23 invoice records.')
    assert not exact('', '')
