"""Read-only paired diagnosis; these probes are development evidence, not confirmation."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from direct_answers.grounded_v2 import normalized, score
from direct_answers.learn import generate
from tool_assistant.runtime import load_model, sha256


def cases(data):
    result = []
    def add(group, task, user, answer):
        result.append(dict(id=f'{group}-{task}', group=group, task=task, user=user, answer=answer))
    training = set(normalized(' '.join(r['user'] + ' ' + r['answer'] for r in data['train'])))
    for group, name, subject, number in [('training_entities', 'Nora', 'invoice', '23'),
                                        ('consumed_entities', 'Nadia', 'reservation', '137'),
                                        ('novel_entities', 'Qevora', 'ledgerpane', '5831')]:
        if group == 'training_entities':
            assert all(v.casefold() in training for v in (name, subject, number))
        if group == 'novel_entities':
            assert all(v.casefold() not in training for v in (name, subject, number))
        text = f'{name} reviewed {number} {subject} records.'
        for task, user, answer in [
            ('copy_name', f'Repeat exactly: {name}', name),
            ('copy_subject', f'Repeat exactly: {subject}', subject),
            ('copy_number', f'Repeat exactly: {number}', number),
            ('copy_sentence', f'Repeat exactly: {text}', text),
            ('extract_name', f'Return only the name from: {text}', name),
            ('extract_number', f'Return only the number from: {text}', number),
            ('thanks', f'Thank {name} for reviewing the {subject} in one sentence.',
             f'Thank you, {name}, for reviewing the {subject}.'),
            ('rewrite', f'Create a brief bug title: the {subject} does not open.', f'{subject} does not open.')]:
            add(group, task, user, answer)
    for family in ('thanks', 'rewrite', 'facts', 'label'):
        row = next(r for r in data['train'][72:] if f'-{family}-' in r['id'])
        add('training_replay', family, row['user'], row['answer'])
    add('control', 'greeting', 'Reply with exactly: Hello!', 'Hello!')
    for label, text in [('success', 'completed successfully'), ('warning', 'is delayed but can continue'), ('error', 'failed and cannot continue')]:
        add('control', label, f'Classify as success, warning, or error: The update {text}.', label)
    assert len(result) == len({r['id'] for r in result}) == 32
    return result


def evaluate(model, tokenizer, probes):
    rows = []
    groups = defaultdict(lambda: {'passed': 0, 'total': 0})
    for row in probes:
        generated = generate(model, tokenizer, row['user'])
        passed = generated['stopped_at_eot'] and score(generated['text'], row)
        rows.append({**row, **generated, 'passed': passed})
        groups[row['group']]['passed'] += passed
        groups[row['group']]['total'] += 1
    return {'passed': sum(r['passed'] for r in rows), 'total': len(rows), 'groups': dict(groups), 'rows': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Refusing to overwrite diagnostic')
    torch.set_num_threads(2)
    evidence = json.loads(Path('reports/ember-direct-grounded-v2-34857841073.json').read_text())
    probes = cases(json.loads(Path('direct_answers/grounded-v2-data.json').read_text()))
    model, tokenizer, manifest = load_model(args.bundle, 'full')
    before = evaluate(model, tokenizer, probes)
    adapters = list(args.artifact.rglob('selected-adapter.pt'))
    assert len(adapters) == 1 and sha256(adapters[0]) == evidence['selected_adapter_sha256']
    adapter = torch.load(adapters[0], map_location='cpu', weights_only=True)
    state = adapter['trainable_state']
    expected = {k for k in model.state_dict() if k.startswith(('blocks.5.', 'ln_f.'))}
    assert set(state) == expected
    model.load_state_dict(state, strict=False)
    model.eval().requires_grad_(False)
    after = evaluate(model, tokenizer, probes)
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), scope='paired_copy_diagnosis_not_confirmation',
                  training_launched=False, production_ready=False, baseline=before, candidate=after,
                  selected_adapter_sha256=sha256(adapters[0]), model_sources=manifest['model_sources'],
                  interpretation='Exact normalized references; finite diagnostic probes. Seen entities do not imply seen task frames. No training or fresh confirmation.')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('EMBER_COPY_DIAGNOSTIC=' + json.dumps(result))


if __name__ == '__main__':
    main()
