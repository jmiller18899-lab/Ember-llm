"""Pinned CPU candidate selection with raw responses and bounded resources."""
import argparse
import base64
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tarfile
import time
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def normalized(text):
    return re.findall(r"[+-]?[^\W_]+(?:[.'’+-][^\W_]+)*|[+-]", text.casefold().rstrip('.!?'))


def exact(text, reference):
    return bool(normalized(text)) and normalized(text) == normalized(reference)


def fetch(url, path, checksum):
    if not path.exists():
        with urllib.request.urlopen(url, timeout=180) as response, open(path, 'wb') as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
    if digest(path) != checksum:
        raise RuntimeError('Download checksum mismatch: ' + str(path))


def prepare(label, target):
    config = json.loads((ROOT / 'benchmarks/qwen_candidates/models.json').read_text())
    target.mkdir(parents=True, exist_ok=True)
    model = config['models'][label]
    fetch(f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{model['file']}", target / 'model.gguf', model['sha256'])
    runtime = config['llama_cpp']
    fetch(runtime['url'], target / 'runtime.tar.gz', runtime['sha256'])
    with tarfile.open(target / 'runtime.tar.gz') as archive:
        archive.extractall(target / 'runtime', filter='data')
    print('PREPARED=' + label, flush=True)


def request(payload=None):
    url = 'http://127.0.0.1:8099/' + ('health' if payload is None else 'v1/chat/completions')
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def cgroup(name):
    path = Path('/sys/fs/cgroup') / name
    return path.read_text().strip() if path.exists() else None


def high_water(pid):
    path = Path(f'/proc/{pid}/status')
    if not path.exists():
        return None
    match = re.search(r'^VmHWM:\s+(\d+) kB', path.read_text(), re.M)
    return int(match[1]) * 1024 if match else None


def run(label, target, output, suite):
    config = json.loads((ROOT / 'benchmarks/qwen_candidates/models.json').read_text())
    spec_path = ROOT / suite
    spec = json.loads(spec_path.read_text())
    if output.exists():
        raise RuntimeError('Refusing to overwrite evidence')
    assert cgroup('memory.max') == str(spec['settings']['memory_limit_bytes'])
    assert cgroup('memory.swap.max') == '0'
    assert digest(target / 'model.gguf') == config['models'][label]['sha256']
    servers = list((target / 'runtime').rglob('llama-server'))
    assert len(servers) == 1
    server = servers[0].resolve()
    env = dict(os.environ, LD_LIBRARY_PATH=str(server.parent), OMP_NUM_THREADS='2')
    command = [str(server), '-m', str((target / 'model.gguf').resolve()), '--host', '127.0.0.1', '--port', '8099',
               '--ctx-size', '2048', '--threads', '2', '--threads-batch', '2', '--parallel', '1',
               '--n-gpu-layers', '0', '--jinja']
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    report = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
              'scope': spec['scope'], 'suite': suite, 'model': label,
              'provenance': config, 'cases_sha256': digest(spec_path), 'settings': spec['settings'],
              'github_sha': os.environ.get('GITHUB_SHA'), 'training_launched': False, 'production_ready': False,
              'memory_limit': cgroup('memory.max'), 'swap_limit': cgroup('memory.swap.max'),
              'cpu_quota': cgroup('cpu.max'), 'cpu_info': Path('/proc/cpuinfo').read_text().split('\n\n')[0],
              'command': command, 'rows': rows, 'manual_review_required': True}
    proc = None
    started = time.monotonic()
    try:
        with open(output.with_suffix('.server.log'), 'w') as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            deadline = time.monotonic() + 180
            while True:
                if proc.poll() is not None:
                    raise RuntimeError('Server exited before readiness')
                try:
                    if request().get('status') == 'ok':
                        break
                except Exception:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('Server readiness timeout')
                time.sleep(1)
            report['load_seconds'] = time.monotonic() - started
            def payload(user):
                return {'model': 'ember-candidate', 'messages': [{'role': 'system', 'content': spec['system']},
                        {'role': 'user', 'content': user}], 'max_tokens': 96, 'temperature': 0,
                        'seed': 42, 'chat_template_kwargs': {'enable_thinking': False}, 'cache_prompt': False}
            report['warmup'] = request(payload('Reply with exactly: Ready.'))
            for case in spec['cases']:
                start = time.monotonic()
                raw = request(payload(case['user']))
                elapsed = time.monotonic() - start
                choice = raw['choices'][0]
                text = choice['message'].get('content') or ''
                row = {**case, 'text': text, 'elapsed_seconds': elapsed, 'raw': raw,
                       'server_peak_rss_bytes': high_water(proc.pid)}
                row['passed'] = None if case['reference'] is None else (
                    choice['finish_reason'] == 'stop' and not choice['message'].get('reasoning_content') and
                    exact(text, case['reference']))
                rows.append(row)
                print(json.dumps({'case': case['id'], 'passed': row['passed'], 'seconds': round(elapsed, 3)}), flush=True)
            report['server_peak_rss_bytes'] = high_water(proc.pid)
            report['memory_peak_bytes'] = cgroup('memory.peak')
            groups = defaultdict(lambda: {'passed': 0, 'total': 0})
            for row in rows:
                if row['passed'] is not None:
                    groups[row['family']]['passed'] += row['passed']
                    groups[row['family']]['total'] += 1
            report['groups'] = dict(groups)
            report['automated_gate_passed'] = len(groups) == 5 and all(v == {'passed': 8, 'total': 8} for v in groups.values())
            report['median_response_seconds'] = statistics.median(r['elapsed_seconds'] for r in rows)
            rates = [r['raw'].get('timings', {}).get('predicted_per_second') for r in rows]
            report['median_decode_tokens_per_second'] = statistics.median([v for v in rates if v is not None]) if any(v is not None for v in rates) else None
            report['status'] = 'completed'
    except Exception as exc:
        report.update(status='error', error=f'{type(exc).__name__}: {exc}', automated_gate_passed=False)
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        report['total_seconds'] = time.monotonic() - started
        raw = (json.dumps(report, indent=2) + '\n').encode()
        output.write_bytes(raw)
        encoded = base64.b64encode(zlib.compress(raw)).decode()
        chunks = [encoded[i:i+16000] for i in range(0, len(encoded), 16000)]
        for i, chunk in enumerate(chunks):
            print('QWEN_EVIDENCE=' + json.dumps({'index': i, 'count': len(chunks), 'sha256': hashlib.sha256(raw).hexdigest(), 'data': chunk}), flush=True)
    if report['status'] != 'completed':
        raise RuntimeError(report['error'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'run'])
    p.add_argument('--model', choices=['0.8B', '2B'], required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--suite', default='benchmarks/qwen_candidates/cases.json')
    a = p.parse_args()
    if a.mode == 'prepare':
        prepare(a.model, a.target)
    else:
        run(a.model, a.target, a.output, a.suite)
