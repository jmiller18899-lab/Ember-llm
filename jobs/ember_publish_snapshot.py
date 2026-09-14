"""Publish a pinned development snapshot without replacing model checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile

SOURCE = 'b34501cc15fe53a7a922a74a0b1ee88f21bc5ad3'
REPO = 'Jmiller18899/ember-v0.0.53-t4'
DEST = 'development-snapshots/direct-grounded-v2-' + SOURCE[:12]
REPORT = 'reports/ember-direct-grounded-v2-34857841073.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    root = Path.cwd()
    evidence = json.loads((root / REPORT).read_text())
    artifacts = Path(os.environ['EMBER_ARTIFACT_DIR'])
    adapters = list(artifacts.rglob('selected-adapter.pt'))
    if len(adapters) != 1 or digest(adapters[0]) != evidence['selected_adapter_sha256']:
        raise RuntimeError('Experimental adapter mismatch')
    raw_reports = [p for p in artifacts.rglob('*.json') if digest(p) == digest(root / REPORT)]
    if not raw_reports:
        raise RuntimeError('Artifact does not contain the exact archived report')
    if evidence['production_ready'] or evidence['confirmation_gate_passed']:
        raise RuntimeError('Unexpected experimental status')
    if not os.environ.get('HF_TOKEN', '').strip():
        raise RuntimeError('GitHub-held HF_TOKEN is unavailable')
    api = HfApi(token=os.environ['HF_TOKEN'])
    info = api.model_info(REPO)
    if not info.private:
        raise RuntimeError('Expected existing private repository')
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / 'snapshot'
        stage.mkdir()
        subprocess.run(['git', 'archive', '--format=zip', '--output=' + str(stage / 'source.zip'), SOURCE], check=True)
        with zipfile.ZipFile(stage / 'source.zip') as archive:
            if archive.read(REPORT) != (root / REPORT).read_bytes():
                raise RuntimeError('Source archive report mismatch')
        with zipfile.ZipFile(stage / 'experimental-run.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(artifacts.rglob('*')):
                if path.is_symlink():
                    raise RuntimeError('Unexpected artifact symlink')
                if path.is_file():
                    archive.write(path, path.relative_to(artifacts))
        for name, source in [('results.json', REPORT), ('scorecard.json', 'reports/ember-non-prediction-scorecard.json'),
                             ('methodology.md', 'docs/ember-direct-grounded-v2.md')]:
            (stage / name).write_bytes((root / source).read_bytes())
        (stage / 'README.md').write_text(f'''# Ember development snapshot\n\nGitHub source: `{SOURCE}` ([PR #29](https://github.com/jmiller18899-lab/Ember-llm/pull/29)).\n\nThis is experimental source and evidence, not a promoted model release.\nGrounded development improved 0/48 to 12/48. Confirmation scored 6/24,\nall status labels; written content preservation scored 0/18. The strict\nquality gate failed. Routing parameters remained unchanged.\n\n`source.zip` contains the complete tracked source at the pinned revision.\n`experimental-run.zip` preserves the CPU run artifact, including the selected\nadapter, checkpoints, optimizer state and raw reports. The adapter is partial\nexperimental weights, not a standalone model or a replacement for the existing\nfull/INT4 checkpoints. See methodology.md for provenance and limitations.\n\nNo training or deployment was launched by this publication.\n''')
        manifest = {'source_commit': SOURCE, 'repo_id': REPO, 'path': DEST,
                    'source_run': 34857841073, 'source_artifact': 10352838270,
                    'production_ready': False, 'confirmation_gate_passed': False,
                    'selected_adapter_sha256': evidence['selected_adapter_sha256'],
                    'files': {p.name: digest(p) for p in sorted(stage.iterdir())}}
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        existing = api.list_repo_files(REPO, revision=info.sha)
        if any(p.startswith(DEST + '/') for p in existing):
            raise RuntimeError('Snapshot already exists; refusing replacement')
        commit = api.create_commit(repo_id=REPO, parent_commit=info.sha,
            operations=[CommitOperationAdd(path_in_repo=DEST + '/' + p.name, path_or_fileobj=str(p))
                        for p in sorted(stage.iterdir())],
            commit_message='Archive latest Ember direct-answer v2 development work (quality gate failed)')
        for p in stage.iterdir():
            downloaded = hf_hub_download(REPO, DEST + '/' + p.name, revision=commit.oid,
                                         token=os.environ['HF_TOKEN'], local_dir=Path(tmp) / 'verify')
            if digest(downloaded) != digest(p):
                raise RuntimeError('Remote read-back mismatch: ' + p.name)
        print('EMBER_HF_SNAPSHOT=' + json.dumps({'repo_id': REPO, 'revision': commit.oid,
              'path': DEST, 'source_commit': SOURCE, 'verified': True, 'production_ready': False}))


if __name__ == '__main__':
    main()
