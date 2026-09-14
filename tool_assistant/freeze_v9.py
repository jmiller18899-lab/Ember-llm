"""Freeze the contact overlay before new confirmation requests are authored."""
import ast
from datetime import datetime, timezone
import json
from .evaluate import case_hash, validate_cases
from .freeze_v8 import SOURCE_FILES as BASE_FILES, prior_hashes as base_prior, verify_freeze as verify_base
from .runtime import sha256
from .context_v9 import REVISION

SOURCE_FILES = tuple(sorted(set(BASE_FILES) | {
    'tool_assistant/data/routing-v8-source-lock.json',
    'tool_assistant/data/routing-v8-confirmation.json',
    'tool_assistant/context_v9.py', 'tool_assistant/runtime_v9.py',
    'tool_assistant/evaluate_v9.py', 'tool_assistant/live_smoke_v9.py',
    'tool_assistant/freeze_v9.py', 'tests/test_routing_v9.py',
    '.github/workflows/ember-routing-v9.yml', '.github/workflows/ember-routing-v9-run.yml',
}))
LOCK_NAME='tool_assistant/data/routing-v9-source-lock.json'


def make_lock(root):
    base_path=root/'tool_assistant/data/routing-v8-source-lock.json'
    base=json.loads(base_path.read_text())
    if set(base['files']) != set(BASE_FILES):raise ValueError('Incomplete base freeze')
    for name,digest in base['files'].items():
        if sha256(root/name)!=digest:raise ValueError(f'Frozen v8 source changed: {name}')
    return {'schema_version':1,'routing_revision':REVISION,
            'created_at':datetime.now(timezone.utc).isoformat(),
            'files':{name:sha256(root/name) for name in SOURCE_FILES},
            'base_source_lock_sha256':sha256(base_path),
            'candidate_files':base['candidate_files'],
            'candidate_manifest_sha256':base['candidate_manifest_sha256'],
            'base_model_trained':False,'routing_head_refitted':False}


def verify_freeze(root,bundle):
    lock=json.loads((root/LOCK_NAME).read_text())
    if lock['routing_revision']!=REVISION or set(lock['files'])!=set(SOURCE_FILES):
        raise ValueError('Incomplete v9 freeze')
    for name,digest in lock['files'].items():
        if sha256(root/name)!=digest:raise ValueError(f'Frozen v9 source changed: {name}')
    base_path=root/'tool_assistant/data/routing-v8-source-lock.json'
    base=json.loads(base_path.read_text())
    if sha256(base_path)!=lock['base_source_lock_sha256'] or any(
        base[k]!=lock[k] for k in ('candidate_files','candidate_manifest_sha256')):
        raise ValueError('Original candidate must be preserved')
    return {'source_lock_sha256':sha256(root/LOCK_NAME),'source_files_verified':len(SOURCE_FILES),
            'base_v8':verify_base(root,bundle),'routing_revision':REVISION}


def prior_hashes(root):
    seen=base_prior(root)
    seen.update(case_hash(c['user']) for c in json.loads((root/'tool_assistant/data/routing-v8-confirmation.json').read_text())['cases'])
    for name in SOURCE_FILES:
        if name.endswith('.py'):
            seen.update(case_hash(n.value) for n in ast.walk(ast.parse((root/name).read_text()))
                        if isinstance(n,ast.Constant) and isinstance(n.value,str))
    return seen


def verify_confirmation(suite,root):
    if suite.get('routing_revision')!=REVISION or suite.get('source_lock_sha256')!=sha256(root/LOCK_NAME):
        raise ValueError('Confirmation must bind the frozen source')
    return validate_cases(suite['cases'],prior_hashes(root))
