"""Independent v7 overlay freeze; v5 bytes and the v6 source stay intact."""
import ast
from datetime import datetime, timezone
import json

from .evaluate import case_hash, validate_cases
from .freeze_v6 import SOURCE_FILES as V6_SOURCE_FILES, prior_hashes as v6_prior_hashes, verify_freeze as verify_v6
from .routing_v7 import REVISION
from .runtime import sha256

SOURCE_FILES = tuple(sorted(set(V6_SOURCE_FILES) | {
    "tool_assistant/data/routing-v6-source-lock.json", "tool_assistant/data/routing-v6-confirmation.json",
    "tool_assistant/data/routing-v7-development.json", "tool_assistant/data/routing-v7-prior-additions.json",
    "tool_assistant/resolver_v7.py", "tool_assistant/routing_v7.py", "tool_assistant/runtime_v7.py",
    "tool_assistant/freeze_v7.py", "tool_assistant/evaluate_v7.py", "tool_assistant/live_smoke_v7.py",
    "tests/test_routing_v7.py", "tests/test_freeze_v7.py",
    ".github/workflows/ember-routing-v7.yml", ".github/workflows/ember-routing-v7-run.yml",
}))
LOCK_NAME = "tool_assistant/data/routing-v7-source-lock.json"


def make_lock(root):
    base_path = root / "tool_assistant/data/routing-v6-source-lock.json"
    base = json.loads(base_path.read_text())
    if set(base["files"]) != set(V6_SOURCE_FILES):
        raise ValueError("Incomplete v6 control freeze")
    for name, digest in base["files"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Frozen v6 control changed: {name}")
    return {"schema_version": 1, "routing_revision": REVISION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": {name: sha256(root / name) for name in SOURCE_FILES},
            "base_source_lock_sha256": sha256(base_path), "candidate_files": base["candidate_files"],
            "candidate_manifest_sha256": base["candidate_manifest_sha256"],
            "base_model_trained": False, "routing_head_refitted": False,
            "interpretation": "V7 source frozen before new requests. All 38 archived v5 candidate files and the 48-file v6 control source remain unchanged; no fitted state is added."}


def verify_freeze(root, bundle):
    path = root / LOCK_NAME
    lock = json.loads(path.read_text())
    if lock.get("routing_revision") != REVISION or set(lock.get("files", {})) != set(SOURCE_FILES):
        raise ValueError("Incomplete v7 source freeze")
    for name, digest in lock["files"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Frozen v7 source changed: {name}")
    base_path = root / "tool_assistant/data/routing-v6-source-lock.json"
    base = json.loads(base_path.read_text())
    if (sha256(base_path) != lock["base_source_lock_sha256"]
            or base["candidate_files"] != lock["candidate_files"]
            or base["candidate_manifest_sha256"] != lock["candidate_manifest_sha256"]):
        raise ValueError("V7 must preserve the original control and candidate")
    return {"source_lock_sha256": sha256(path), "source_files_verified": len(SOURCE_FILES),
            "base_v6": verify_v6(root, bundle), "routing_revision": REVISION}


def prior_hashes(root):
    seen = v6_prior_hashes(root)
    suite = json.loads((root / "tool_assistant/data/routing-v6-confirmation.json").read_text())
    seen.update(case_hash(c["user"]) for c in suite["cases"])
    development = json.loads((root / "tool_assistant/data/routing-v7-development.json").read_text())
    for key in ("routing", "arithmetic"):
        seen.update(case_hash(c["user"]) for c in development[key])
    seen.update(case_hash(s) for s in development["fallback"])
    seen.update(case_hash("Calculate " + s) for s in development["arithmetic_refusals"])
    additions = json.loads((root / "tool_assistant/data/routing-v7-prior-additions.json").read_text())
    seen.update(case_hash(s) for s in additions["requests"])
    for name in SOURCE_FILES:
        if name.endswith(".py"):
            tree = ast.parse((root / name).read_text())
            seen.update(case_hash(n.value) for n in ast.walk(tree)
                        if isinstance(n, ast.Constant) and isinstance(n.value, str))
    return seen


def verify_confirmation(suite, root):
    if suite.get("routing_revision") != REVISION or suite.get("source_lock_sha256") != sha256(root / LOCK_NAME):
        raise ValueError("New requests must be bound to the frozen v7 source")
    return validate_cases(suite["cases"], prior_hashes(root))
