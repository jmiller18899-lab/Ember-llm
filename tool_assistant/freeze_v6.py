"""Freeze a source overlay while keeping every original candidate byte."""
import ast
from datetime import datetime, timezone
import json

from .evaluate import case_hash, validate_cases
from .evaluate_v5 import SOURCE_FILES as V5_SOURCE_FILES, prior_hashes as v5_prior_hashes
from .live_smoke import verify_freeze as verify_v5
from .routing_v6 import REVISION
from .runtime import sha256

SOURCE_FILES = tuple(sorted(set(V5_SOURCE_FILES) | {
    "tool_assistant/data/routing-v5-source-lock.json",
    "tool_assistant/data/routing-v5-confirmation.json",
    "tool_assistant/data/routing-v6-development.json",
    "reports/ember-routing-v5-manifest.json",
    "tool_assistant/routing_v6.py", "tool_assistant/runtime_v6.py",
    "tool_assistant/freeze_v6.py", "tool_assistant/evaluate_v6.py",
    "tool_assistant/live_smoke_v6.py", "tool_assistant/live_smoke.py",
    "tool_assistant/live_services.py", "tool_assistant/restore_live_candidate.py",
    "tests/test_routing_v6.py", "tests/test_freeze_v6.py",
    "tests/test_live_services.py", "tests/test_restore_live_candidate.py",
    ".github/workflows/ember-routing-v6.yml",
    ".github/workflows/ember-routing-v6-run.yml",
}))
LOCK_NAME = "tool_assistant/data/routing-v6-source-lock.json"


def make_lock(root):
    original_path = root / "tool_assistant/data/routing-v5-source-lock.json"
    original = json.loads(original_path.read_text())
    if set(original["files"]) != set(V5_SOURCE_FILES):
        raise ValueError("Incomplete original source freeze")
    for name, digest in original["files"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Original source changed: {name}")
    if sha256(root / "reports/ember-routing-v5-manifest.json") != original["candidate_manifest_sha256"]:
        raise ValueError("Original manifest changed")
    return {"schema_version": 1, "routing_revision": REVISION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": {name: sha256(root / name) for name in SOURCE_FILES},
            "base_source_lock_sha256": sha256(original_path),
            "candidate_files": original["candidate_files"],
            "candidate_manifest_sha256": original["candidate_manifest_sha256"],
            "base_model_trained": False, "routing_head_refitted": False,
            "interpretation": "Source overlay frozen before new requests. All 38 candidate files are the already archived v5 bytes; the new runtime is loaded from the separately hashed source overlay."}


def verify_freeze(root, bundle):
    lock_path = root / LOCK_NAME
    lock = json.loads(lock_path.read_text())
    if lock.get("routing_revision") != REVISION or set(lock.get("files", {})) != set(SOURCE_FILES):
        raise ValueError("Incomplete v6 source freeze")
    for name, digest in lock["files"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Frozen v6 source changed: {name}")
    original_path = root / "tool_assistant/data/routing-v5-source-lock.json"
    original = json.loads(original_path.read_text())
    if (lock["base_source_lock_sha256"] != sha256(original_path)
            or lock["candidate_files"] != original["candidate_files"]
            or lock["candidate_manifest_sha256"] != original["candidate_manifest_sha256"]):
        raise ValueError("V6 must retain the original candidate freeze")
    verified = verify_v5(root, bundle=bundle)
    if verified["candidate_manifest_sha256"] != lock["candidate_manifest_sha256"]:
        raise ValueError("The original archived manifest is required")
    return {"source_lock_sha256": sha256(lock_path), "source_files_verified": len(SOURCE_FILES),
            "base_v5": verified, "routing_revision": REVISION}


def prior_hashes(root):
    seen = v5_prior_hashes(root)
    prior = json.loads((root / "tool_assistant/data/routing-v5-confirmation.json").read_text())
    seen.update(case_hash(c["user"]) for c in prior["cases"])
    development = json.loads((root / "tool_assistant/data/routing-v6-development.json").read_text())
    seen.update(case_hash(c["user"]) for c in development["positive"])
    seen.update(case_hash(s) for s in development["fallback"])
    for name in SOURCE_FILES:
        if name.endswith(".py"):
            tree = ast.parse((root / name).read_text())
            seen.update(case_hash(n.value) for n in ast.walk(tree)
                        if isinstance(n, ast.Constant) and isinstance(n.value, str))
    return seen


def verify_confirmation(suite, root):
    if suite.get("routing_revision") != REVISION or suite.get("source_lock_sha256") != sha256(root / LOCK_NAME):
        raise ValueError("New requests must be bound to the frozen v6 overlay")
    return validate_cases(suite["cases"], prior_hashes(root))
