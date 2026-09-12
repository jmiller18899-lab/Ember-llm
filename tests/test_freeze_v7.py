import json
from pathlib import Path
import shutil

import pytest

from tool_assistant.evaluate import case_hash
from tool_assistant.freeze_v6 import make_lock as make_v6_lock
from tool_assistant.freeze_v7 import LOCK_NAME, SOURCE_FILES, make_lock, prior_hashes, verify_confirmation, verify_freeze
from tool_assistant.runtime import sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def frozen(tmp_path):
    root, bundle = tmp_path / "source", tmp_path / "candidate"
    for name in SOURCE_FILES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    old_path = root / "tool_assistant/data/routing-v5-source-lock.json"
    old = json.loads(old_path.read_text())
    files = {}
    for name in old["candidate_files"]:
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("offline-v7-freeze-test-fixture")
        files[name] = sha256(target)
    manifest = json.dumps({"files": files})
    (bundle / "manifest.json").write_text(manifest)
    (root / "reports/ember-routing-v5-manifest.json").write_text(manifest)
    old.update(candidate_files=files, candidate_manifest_sha256=sha256(bundle / "manifest.json"))
    old_path.write_text(json.dumps(old))
    (root / "tool_assistant/data/routing-v6-source-lock.json").write_text(json.dumps(make_v6_lock(root)))
    (root / LOCK_NAME).write_text(json.dumps(make_lock(root)))
    return root, bundle


def test_verifies_both_prior_freezes_and_all_candidate_files(frozen):
    root, bundle = frozen
    result = verify_freeze(root, bundle)
    assert result["source_files_verified"] == len(SOURCE_FILES)
    assert result["base_v6"]["source_files_verified"] == 48
    assert result["base_v6"]["base_v5"]["candidate_files_verified"] == 38


@pytest.mark.parametrize("name", ["tool_assistant/routing_v7.py", "tool_assistant/resolver_v7.py",
                                  "tool_assistant/runtime_v6.py"])
def test_changed_new_or_control_source_is_rejected(frozen, name):
    root, bundle = frozen
    (root / name).write_text("changed after freeze")
    with pytest.raises(ValueError, match="Frozen v7 source changed"):
        verify_freeze(root, bundle)


@pytest.mark.parametrize("name", ["routing-v5.pt", "model-full.pt", "model-int4.pt"])
def test_changed_archived_candidate_is_rejected(frozen, name):
    root, bundle = frozen
    (bundle / name).write_text("replacement bytes")
    with pytest.raises(ValueError, match="Frozen candidate changed"):
        verify_freeze(root, bundle)


def test_incomplete_source_lock_is_rejected(frozen):
    root, bundle = frozen
    path = root / LOCK_NAME
    lock = json.loads(path.read_text())
    lock["files"].pop(".github/workflows/ember-routing-v7-run.yml")
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="Incomplete v7 source freeze"):
        verify_freeze(root, bundle)


def test_consumed_and_unbound_confirmation_are_rejected(frozen):
    root, _ = frozen
    suite = json.loads((root / "tool_assistant/data/routing-v6-confirmation.json").read_text())
    suite.update(routing_revision="request-routing-v7", source_lock_sha256=sha256(root / LOCK_NAME))
    with pytest.raises(ValueError, match="overlap"):
        verify_confirmation(suite, root)
    suite["source_lock_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bound"):
        verify_confirmation(suite, root)


def test_generated_development_requests_are_also_consumed():
    hashes = prior_hashes(ROOT)
    powers = {2: "squared", 3: "cubed"}
    for a in (-3, 2, 5):
        for b in (-4, 1):
            for p, q in ((2, 2), (3, 2), (2, 3), (3, 3)):
                for operator in ("plus", "minus"):
                    assert case_hash(f"Calculate {a} {powers[p]} {operator} {b} {powers[q]}.") in hashes
