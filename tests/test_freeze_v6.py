import json
from pathlib import Path
import shutil

import pytest

from tool_assistant.freeze_v6 import LOCK_NAME, SOURCE_FILES, make_lock, verify_confirmation, verify_freeze
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
        target.write_text("offline-freeze-test-fixture")
        files[name] = sha256(target)
    manifest = json.dumps({"files": files})
    (bundle / "manifest.json").write_text(manifest)
    (root / "reports/ember-routing-v5-manifest.json").write_text(manifest)
    old.update(candidate_files=files, candidate_manifest_sha256=sha256(bundle / "manifest.json"))
    old_path.write_text(json.dumps(old))
    lock = make_lock(root)
    (root / LOCK_NAME).write_text(json.dumps(lock))
    return root, bundle


def test_freeze_checks_the_source_overlay_and_all_candidate_files(frozen):
    root, bundle = frozen
    result = verify_freeze(root, bundle)
    assert result["base_v5"]["candidate_files_verified"] == 38
    assert result["source_files_verified"] == len(SOURCE_FILES)
    (root / "tool_assistant/routing_v6.py").write_text("changed after freezing")
    with pytest.raises(ValueError, match="Frozen v6 source changed"):
        verify_freeze(root, bundle)


@pytest.mark.parametrize("name", ["routing-v5.pt", "model-full.pt", "model-int4.pt"])
def test_refitted_or_replaced_candidate_is_rejected(frozen, name):
    root, bundle = frozen
    (bundle / name).write_text("replacement bytes")
    with pytest.raises(ValueError, match="Frozen candidate changed"):
        verify_freeze(root, bundle)


def test_incomplete_lock_is_rejected(frozen):
    root, bundle = frozen
    path = root / LOCK_NAME
    lock = json.loads(path.read_text())
    lock["files"].pop("tool_assistant/runtime_v6.py")
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="Incomplete v6 source freeze"):
        verify_freeze(root, bundle)


def test_changed_base_freeze_is_rejected(frozen):
    root, bundle = frozen
    path = root / LOCK_NAME
    lock = json.loads(path.read_text())
    lock["candidate_files"]["routing-v5.pt"] = "0" * 64
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="retain the original"):
        verify_freeze(root, bundle)


def test_consumed_confirmation_cannot_be_relabelled_fresh(frozen):
    root, _ = frozen
    suite = json.loads((root / "tool_assistant/data/routing-v5-confirmation.json").read_text())
    suite.update(routing_revision="context-routing-v6", source_lock_sha256=sha256(root / LOCK_NAME))
    with pytest.raises(ValueError, match="overlap"):
        verify_confirmation(suite, root)
    suite["source_lock_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bound"):
        verify_confirmation(suite, root)
