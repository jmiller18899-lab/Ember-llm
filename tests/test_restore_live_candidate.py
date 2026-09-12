import json

import pytest

from tool_assistant.restore_live_candidate import HEADS, restore
from tool_assistant.runtime import sha256


@pytest.fixture
def fixture(tmp_path):
    bundle, archives = tmp_path / "bundle", tmp_path / "archives"
    bundle.mkdir()
    files = {}
    for name, group in HEADS.items():
        p = archives / group / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("original fitted bytes " + name)
        files[name] = sha256(p)
        (bundle / name).write_text("different rebuilt bytes " + name)
    model = bundle / "model-full.pt"
    model.write_text("unchanged model")
    files[model.name] = sha256(model)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": files}))
    (bundle / "manifest.json").write_text("rebuilt metadata")
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"candidate_manifest_sha256": sha256(manifest), "candidate_files": files}))
    return bundle, archives, lock, manifest, tmp_path / "report.json"


def test_restores_only_verified_archived_heads_and_keeps_model(fixture):
    result = restore(*fixture)
    bundle, archives, lock, manifest, report = fixture
    assert result["verified_files"] == 7
    assert len(result["restored_heads"]) == 6
    assert (bundle / "model-full.pt").read_text() == "unchanged model"
    assert (bundle / "manifest.json").read_bytes() == manifest.read_bytes()
    assert report.exists()
    with pytest.raises(ValueError, match="overwrite"):
        restore(*fixture)


def test_bad_archive_is_rejected_before_any_candidate_mutation(fixture):
    bundle, archives, *_ = fixture
    before = {p.name: p.read_bytes() for p in bundle.iterdir()}
    (archives / "v5" / "routing-v5.pt").write_text("not the frozen helper")
    with pytest.raises(ValueError, match="checksum-matching"):
        restore(*fixture)
    assert {p.name: p.read_bytes() for p in bundle.iterdir()} == before


def test_changed_model_is_not_repaired_or_ignored(fixture):
    bundle = fixture[0]
    (bundle / "model-full.pt").write_text("different model")
    with pytest.raises(ValueError, match="preserved"):
        restore(*fixture)
