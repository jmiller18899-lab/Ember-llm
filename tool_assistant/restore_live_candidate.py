"""Restore archived fitted bytes after a rebuild, without changing the freeze."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .evidence_v5 import emit_json
from .runtime import sha256

ARCHIVES = {
    "base": {"run_id": 34565654365, "artifact_id": 10185934664},
    "v4": {"run_id": 34691874565, "artifact_id": 10297810403},
    "v5": {"run_id": 34694061949, "artifact_id": 10297993516},
}
HEADS = {"family.pt": "base", "router-full.pt": "base", "router-int4.pt": "base",
         "routing-v4-full.pt": "v4", "routing-v4-int4.pt": "v4", "routing-v5.pt": "v5"}


def restore(bundle, archives, lock_path, manifest_path, report_path):
    if report_path.exists():
        raise ValueError("Refusing to overwrite restoration evidence")
    lock = json.loads(lock_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if sha256(manifest_path) != lock["candidate_manifest_sha256"] or manifest["files"] != lock["candidate_files"]:
        raise ValueError("The archived manifest does not match the original freeze")
    expected = lock["candidate_files"]
    if not set(HEADS).issubset(expected):
        raise ValueError("Incomplete candidate freeze")
    sources, rows = {}, []
    # Validate everything before changing any fitted file. Model weights are
    # preserved and checked, never replaced by the restoration step.
    for name, digest in expected.items():
        dest = (bundle / name).resolve()
        if not dest.is_relative_to(bundle.resolve()):
            raise ValueError("Invalid candidate path")
        if name not in HEADS:
            if sha256(dest) != digest:
                raise ValueError(f"A preserved candidate file differs: {name}")
            continue
        archive = archives / HEADS[name]
        matches = [p for p in archive.rglob(name) if p.is_file() and p.resolve().is_relative_to(archive.resolve())]
        matches = [p for p in matches if sha256(p) == digest]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one checksum-matching archived file: {name}")
        sources[name] = matches[0]
        rows.append({"file": name, "archive": HEADS[name], "archived_sha256": digest,
                     "rebuilt_sha256": sha256(dest)})
    before = sha256(bundle / "manifest.json")
    for name, source in sources.items():
        shutil.copyfile(source, bundle / name)
    shutil.copyfile(manifest_path, bundle / "manifest.json")
    if any(sha256(bundle / name) != digest for name, digest in expected.items()):
        raise RuntimeError("Restored candidate does not match the freeze")
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "source_lock_sha256": sha256(lock_path), "rebuild_manifest_sha256": before,
        "restored_manifest_sha256": sha256(bundle / "manifest.json"),
        "verified_files": len(expected), "restored_heads": rows, "archives": ARCHIVES,
        "interpretation": "Original checksum-matching fitted bytes restored from retained Actions artifacts. Rebuilt helper bytes were not used for live measurement. Model weights and the freeze are unchanged."}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    emit_json(report_path, "live-candidate-restore.json")
    emit_json(bundle / "manifest.json", "live-candidate-manifest.json")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    restore(args.bundle, args.archives, root / "tool_assistant/data/routing-v5-source-lock.json",
            root / "reports/ember-routing-v5-manifest.json", args.report)


if __name__ == "__main__":
    main()
