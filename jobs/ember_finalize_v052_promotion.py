"""Finalize Ember v0.0.52 promotion state after formal evaluation.

Promotion is PASS only when BOTH the formal repository evaluator and the newer
strict copy+structure gate are true. Otherwise the candidate remains saved and
the current promoted model is left untouched.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from huggingface_hub import HfApi, hf_hub_download

REPO = "Jmiller18899/ember-v0.0.52-t4"


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    api = HfApi(token=token)
    with tempfile.TemporaryDirectory(prefix="ember-v052-finalize-") as td:
        work = Path(td)
        state_path = Path(hf_hub_download(
            repo_id=REPO, repo_type="model", filename="run-state.json",
            token=token, local_dir=work,
        ))
        state = json.loads(state_path.read_text(encoding="utf-8"))
        formal = bool(state.get("promotion_eligible"))
        strict = bool(state.get("strict_promotion_eligible"))
        promoted = formal and strict
        state.update({
            "promotion": "PASS" if promoted else "BLOCKED",
            "promotion_blocked_reason": None if promoted else {
                "formal_promotion_eligible": formal,
                "strict_promotion_eligible": strict,
                "note": "Candidate remains saved; current promoted model was not replaced.",
            },
            "production_authorized": False,
            "promoted_candidate": promoted,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        out = work / "run-state-final.json"
        out.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        api.upload_file(
            repo_id=REPO, repo_type="model", path_or_fileobj=str(out),
            path_in_repo="run-state.json",
            commit_message="Finalize Ember v0.0.52 gated promotion state",
        )
        print(f"FORMAL_PROMOTION_ELIGIBLE={str(formal).lower()}", flush=True)
        print(f"STRICT_PROMOTION_ELIGIBLE={str(strict).lower()}", flush=True)
        print(f"EMBER_V052_PROMOTION={'PASS' if promoted else 'BLOCKED'}", flush=True)


if __name__ == "__main__":
    main()
