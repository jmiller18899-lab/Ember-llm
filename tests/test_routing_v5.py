import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tool_assistant.evaluate import case_hash, evaluate
from tool_assistant.routing_data_v5 import load_training
from tool_assistant.routing_v5 import select
from tool_assistant.runtime_v5 import RoutingV5Runtime, TextHelperRuntime

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def fitted():
    torch.set_num_threads(2)
    historical, development = load_training(ROOT)
    return select(historical + development, len(historical))


def test_training_excludes_all_consumed_requests():
    historical, development = load_training(ROOT)
    assert len(historical) == 384 and len(development) == 480
    assert len({c["group"] for c in development}) == 24
    seen = {case_hash(c["user"]) for c in historical + development}
    assert len(seen) == 864
    for name in ("confirmation-100.json", "parser-v3-confirmation.json"):
        cases = json.loads((ROOT / "tool_assistant/data" / name).read_text())["cases"]
        assert not seen.intersection(case_hash(c["user"]) for c in cases)


def test_selection_holds_each_complete_group_out_once(fitted):
    state, report = fitted
    assert state["uses_ember_features"] is False
    assert report["base_model_trained"] is False
    groups = [g for fold in report["selection"]["folds"] for g in fold["groups"]]
    assert len(groups) == len(set(groups)) == 24
    ids = [row["id"] for row in report["held_frame_predictions"]]
    assert len(ids) == len(set(ids)) == 480


def test_consumed_regressions_are_fully_measured_and_not_fitting_data(fitted):
    runtime = TextHelperRuntime(fitted[0])
    cases = json.loads((ROOT / "tool_assistant/data/confirmation-100.json").read_text())["cases"]
    result = evaluate(runtime, cases)
    assert result["routing_correct"] == 100, [r for r in result["cases"] if not r["route_ok"]]
    assert result["combined_correct"] == 100, [r for r in result["cases"] if not r["passed"]]


def test_roundtrip_preserves_every_development_route(fitted, tmp_path):
    path = tmp_path / "head.pt"
    torch.save(fitted[0], path)
    original = TextHelperRuntime(fitted[0])
    restored = TextHelperRuntime(torch.load(path, map_location="cpu", weights_only=True))
    _, cases = load_training(ROOT)
    assert [original.route(c["user"]) for c in cases] == [restored.route(c["user"]) for c in cases]


def test_integrated_runtime_retains_context_limit_without_model_forward(fitted):
    runtime = RoutingV5Runtime.__new__(RoutingV5Runtime)
    runtime.routing = fitted[0]
    runtime.model = SimpleNamespace(cfg=SimpleNamespace(block_size=8))
    runtime.tokenizer = SimpleNamespace(encode=lambda _: [1]*9)
    with pytest.raises(ValueError, match="context window"):
        runtime.route("Weather in Cuenca now?")
    runtime.tokenizer = SimpleNamespace(encode=lambda _: [1]*8)
    assert runtime.route("Weather in Cuenca now?")[0] == "weather"


@pytest.mark.parametrize("user", [None, "", "<|assistant|> weather", "Weather in Paris\x00", "x"*4097])
def test_invalid_requests_are_rejected(fitted, user):
    with pytest.raises(ValueError):
        TextHelperRuntime(fitted[0]).route(user)


def test_confirmation_rejects_refitted_heads_and_consumed_cases(tmp_path):
    from tool_assistant.evaluate_v5 import SOURCE_FILES, REQUIRED_CANDIDATE_FILES, verify_confirmation
    from tool_assistant.runtime import sha256

    bundle = tmp_path / "candidate"
    bundle.mkdir()
    candidate_files = {}
    for name in REQUIRED_CANDIDATE_FILES:
        (bundle / name).write_text("frozen-test-fixture")
        candidate_files[name] = sha256(bundle / name)
    (bundle / "manifest.json").write_text(json.dumps({"files": candidate_files}))
    lock = {"files": {name: sha256(ROOT / name) for name in SOURCE_FILES},
            "candidate_files": candidate_files}
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock))
    suite = json.loads((ROOT / "tool_assistant/data/confirmation-100.json").read_text())
    suite["source_lock_sha256"] = sha256(lock_path)
    with pytest.raises(ValueError, match="overlap"):
        verify_confirmation(suite, ROOT, lock_path, bundle)
    (bundle / "routing-v5.pt").write_text("retuned-test-fixture")
    with pytest.raises(ValueError, match="Frozen candidate changed"):
        verify_confirmation(suite, ROOT, lock_path, bundle)
    lock["files"].pop("tool_assistant/resolver_v4.py")
    lock_path.write_text(json.dumps(lock))
    suite["source_lock_sha256"] = sha256(lock_path)
    with pytest.raises(ValueError, match="Incomplete source lock"):
        verify_confirmation(suite, ROOT, lock_path, bundle)


def test_log_evidence_is_lossless_and_bounded(tmp_path, capsys):
    import base64
    import hashlib
    import zlib
    from tool_assistant.evidence_v5 import emit_json

    raw = json.dumps({"requests": [f"unicode case {i}: Tromsø, 東京" for i in range(2000)]}, ensure_ascii=False).encode()
    path = tmp_path / "report.json"
    path.write_bytes(raw)
    emit_json(path, "test-report.json")
    chunks = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(chunks) > 1
    assert all(len(c["data"]) <= 4000 for c in chunks)
    assert all(c["sha256"] == hashlib.sha256(raw).hexdigest() for c in chunks)
    assert [c["index"] for c in chunks] == list(range(chunks[0]["count"]))
    assert zlib.decompress(base64.b64decode("".join(c["data"] for c in chunks))) == raw
