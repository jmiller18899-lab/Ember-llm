import json
from pathlib import Path

import pytest
import torch

from tool_assistant.routing_data_v4 import LABELS, development_cases, load_training
from tool_assistant.routing_v4 import fit_features, fit_head, predict, select, transform, validate_request

ROOT = Path(__file__).resolve().parent.parent


def test_development_groups_and_confirmation_separation():
    historical, development = load_training(ROOT)
    assert len(historical) == 384
    assert len(development) == 320
    for group in {c["group"] for c in development}:
        cases = [c for c in development if c["group"] == group]
        assert {label: sum(c["route"] == label for c in cases) for label in LABELS} == dict.fromkeys(LABELS, 4)
    seen = {" ".join(c["user"].casefold().split()) for c in historical + development}
    for name in ("confirmation-100.json", "parser-v3-confirmation.json"):
        suite = json.loads((ROOT / "tool_assistant/data" / name).read_text())
        assert not seen.intersection(" ".join(c["user"].casefold().split()) for c in suite["cases"])


def test_text_control_really_ignores_frozen_hidden_features(tmp_path):
    torch.set_num_threads(2)
    cases = development_cases()[:40]
    texts = [c["user"] for c in cases]
    hidden = torch.randn(len(cases), 9, dtype=torch.double)
    feats = fit_features(texts, hidden)
    x = transform(texts, hidden, feats, 0.0)
    state = {"features": feats, "embedding_weight": 0.0,
             "weight": fit_head(x, torch.tensor([LABELS.index(c["route"]) for c in cases]), 0.01)}
    original = predict(state, texts, hidden)[1]
    torch.save(state, tmp_path / "state.pt")
    loaded = torch.load(tmp_path / "state.pt", weights_only=True)
    torch.testing.assert_close(original, predict(loaded, texts, hidden*200 + 30)[1], rtol=0, atol=0)
    assert (original.argmax(1) == torch.tensor([LABELS.index(c["route"]) for c in cases])).all()


def test_folds_do_not_split_paraphrase_variants():
    torch.set_num_threads(2)
    cases = development_cases()
    # A small, separate training prefix keeps this a real fitting check.
    historical = [{**c, "id": "history-"+c["id"], "user": "Earlier: "+c["user"], "group": "historical-training"}
                  for c in cases[:20]]
    cases = historical + cases
    generator = torch.Generator().manual_seed(71)
    features = torch.randn(len(cases), 8, generator=generator, dtype=torch.double)
    _, report = select(cases, features, len(historical))
    folds = report["selection"]["folds"]
    assert len({g for f in folds for g in f["groups"]}) == 16
    assert sum(len(f["groups"]) for f in folds) == 16
    assert sum(f["total"] for f in folds) == 320
    ids = [c["id"] for c in report["held_frame_predictions"]]
    assert len(ids) == len(set(ids)) == 320
    assert all(not i.startswith("history-") for i in ids)


@pytest.mark.parametrize("user", [None, "", "  ", "<|assistant|> answer"])
def test_invalid_requests_rejected(user):
    with pytest.raises(ValueError):
        validate_request(user)


def test_invalid_features_rejected():
    with pytest.raises(ValueError, match="finite"):
        select(development_cases(), torch.full((320, 2), float("nan")), 20)


def test_routing_runtime_retains_parser_and_single_dispatch(monkeypatch):
    from tool_assistant import runtime_v4

    torch.set_num_threads(2)
    texts = ["Compute (12+8)/4.", "Tell me a story about a calculator.", "Calculate 7!"]
    vectors = torch.eye(3, dtype=torch.double)
    features = fit_features(texts, vectors)
    x = transform(texts, vectors, features, 1.0)
    state = {"features": features, "embedding_weight": 1.0,
             "weight": fit_head(x, torch.tensor([2, 0, 2]), 0.01)}
    runtime = runtime_v4.RoutingRuntime.__new__(runtime_v4.RoutingRuntime)
    runtime.model = runtime.tokenizer = None
    runtime.routing = state
    monkeypatch.setattr(runtime_v4, "hidden", lambda _m, _t, p: vectors[texts.index(p.split("<|user|>\n")[1].split("\n<|assistant|>")[0])])
    calls = []
    def calculator(**arguments):
        calls.append(arguments)
        return 5
    result = runtime.run(texts[0], {"calculator": calculator})
    assert result["status"] == "tool_result"
    assert result["result"] == 5
    assert calls == [{"expression": "(12+8)/4"}]
    assert runtime.run(texts[1], {"calculator": calculator})["status"] == "direct_answer_unavailable"
    assert runtime.run(texts[2], {"calculator": calculator})["status"] == "needs_clarification"
    assert len(calls) == 1


def test_confirmation_requires_frozen_fitted_candidate(tmp_path):
    from tool_assistant.evaluate_v4 import SOURCE_FILES, REQUIRED_CANDIDATE_FILES, verify_confirmation
    from tool_assistant.runtime import sha256

    bundle = tmp_path / "candidate"
    bundle.mkdir()
    lock_path = tmp_path / "lock.json"
    source_files = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    lock_path.write_text(json.dumps({"files": source_files}))
    suite = {"source_lock_sha256": sha256(lock_path), "cases": []}
    with pytest.raises(ValueError, match="fitted candidate"):
        verify_confirmation(suite, ROOT, lock_path, bundle)
    candidate_files = {}
    for name in REQUIRED_CANDIDATE_FILES:
        (bundle / name).write_text("frozen")
        candidate_files[name] = sha256(bundle / name)
    lock_path.write_text(json.dumps({"files": source_files, "candidate_files": candidate_files}))
    (bundle / "manifest.json").write_text(json.dumps({"files": candidate_files}))
    suite["source_lock_sha256"] = sha256(lock_path)
    (bundle / "routing-v4-full.pt").write_text("retuned")
    with pytest.raises(ValueError, match="Frozen candidate changed"):
        verify_confirmation(suite, ROOT, lock_path, bundle)


def test_confirmation_rejects_consumed_requests(tmp_path):
    from tool_assistant.evaluate_v4 import SOURCE_FILES, REQUIRED_CANDIDATE_FILES, verify_confirmation
    from tool_assistant.runtime import sha256

    bundle = tmp_path / "candidate"
    bundle.mkdir()
    files = {}
    for name in REQUIRED_CANDIDATE_FILES:
        (bundle / name).write_text("fixture")
        files[name] = sha256(bundle / name)
    (bundle / "manifest.json").write_text(json.dumps({"files": files}))
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps({"files": {name: sha256(ROOT / name) for name in SOURCE_FILES},
                                     "candidate_files": files}))
    suite = json.loads((ROOT / "tool_assistant/data/confirmation-100.json").read_text())
    suite["source_lock_sha256"] = sha256(lock_path)
    with pytest.raises(ValueError, match="overlap"):
        verify_confirmation(suite, ROOT, lock_path, bundle)
