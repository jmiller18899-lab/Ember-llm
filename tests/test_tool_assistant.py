import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tool_assistant import binary, family
from tool_assistant.evaluate import arithmetic, score, validate_cases
from tool_assistant.runtime import Runtime, hidden


def test_binary_roundtrip(tmp_path):
    torch.manual_seed(92)
    x = torch.randn(24, 7, dtype=torch.double)
    y = (x[:, 0] > 0).long()
    state = binary.fit_ridge(x, y, 2, 0.01)
    torch.save(state, tmp_path / "head.pt")
    loaded = torch.load(tmp_path / "head.pt", weights_only=True)
    torch.testing.assert_close(binary.predict(state, x)[1], binary.predict(loaded, x)[1], rtol=0, atol=0)


def test_original_family_selection_reproduces():
    torch.set_num_threads(2)
    data = json.loads(Path("tool_assistant/data/router-training.json").read_text())
    by_id = {c["id"]: c for c in data["cases"]}
    fitted = family.select_family([by_id[i] for i in data["family_ids"]])
    assert fitted["cv"]["mode"] == "hybrid"
    assert fitted["cv"]["ridge"] == 0.01
    assert fitted["cv"]["correct"] == 256
    assert len(fitted["vectorizer"]["vocab"]) == 2026


@pytest.mark.parametrize("expression,result", [("18*(7+3)", 180), ("-17+42", 25), ("(6.5/100)*280", 18.2)])
def test_arithmetic_meaning(expression, result):
    assert arithmetic(expression) == pytest.approx(result)


@pytest.mark.parametrize("expression", ["__import__('os').system('true')", "2**99999", "(1).__class__", "1/0"])
def test_unsafe_or_undefined_arithmetic_rejected(expression):
    with pytest.raises((ValueError, ZeroDivisionError)):
        arithmetic(expression)


def test_extra_location_text_is_failure():
    case = {"route": "weather", "arguments": {"location": "Dakar"}}
    output = {"route": "weather", "call": {"name": "weather", "arguments": {"location": "Dakar tomorrow morning"}}}
    assert not score(case, output)["passed"]


def test_direct_misroute_is_measured_without_crashing():
    case = {"route": "calculator", "result": 3}
    assert not score(case, {"route": "direct", "call": None})["passed"]


def test_dispatch_uses_only_predicted_arguments():
    runtime = Runtime.__new__(Runtime)
    runtime.plan = lambda _: {"route": "calculator", "status": "tool_call", "call": {"name": "calculator", "arguments": {"expression": "4+7"}}}
    assert runtime.run("ignored", {"calculator": arithmetic})["result"] == 11
    assert runtime.run("ignored", {})["status"] == "tool_unavailable"


def test_direct_and_missing_arguments_never_dispatch():
    runtime = Runtime.__new__(Runtime)
    runtime.plan = lambda _: {"route": "direct", "status": "direct_answer_unavailable", "call": None}
    assert runtime.run("explain a cache", {})["call"] is None


def test_hook_removed_after_forward_exception():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(5)])
            self.cfg = SimpleNamespace(block_size=32)
        def forward(self, *args):
            raise RuntimeError("synthetic failure")
    model = Model()
    with pytest.raises(RuntimeError, match="synthetic failure"):
        hidden(model, SimpleNamespace(encode=lambda _: [1, 2]), "request")
    assert not model.blocks[4]._forward_hooks


def test_confirmation_is_novel_and_balanced():
    cases = json.loads(Path("tool_assistant/data/confirmation-100.json").read_text())["cases"]
    old = json.loads(Path("tool_assistant/data/prior-string-hashes.json").read_text())
    assert sum(validate_cases(cases, old).values()) == 100


def test_confirmation_source_lock_is_unchanged():
    path = Path("tool_assistant/data/candidate-source-lock.json")
    lock = json.loads(path.read_text())
    for name, expected in lock["files"].items():
        assert hashlib.sha256((Path("tool_assistant") / name).read_bytes()).hexdigest() == expected
    suite = json.loads(Path("tool_assistant/data/confirmation-100.json").read_text())
    assert suite["source_lock_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
