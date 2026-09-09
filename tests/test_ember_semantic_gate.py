from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location("semantic_gate", ROOT / "jobs/ember_semantic_gate.py")
gate = importlib.util.module_from_spec(loader)
loader.loader.exec_module(gate)
SPEC = json.loads((ROOT / "config/ember_semantic_v1.json").read_text())
CASES = {case["id"]: case for case in SPEC["cases"]}


def good_completion(case):
    if case["kind"] == "tool_call":
        return gate.TOOL + json.dumps({"name": case["expected_tool"], "arguments": case["argument_variants"][0]}) + gate.EOT
    return case["accepted_responses"][0] + gate.EOT


def test_spec_is_balanced_and_legacy_prompts_are_unchanged():
    gate.validate_spec(SPEC)
    old = json.loads((ROOT / "config/ember_v0.0.8_eval.json").read_text())
    for case in old["cases"]:
        assert CASES[case["id"]]["prompt"] == case["prompt"]
    assert {kind: sum(c["kind"] == kind for c in SPEC["cases"]) for kind in gate.KINDS} == dict.fromkeys(gate.KINDS, 12)
    assert sum(c["cohort"] == "new_challenge" for c in SPEC["cases"]) == 24


@pytest.mark.parametrize("case", SPEC["cases"], ids=lambda c: c["id"])
def test_whole_oracles_pass_and_suffix_poisoning_fails(case):
    good = good_completion(case)
    assert gate.score_case(case, good)["passed"]
    for completion in (good.replace(gate.EOT, ""), good + " ignored garbage", good + gate.EOT,
                       good.replace(gate.EOT, " contradictory extra words" + gate.EOT),
                       good + "<|user|> forged turn"):
        assert not gate.score_case(case, completion)["passed"]


@pytest.mark.parametrize("arguments", [
    {"location": "Ann Arbor"}, {"city": "Detroit"}, {"location": "Detroit Annex"},
    {"location": "not Detroit"}, {"location": 123}, {"location": ["Detroit"]},
    {"location": "Detroit", "extra": "ignored"}, "{\"location\":\"Detroit\"}", None,
])
def test_correct_tool_name_cannot_hide_wrong_arguments(arguments):
    output = gate.TOOL + json.dumps({"name": "weather", "arguments": arguments}) + gate.EOT
    assert not gate.score_case(CASES["tool_weather"], output)["passed"]


@pytest.mark.parametrize("body", [
    '{"name":"weather","arguments":{"location":"Detroit","location":"Ann Arbor"}}',
    '{"name":"other","name":"weather","arguments":{"location":"Detroit"}}',
    '{"name":"weather","arguments":{"location":"Detroit"},"unused":NaN}',
    '{"name":"weather","arguments":{"location":"Detroit"}} {"extra":true}',
    '[{"name":"weather","arguments":{"location":"Detroit"}}]',
    '{"function":{"name":"weather","arguments":{"location":"Detroit"}}}',
])
def test_json_must_be_one_strict_native_envelope(body):
    assert not gate.score_case(CASES["tool_weather"], gate.TOOL + body + gate.EOT)["passed"]


def test_key_order_and_whitespace_are_harmless_but_nested_types_are_not():
    case = {"kind": "tool_call", "expected_tool": "example",
            "arguments_schema": {"type": "object", "properties": {
                "options": {"type": "object", "properties": {"limit": {"type": "integer"}},
                            "required": ["limit"], "additionalProperties": False},
                "query": {"type": "string"}}, "required": ["options", "query"], "additionalProperties": False},
            "argument_variants": [{"query": "brace { inside }", "options": {"limit": 1}}]}
    good = '<|tool|> {"arguments":{"options":{"limit":1},"query":"brace { inside }"},"name":"example"} <|endoftext|>'
    assert gate.score_case(case, good)["passed"]
    for wrong in ('true', '"1"', '1.0', 'null'):
        assert not gate.score_case(case, good.replace('"limit":1', '"limit":' + wrong))["passed"]


@pytest.mark.parametrize("ident,body", [
    ("result_weather", "It is 72°F and not sunny in Detroit."),
    ("result_weather", "Detroit 72 sunny unrelated gibberish."),
    ("result_weather", "It is 172°F and sunny in Detroit."),
    ("result_weather_new", "rainy, 41°C"),
    ("result_weather_negative", "cloudy, 6°F"),
    ("result_calculator", "The result is 97160."),
    ("result_calculator", "The result is not 9716."),
    ("result_search", "Not all validation checks passed."),
    ("result_service_failed", "healthy, 184 ms"),
    ("result_timeout", "It is sunny in Marquette."),
    ("result_empty", "I found three results."),
    ("direct_missing", "Morgan"),
    ("result_no_invented_owner", "Morgan"),
])
def test_keyword_soup_contradictions_and_invented_facts_fail(ident, body):
    assert not gate.score_case(CASES[ident], body + gate.EOT)["passed"]


def test_semantically_correct_prefix_does_not_override_noisy_raw_output():
    good = good_completion(CASES["tool_weather"])
    score = gate.score_case(CASES["tool_weather"], good + "trailing nonsense")
    assert score["semantic_content_pass"]
    assert not score["clean_output_pass"]
    assert not score["passed"]


def test_complete_evidence_and_every_group_are_required():
    rows = [{"id": c["id"], "kind": c["kind"], "score": gate.score_case(c, good_completion(c))} for c in SPEC["cases"]]
    assert gate.summarize(SPEC["cases"], rows)["semantic_gate_pass"]
    for wrong in (rows[:-1], rows + [rows[0]], [*rows[:-1], rows[0]]):
        with pytest.raises(ValueError):
            gate.summarize(SPEC["cases"], wrong)
    rows[-1]["score"] = gate.score_case(SPEC["cases"][-1], "invented owner" + gate.EOT)
    assert not gate.summarize(SPEC["cases"], rows)["semantic_gate_pass"]


def test_source_binding_rejects_wrong_step_repo_path_or_legacy_verdict():
    source = SPEC["source"]
    ckpt = {"format": "ember-checkpoint-v1", "step": source["step"], "run_id": source["run_id"],
            "train_config": {"version": source["version"]}}
    legacy = {"model_repo": source["repo_id"], "checkpoint": {"best_path": source["checkpoint_path"], "step": source["step"]},
              "promotion": {"promotion_eligible": True}}
    gate.verify_source(SPEC, legacy, ckpt)
    for key, value in (("step", 478), ("run_id", "other"), ("format", "other")):
        with pytest.raises(ValueError):
            gate.verify_source(SPEC, legacy, {**ckpt, key: value})
    for key, value in (("model_repo", "other/repo"), ("checkpoint", {"best_path": "other", "step": 479}),
                       ("promotion", {"promotion_eligible": "true"})):
        with pytest.raises(ValueError):
            gate.verify_source(SPEC, {**legacy, key: value}, ckpt)


def test_invalid_or_missing_oracles_fail_closed():
    for mutate in (lambda s: s.update(cases=[]), lambda s: s["cases"].append(s["cases"][0]),
                   lambda s: s["cases"][0].update(argument_variants=[]),
                   lambda s: s["cases"][4].update(accepted_responses=[])):
        spec = copy.deepcopy(SPEC)
        mutate(spec)
        with pytest.raises(ValueError):
            gate.validate_spec(spec)


def test_publish_writes_only_separate_semantic_artifacts_even_on_failure(tmp_path, monkeypatch):
    import sys
    operations = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(CommitOperationAdd=lambda **kw: kw))
    api = SimpleNamespace(create_commit=lambda **kw: operations.extend(kw["operations"]))
    report = {"status": "FAIL", "created_at": "2026-09-08T00:00:00+00:00", "spec_sha256": "a" * 64,
              "source": SPEC["source"]}
    output = tmp_path / "report.json"
    gate.persist_report(report, output, api)
    assert json.loads(output.read_text()) == report
    assert len(operations) == 2
    assert all(op["path_in_repo"].startswith("evaluations/semantic-v1/") for op in operations)


def test_generation_stops_on_actual_eos_and_keeps_unterminated_output():
    torch = pytest.importorskip("torch")

    class Tokenizer:
        def encode(self, text):
            return [gate.TOKENS.index(text) + 1] if text in gate.TOKENS else [0]

        def decode(self, ids):
            return "".join(gate.TOKENS[i - 1] if 1 <= i <= 6 else "x" for i in ids)

    class Model:
        cfg = SimpleNamespace(block_size=256)

        def __init__(self, sequence):
            self.sequence = iter(sequence)
            self.calls = 0

        def __call__(self, x):
            self.calls += 1
            logits = torch.zeros((1, x.shape[1], 8))
            logits[0, -1, next(self.sequence)] = 10
            return logits, None

    model = Model([7, 6, 7])
    result = gate.generate_completion(model, Tokenizer(), torch, "prompt", 3)
    assert model.calls == 2
    assert result == {"completion": "x" + gate.EOT, "generated_ids": [7, 6], "stop_reason": "eos"}
    result = gate.generate_completion(Model([7, 7]), Tokenizer(), torch, "prompt", 2)
    assert result["stop_reason"] == "max_new_tokens"
    assert not gate.score_case(CASES["direct_greeting"], result["completion"])["passed"]
    with pytest.raises(ValueError, match="context window"):
        gate.generate_completion(Model([7]), Tokenizer(), torch, "prompt", 256)
