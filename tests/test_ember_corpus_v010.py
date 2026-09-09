"""Guards for the Ember v0.1.0 re-pretraining corpus draft.

v0.0.33 through v0.0.51 established that a fine-tune small enough to preserve
v0.0.31's routing cannot move copy placement into the JSON argument slot. The
v0.1.0 corpus answers that with a token budget (300M, ~10.8 tokens per parameter
per epoch) and a synthetic envelope-copy slice mixed into pretraining. These
tests pin the config arithmetic, the slice's format, its held-out hygiene, and
its envelope-level format parity. They need no network, tokenizer, or model.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORPUS_CONFIG = ROOT / "config" / "corpus_v0.1.0.json"
LEGACY_CONFIG = ROOT / "ember-v0.0.7-hf-ready.zip"
SEMANTIC_V1 = ROOT / "config" / "ember_semantic_v1.json"
GENERATOR = ROOT / "jobs" / "ember_corpus_envelope_copy_v010.py"
DATA_V026 = ROOT / "jobs" / "ember_sft_data_v026.py"
EOT = "<|endoftext|>"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cfg() -> dict:
    return json.loads(CORPUS_CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def gen():
    return load(GENERATOR, "ember_corpus_envelope_copy_v010")


@pytest.fixture(scope="module")
def sample(gen, cfg) -> list[dict]:
    return list(gen.iter_documents(cfg, 6000))


# --------------------------------------------------------------------------
# Corpus config arithmetic
# --------------------------------------------------------------------------

def test_config_identity_and_authorization_scope(cfg):
    assert cfg["version"] == "0.1.0"
    assert cfg["supersedes"] == "config/corpus_v0.0.7.json"
    assert "no corpus build" in cfg["status"] and "GPU" in cfg["status"]
    assert cfg["hardware_note"].endswith("never launches GPU training.")


def test_mix_sums_to_one_and_names_every_source(cfg):
    assert abs(sum(cfg["mix"].values()) - 1.0) < 1e-9
    assert set(cfg["mix"]) == set(cfg["sources"])
    assert all(v > 0 for v in cfg["mix"].values())


def test_token_gate_is_a_real_budget_for_the_model(cfg):
    target, low, high = cfg["target_total_tokens"], cfg["min_total_tokens"], cfg["max_total_tokens"]
    assert low <= target <= high
    params = cfg["model_reference"]["parameters"]
    assert params == 27_662_848
    tokens_per_parameter = target / params
    assert tokens_per_parameter >= 10.0, tokens_per_parameter
    assert tokens_per_parameter * cfg["recommended_epochs"] >= 20.0
    # This is a re-pretraining of the same architecture, not a size change.
    assert cfg["model_reference"]["architecture_change"].startswith("none")
    assert cfg["tokenizer"]["vocab_size"] == 16384 == cfg["model_reference"]["vocab_size"]
    assert cfg["hard_gates"]["actual_ember_tokens"] == [low, high]


def test_budget_is_at_least_fifteen_times_v007(cfg):
    assert cfg["target_total_tokens"] >= 15 * 15_000_000


def test_every_external_source_has_license_and_loader(cfg):
    for name, source in cfg["sources"].items():
        assert source["license"], name
        assert source["loader"], name
        if name == "envelope_copy":
            assert source["dataset"] is None and source["generator"] == "jobs/ember_corpus_envelope_copy_v010.py"
        else:
            assert source["dataset"], name


def test_finite_sources_are_marked_and_kept_small(cfg):
    finite = {n for n, s in cfg["sources"].items() if s.get("finite_source")}
    assert finite == {"instruction_following_human", "function_calling", "tool_result_interpretation"}
    target = cfg["target_total_tokens"]
    # OASST1 delivered ~3M tokens for v0.0.7; do not ask it for more than that.
    assert cfg["mix"]["instruction_following_human"] * target <= 3_000_000
    # Both Glaive categories together must stay well under the source's ~68M tokens.
    glaive = (cfg["mix"]["function_calling"] + cfg["mix"]["tool_result_interpretation"]) * target
    assert glaive <= 30_000_000


def test_envelope_copy_share_is_a_small_pretraining_slice(cfg):
    share = cfg["mix"]["envelope_copy"]
    assert 0.03 <= share <= 0.06
    block = cfg["sources"]["envelope_copy"]
    assert block["dedup"] == "exact_only"
    assert abs(sum(block["shape_weights"].values()) - 1.0) < 1e-9
    assert list(block["shape_weights"]) == block["shapes"]
    kinds = set(block["value_kinds"])
    for tool, fields in block["tools"].items():
        assert 1 <= len(fields) <= 2, tool
        for field, allowed in fields.items():
            assert allowed and set(allowed) <= kinds, (tool, field)


def test_builder_requirements_cover_the_known_pipeline_gaps(cfg):
    joined = "\n".join(cfg["builder_requirements"])
    for needle in ("smoltalk", "envelope_copy", "finite_source", "<|assistant|>\\n<|tool|>", "stream", "envelope_copy_audit"):
        assert needle in joined, needle


# --------------------------------------------------------------------------
# Generator: determinism, format, hygiene
# --------------------------------------------------------------------------

def test_generation_is_deterministic(gen, cfg):
    first = [d["text"] for d in gen.iter_documents(cfg, 300)]
    second = [d["text"] for d in gen.iter_documents(cfg, 300)]
    assert first == second
    assert len(set(first)) == len(first), "documents must be exactly distinct"


def test_every_document_is_canonical_ember_text(gen, cfg, sample):
    tools = cfg["sources"]["envelope_copy"]["tools"]
    for doc in sample:
        text = doc["text"]
        assert text.startswith("<|system|>\n")
        assert text.endswith("\n" + EOT + "\n") and text.count(EOT) == 1
        assert "<|assistant|>\n\n" not in text, "bare tool-call turns must not carry an empty content line"
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line == "<|tool|>":
                assert lines[i - 1] == "<|assistant|>"
                payload = json.loads(lines[i + 1])
                assert lines[i + 1] == gen.compact(payload)
                assert list(payload) == ["arguments", "name"]
                assert set(payload["arguments"]) == set(tools[payload["name"]])
        for value in doc["values"]:
            assert text.count(value) >= 2, (doc["id"], value)


def test_tool_shapes_call_tools_and_direct_shapes_do_not(sample):
    for doc in sample:
        if doc["shape"].startswith("tool_call") or doc["shape"] == "two_turn":
            assert doc["tools"] and "<|tool|>" in doc["text"]
        else:
            assert not doc["tools"] and "<|tool|>" not in doc["text"]
        if doc["shape"] == "two_turn":
            assert doc["text"].count("<|tool|>") == 2 and len(doc["tools"]) == 2
        if doc["shape"] == "tool_call_multi_arg":
            assert len(doc["values"]) == 2


def test_available_tools_prompts_advertise_every_called_tool(sample):
    seen = 0
    for doc in sample:
        if "\nAVAILABLE_TOOLS\n" not in doc["text"]:
            continue
        seen += 1
        listed = {spec["name"] for spec in json.loads(doc["text"].split("\nAVAILABLE_TOOLS\n", 1)[1].split("\n", 1)[0])}
        assert set(doc["tools"]) <= listed, doc["id"]
    assert seen > 500


def test_tool_results_are_grounded_in_the_answer(gen, sample):
    checked = 0
    for doc in sample:
        if doc["shape"] != "tool_call_with_result":
            continue
        text = doc["text"]
        result_json = text.split("<|tool_result|>\n", 1)[1].split("\n", 1)[0]
        answer = text.rsplit("<|assistant|>\n", 1)[1].split("\n" + EOT)[0]
        result = json.loads(result_json)
        if "error" in result:
            assert result["error"] in answer
        else:
            # Every scalar in the result payload (other than the echoed argument) is quoted in the answer.
            for key, value in result.items():
                if isinstance(value, (int, float, str)) and key not in ("title",):
                    assert str(value) in answer, (doc["id"], key, answer)
        checked += 1
    assert checked > 500


def test_expression_results_are_exact(gen):
    assert gen.expression_result("12+30") == "42"
    assert gen.expression_result("50-8") == "42"
    assert gen.expression_result("6*7") == "42"
    assert gen.expression_result("(40+2)*1") == "42"
    assert gen.expression_result("84/2") == "42"
    assert gen.expression_result("5/2") == "2.5"
    with pytest.raises(ValueError):
        gen.expression_result("2**3")


def test_frozen_held_out_values_are_excluded(gen, cfg, sample):
    block = cfg["sources"]["envelope_copy"]
    exclusions = gen.Exclusions.from_config(block)
    semantic = json.loads(SEMANTIC_V1.read_text(encoding="utf-8"))
    for case in semantic["cases"]:
        for variant in case.get("argument_variants", []) or []:
            for value in variant.values():
                assert exclusions.blocks(str(value)), value
    for value in load(DATA_V026, "v026_for_exclusions").HELD_OUT_VALUES:
        assert exclusions.blocks(value), value
    for literal in ("Detroit", "detroit, MI", "Asia/Tokyo", "ticket ZV82 closed status", "report 9037", "Toronto"):
        assert exclusions.blocks(literal), literal
    for doc in sample:
        for value in doc["values"]:
            assert not exclusions.blocks(value), (doc["id"], value)
    assert "Detroit" not in "".join(d["text"] for d in sample)


def test_structural_audit_has_no_per_document_failures_on_a_sample(gen, cfg, sample):
    relaxed = copy.deepcopy(cfg)
    reqs = relaxed["sources"]["envelope_copy"]["coverage_requirements"]
    for key in ("min_documents_per_tool", "min_documents_per_shape", "min_documents_per_kind", "min_distinct_values_per_kind"):
        reqs[key] = 0
    report = gen.audit(sample, relaxed)
    assert report["status"] == "PASS", report["failures"]
    assert report["held_out_leakage"] is False
    assert 60 <= report["approx_tokens_per_document"] <= 200


def test_candidate_documents_cover_the_share_with_headroom(gen, cfg, sample):
    block = cfg["sources"]["envelope_copy"]
    relaxed = copy.deepcopy(cfg)
    for key in relaxed["sources"]["envelope_copy"]["coverage_requirements"]:
        if key.startswith("min_"):
            relaxed["sources"]["envelope_copy"]["coverage_requirements"][key] = 0
    per_doc = gen.audit(sample, relaxed)["approx_tokens_per_document"]
    share_tokens = cfg["mix"]["envelope_copy"] * cfg["target_total_tokens"]
    assert block["candidate_documents"] * per_doc >= cfg["candidate_headroom"] * share_tokens


def test_full_candidate_set_passes_format_parity_audit(gen, cfg):
    documents = list(gen.iter_documents(cfg))
    block = cfg["sources"]["envelope_copy"]
    assert len(documents) == block["candidate_documents"]
    report = gen.audit(documents, cfg)
    assert report["status"] == "PASS", report["failures"]
    single = {t for t, f in block["tools"].items() if len(f) == 1}
    multi = {t for t, f in block["tools"].items() if len(f) == 2}
    for shape in ("tool_call", "tool_call_choice", "tool_call_with_result", "two_turn"):
        assert set(report["tools_by_shape"][shape]) == single, shape
    assert set(report["tools_by_shape"]["tool_call_multi_arg"]) == multi
    for kind in block["value_kinds"]:
        assert report["documents_per_kind"][kind] >= block["coverage_requirements"]["min_documents_per_kind"]
    assert report["approx_tokens_at_4.8_chars_per_token"] >= cfg["mix"]["envelope_copy"] * cfg["target_total_tokens"]


def test_cli_assert_coverage_exits_nonzero_when_underfilled(gen, capsys):
    assert gen.main(["--count", "500", "--assert-coverage"]) == 1
    out = capsys.readouterr().out
    assert '"status": "FAIL"' in out
    assert gen.main(["--count", "500"]) == 0
