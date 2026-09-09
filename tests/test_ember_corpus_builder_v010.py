"""Exercise real rendering, dedup, disk splits, and saved SentencePiece counts."""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import sqlite3
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "jobs"))
builder = importlib.import_module("ember_corpus_build_v010")
sources = importlib.import_module("ember_corpus_sources_v010")


@pytest.fixture
def cfg():
    return json.loads((ROOT / "config/corpus_v0.1.0.json").read_text())


def fixture_doc(category, index, *, text=None, group_id=None):
    """Test fixtures only. These are never reported as downloaded source data."""
    value = f"{category}-{index:05d}-" + builder.digest(f"{category}:{index}")[:20]
    messages = [
        {"role": "system", "content": "Use lookup with JSON arguments for the request."},
        {"role": "user", "content": f"Find the exact key {value} in this test fixture."},
        {"role": "assistant", "tool_calls": [{"function": {"name": "lookup", "arguments": {"key": value}}}]},
        {"role": "tool", "content": json.dumps({"value": value, "status": "available"})},
        {"role": "assistant", "content": f"The result for {value} is available."},
    ]
    return {"category": category, "source": "test-fixtures/" + category, "source_id": str(index),
            "group_id": group_id or str(index), "text": text or builder.render_agent_record({"messages": messages}),
            "provenance": {"fixture": True, "tool_calls": 1, "has_tool_result_interpretation": True,
                           "real_error_recovery": True}}


def test_canonical_renderer_preserves_arguments_results_and_multiple_calls():
    text = builder.render_agent_record({"messages": [
        {"role": "user", "content": "Copy Qx-42"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "lookup", "arguments": '{"key":"Qx-42"}'}},
            {"function": {"name": "search", "arguments": {"query": "second"}}}]},
        {"role": "tool", "name": "lookup", "content": '{"status":"found"}'},
        {"role": "assistant", "content": "Found."}]})
    assert text.count("<|assistant|>\n<|tool|>\n") == 2
    assert "<|assistant|>\n\n" not in text
    assert '<|tool_result|>\n{"status":"found"}\n' in text
    assert '"key":"Qx-42"' in text
    assert text.count(builder.EOT) == 1


def test_glaive_parser_and_new_renderer_agree():
    corpus, _ = builder.legacy()
    messages, meta = corpus.parse_glaive_row({"system": "SYSTEM: Use weather.", "chat":
        'USER: Weather in Duluth? ASSISTANT: <functioncall> {"name":"weather","arguments":{"location":"Duluth"}} '
        'FUNCTION RESPONSE: {"temperature_f":25} ASSISTANT: It is 25°F.'})
    text = builder.render_openai_messages(messages, drop_tool_call_reasoning=True)
    assert meta["has_tool_result_interpretation"]
    assert '<|assistant|>\n<|tool|>\n{"arguments":{"location":"Duluth"},"name":"weather"}' in text
    builder.validate_document({"category": "tool_result_interpretation", "text": text, "provenance": meta})


def test_smoltalk_excludes_apigen_and_native_tool_payloads(cfg):
    row = {"source": "smol-magpie-ultra", "messages": [
        {"role": "user", "content": "How do I organize a folder?"},
        {"role": "assistant", "content": "Put files with similar uses together and choose clear names."}]}
    doc = sources.smoltalk_document(row, cfg, {"HuggingFaceTB/smoltalk": "immutable-sha"}, 3)
    assert doc["provenance"]["dataset_revision"] == "immutable-sha"
    assert doc["provenance"]["subset"] == "smol-magpie-ultra"
    assert sources.smoltalk_document({**row, "source": "apigen-80k"}, cfg, {}, 3) is None
    row["messages"][1]["tool_calls"] = [{"function": {"name": "x"}}]
    assert sources.smoltalk_document(row, cfg, {}, 3) is None


def test_revision_resolution_ignores_null_synthetic_source(cfg, monkeypatch):
    import huggingface_hub
    seen = []
    class Api:
        def dataset_info(self, name):
            seen.append(name)
            return type("Info", (), {"sha": "a" * 40})()
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    revisions = sources.resolve_revisions(cfg)
    assert None not in seen
    assert len(seen) == len(set(seen)) == 7
    assert cfg["sources"]["agent_trajectories"]["license_join_dataset"] in revisions


@pytest.mark.parametrize("category", ["instruction_following_human", "function_calling", "tool_result_interpretation"])
def test_finite_sources_may_lose_headroom_but_never_the_minimum(cfg, tmp_path, category):
    store = builder.CandidateStore(tmp_path / "candidates.db", cfg, 10_000)
    try:
        minimum = int(store.targets[category] * cfg["candidate_chars_per_token"])
        store.chars[category] = minimum
        store.finish_category(category)
        assert category in store.finite_shortfalls
        store.chars[category] -= 1
        with pytest.raises(RuntimeError, match="source exhausted"):
            store.finish_category(category)
    finally:
        store.close()


def test_infinite_source_still_requires_headroom(cfg, tmp_path):
    store = builder.CandidateStore(tmp_path / "candidates.db", cfg, 10_000)
    try:
        store.chars["code"] = int(store.targets["code"] * cfg["candidate_chars_per_token"])
        with pytest.raises(RuntimeError, match="source exhausted"):
            store.finish_category("code")
    finally:
        store.close()


def test_synthetic_exact_dedup_preserves_case_sensitive_distinct_values(cfg, tmp_path):
    store = builder.CandidateStore(tmp_path / "candidates.db", cfg, 1_000_000)
    try:
        doc = next(sources.synthetic_documents(cfg))
        assert store.add(doc)
        assert not store.add(copy.deepcopy(doc))
        altered = copy.deepcopy(doc)
        old = altered["provenance"]["envelope_copy"]["values"][0]
        new = old.swapcase()
        assert old != new
        altered["text"] = altered["text"].replace(old, new)
        altered["source_id"] = "different-case"
        altered["provenance"]["envelope_copy"]["values"][0] = new
        assert store.add(altered)
        assert store.near.accepted == 0
        assert store.exact_rejected == 1
        assert store.db.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == 2
    finally:
        store.close()


def test_external_near_duplicates_are_still_filtered(cfg, tmp_path):
    store = builder.CandidateStore(tmp_path / "candidates.db", cfg, 1_000_000)
    try:
        first = fixture_doc("code", 1)
        second = copy.deepcopy(first)
        second["text"] = second["text"].replace("test fixture", "test  fixture")
        second["source_id"] = "another-source"
        assert store.add(first)
        assert not store.add(second)
        assert store.near.stats()["exact_duplicates_rejected"] == 1
    finally:
        store.close()


def test_invalid_function_and_recovery_examples_are_rejected():
    doc = fixture_doc("function_calling", 1, text="<|user|>\nHi\n<|assistant|>\nHello\n<|endoftext|>\n")
    with pytest.raises(ValueError, match="no real tool call"):
        builder.validate_document(doc)
    doc = fixture_doc("error_recovery", 1)
    doc["provenance"]["real_error_recovery"] = False
    with pytest.raises(ValueError, match="not a real recovery"):
        builder.validate_document(doc)


def test_wrong_synthetic_argument_is_rejected_even_with_valid_json(cfg):
    for doc in sources.synthetic_documents(cfg):
        meta = doc["provenance"]["envelope_copy"]
        if meta["tools"]:
            break
    lines = doc["text"].splitlines(keepends=True)
    index = lines.index("<|tool|>\n") + 1
    payload = json.loads(lines[index])
    key = next(iter(payload["arguments"]))
    payload["arguments"][key] = "wrong-value"
    lines[index] = builder.envelope_copy.compact(payload) + "\n"
    doc["text"] = "".join(lines)
    with pytest.raises(ValueError, match="values do not match"):
        builder.validate_document(doc)


def test_audit_accepts_a_one_pass_iterator_and_only_counts_selected_documents(cfg):
    relaxed = copy.deepcopy(cfg)
    requirements = relaxed["sources"]["envelope_copy"]["coverage_requirements"]
    for key in requirements:
        requirements[key] = 0
    report = builder.envelope_copy.audit(builder.envelope_copy.iter_documents(cfg, 30), relaxed)
    assert report["documents"] == 30
    assert report["status"] == "PASS"
    # Full coverage must fail on this small set even if a larger candidate pool passed.
    full = builder.envelope_copy.audit(builder.envelope_copy.iter_documents(cfg, 30), cfg)
    assert full["status"] == "FAIL"


def test_split_preserves_whole_source_groups_and_rejects_token_shortage(cfg, tmp_path):
    cfg["mix"] = {"code": 1.0}
    cfg["sources"] = {"code": cfg["sources"]["code"]}
    cfg["validation_ratio"] = 0.25
    cfg["candidate_chars_per_token"] = 30.0
    store = builder.CandidateStore(tmp_path / "candidates.db", cfg, 120)
    class Tokenizer:
        def encode(self, text):
            return [1] * (1 if text == builder.EOT + "\n" else 21 if text.startswith(builder.EOT) else 20)
    try:
        for i in range(12):
            store.add(fixture_doc("code", i, group_id=str(i // 2)))
        totals, _ = builder.select_and_split(store, Tokenizer())
        assert totals["code"] >= 120
        assert not store.db.execute("""SELECT group_id FROM docs WHERE selected=1
             GROUP BY group_id HAVING COUNT(DISTINCT split)>1""").fetchall()
    finally:
        store.close()


def test_short_actual_token_count_fails_even_if_character_goal_passed(cfg, tmp_path):
    cfg["mix"] = {"code": 1.0}
    cfg["sources"] = {"code": cfg["sources"]["code"]}
    store = builder.CandidateStore(tmp_path / "candidates.db", cfg, 100)
    class TinyTokenizer:
        def encode(self, text):
            return [1] * (2 if text.startswith(builder.EOT + "\n") and text != builder.EOT + "\n" else 1)
    try:
        store.add(fixture_doc("code", 1))
        store.add(fixture_doc("code", 2))
        store.finish_category("code")
        with pytest.raises(RuntimeError, match="actual Ember tokens"):
            builder.select_and_split(store, TinyTokenizer())
    finally:
        store.close()


def test_one_large_trajectory_cannot_consume_both_splits(cfg, tmp_path):
    cfg["mix"] = {"error_recovery": 1.0}
    cfg["sources"] = {"error_recovery": cfg["sources"]["error_recovery"]}
    store = builder.CandidateStore(tmp_path / "candidates.db", cfg, 10)
    class Tokenizer:
        def encode(self, text):
            return [1] * (1 if text == builder.EOT + "\n" else 101 if text.startswith(builder.EOT) else 100)
    try:
        assert store.add(fixture_doc("error_recovery", 1))
        assert store.chars["error_recovery"] > store.char_goals["error_recovery"]
        assert not store.full("error_recovery")
        assert store.add(fixture_doc("error_recovery", 2))
        assert store.full("error_recovery")
        totals, splits = builder.select_and_split(store, Tokenizer())
        assert totals["error_recovery"] == 200
        assert splits["error_recovery"]["train_docs"] == 1
        assert splits["error_recovery"]["val_docs"] == 1
    finally:
        store.close()


def test_small_budget_cannot_claim_production_pass(cfg, tmp_path):
    with pytest.raises(ValueError, match="requires --smoke"):
        builder.build(cfg, tmp_path / "out", target_total=1000)


def test_existing_output_cannot_be_overwritten(cfg, tmp_path):
    (tmp_path / "existing.txt").write_text("existing")
    with pytest.raises(ValueError, match="must be empty"):
        builder.build(cfg, tmp_path, smoke=True, target_total=1000)
    assert (tmp_path / "existing.txt").read_text() == "existing"


def test_real_sentencepiece_pipeline_reloads_and_recounts_all_nine_categories(cfg, tmp_path):
    cfg["tokenizer"]["vocab_size"] = 512
    cfg["validation_ratio"] = 0.1
    cfg["candidate_chars_per_token"] = 8.0
    target = 20_000
    def documents():
        for category in cfg["mix"]:
            if category == "envelope_copy":
                yield from sources.synthetic_documents({**cfg, "sources": {**cfg["sources"],
                    "envelope_copy": {**cfg["sources"]["envelope_copy"], "candidate_documents": 1000}}})
            else:
                for index in range(1000):
                    yield fixture_doc(category, index)
    output = tmp_path / "corpus"
    report = builder.build(cfg, output, smoke=True, target_total=target, document_source=documents(),
                           revisions={"test-fixtures": "local"})
    assert report["status"] == "SMOKE_PASS"
    assert report["production_ready"] is False
    assert set(report["category_selected_tokens"]) == set(cfg["mix"])
    assert all(report["category_selected_tokens"][c] >= n for c, n in report["category_target_tokens"].items())
    assert report["envelope_copy_document_audit"]["status"] == "PASS"
    assert report["envelope_copy_audit"]["status"] == "FAIL"  # insufficient sample coverage is disclosed
    _, module = builder.legacy()
    loaded = module.SentencePieceBPETokenizer.from_model_file(output / "ember_tokenizer.model")
    for split in ("train", "val"):
        # Whole-file encoding is an independent cross-check of the streaming count.
        actual = len(loaded.encode((output / f"{split}.txt").read_text()))
        assert actual == report[split + "_tokens"]
    provenance = json.loads((output / "provenance.json").read_text())
    assert sum(d["token_count"] for d in provenance["documents"]) == report["actual_ember_tokens"]
    assert report["gpu_training_launched"] is False
