#!/usr/bin/env python3
"""Build the v0.1.0 corpus on CPU, with disk-backed selection and explicit gates.

The v0.0.7 archive is read by checksum, never modified. This program builds data
and a tokenizer only; it has no model-training, upload, or deployment operation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "jobs"
if str(JOBS) not in sys.path:
    sys.path.insert(0, str(JOBS))
import ember_corpus_envelope_copy_v010 as envelope_copy

PACKAGE_SHA256 = "27e8f7c80317652a22b3d58a0bd474724491a685dfe9e20c0b997b7c5907a289"
EOT = "<|endoftext|>"
_PACKAGE_DIRECTORY = None


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def document_tokens(tokenizer, text: str, *, first: bool = False) -> int:
    """Count the exact contribution to a concatenated EOT-delimited text stream.

    SentencePiece adds a dummy prefix to an independently encoded string. The
    EOT sentinel supplies the preceding document boundary for every later row,
    preventing one extra dummy-prefix token per document in corpus statistics.
    A special token is indivisible, so BPE cannot merge across this boundary.
    """
    if first:
        return len(tokenizer.encode(text))
    prefix = EOT + "\n"
    return len(tokenizer.encode(prefix + text)) - len(tokenizer.encode(prefix))


def legacy():
    """Import only corpus/data/tokenizer code from the immutable source package."""
    global _PACKAGE_DIRECTORY
    if _PACKAGE_DIRECTORY is None:
        archive = ROOT / "ember-v0.0.7-hf-ready.zip"
        if file_hash(archive) != PACKAGE_SHA256:
            raise RuntimeError("authoritative v0.0.7 archive checksum mismatch")
        _PACKAGE_DIRECTORY = tempfile.TemporaryDirectory(prefix="ember-v010-package-")
        with zipfile.ZipFile(archive) as package:
            package.extractall(_PACKAGE_DIRECTORY.name)
        package_root = Path(_PACKAGE_DIRECTORY.name) / "ember"
        # Existing repo tests may already have loaded this same authoritative src.
        if "src.corpus" not in sys.modules:
            sys.path.insert(0, str(package_root))
    return importlib.import_module("src.corpus"), importlib.import_module("src.tokenizer")


def targets_from_mix(total: int, mix: dict) -> dict[str, int]:
    if total <= 0 or not mix or any(float(v) <= 0 for v in mix.values()):
        raise ValueError("positive total and positive category shares required")
    if abs(sum(mix.values()) - 1.0) > 1e-9:
        raise ValueError("corpus mix must sum to 1.0")
    out = {key: int(total * value) for key, value in mix.items()}
    out[next(reversed(out))] += total - sum(out.values())
    if min(out.values()) < 1:
        raise ValueError("token budget must allocate at least one token to every category")
    return out


def render_agent_record(record: dict) -> str:
    """Canonical v0.1.0 role boundaries; preserve values and raw result content."""
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("nonempty messages required")
    compact = envelope_copy.compact
    chunks = []
    if record.get("tools"):
        chunks.append("<|system|>\nAVAILABLE_TOOLS\n" + compact(record["tools"]) + "\n")
    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        calls = message.get("tool_calls") or []
        if role == "assistant" and calls:
            if content:
                chunks.append("<|assistant|>\n" + content + "\n")
            for call in calls:
                function = call.get("function", call)
                args = function.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                name = function.get("name")
                if not isinstance(name, str) or not name or not isinstance(args, dict):
                    raise ValueError("a real named tool call with object arguments is required")
                chunks.append("<|assistant|>\n<|tool|>\n" + compact({"arguments": args, "name": name}) + "\n")
        elif role in ("system", "user", "assistant", "tool"):
            token = "tool_result" if role == "tool" else role
            chunks.append(f"<|{token}|>\n{content}\n")
        else:
            raise ValueError(f"unsupported role: {role}")
    return "".join(chunks) + EOT + "\n"


def render_openai_messages(messages, *, tools=None, drop_tool_call_reasoning=False):
    corpus, _ = legacy()
    clean = corpus.sanitize_agent_messages(messages, drop_tool_call_reasoning=drop_tool_call_reasoning)
    return render_agent_record({"messages": clean, "tools": tools})


def validate_document(doc: dict) -> None:
    text = doc["text"]
    if not text.endswith(EOT + "\n") or text.count(EOT) != 1:
        raise ValueError("document must have one terminal end token")
    calls = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line != "<|tool|>":
            continue
        if i == 0 or lines[i - 1] != "<|assistant|>" or i + 1 >= len(lines):
            raise ValueError("tool call must directly follow a bare assistant role")
        payload = json.loads(lines[i + 1])
        if (not isinstance(payload, dict) or set(payload) != {"name", "arguments"}
                or not isinstance(payload["name"], str) or not payload["name"]
                or not isinstance(payload["arguments"], dict)):
            raise ValueError("invalid tool envelope")
        if lines[i + 1] != envelope_copy.compact(payload):
            raise ValueError("tool envelope must be canonical JSON")
        calls.append(payload)
    category, provenance = doc["category"], doc["provenance"]
    if category == "function_calling" and not calls:
        raise ValueError("function_calling document has no real tool call")
    if category == "tool_result_interpretation":
        if not calls or "<|tool_result|>\n" not in text:
            raise ValueError("interpretation document has no real call and result")
        if not provenance.get("has_tool_result_interpretation"):
            raise ValueError("interpretation document has no observed answer after its result")
    if category == "error_recovery" and (not calls or not provenance.get("real_error_recovery")):
        raise ValueError("error_recovery document is not a real recovery")
    if category == "envelope_copy":
        meta = provenance["envelope_copy"]
        if [c["name"] for c in calls] != meta["tools"]:
            raise ValueError("synthetic tool metadata does not match the serialized calls")
        actual = [v for call in calls for v in call["arguments"].values()]
        # sorted-key JSON can reorder multi-argument fields; compare the multiset.
        if calls and Counter(actual) != Counter(meta["values"]):
            raise ValueError("synthetic argument values do not match their source metadata")
        if any(not isinstance(v, str) for v in actual):
            raise ValueError("synthetic arguments must be strings")


class CandidateStore:
    """Candidate text, exact hashes, ordering, and selection live in SQLite."""

    def __init__(self, path: Path, cfg: dict, target_total: int):
        corpus, _ = legacy()
        self.cfg = cfg
        self.targets = targets_from_mix(target_total, cfg["mix"])
        self.char_goals = {c: int(n * cfg["candidate_chars_per_token"] * cfg["candidate_headroom"])
                           for c, n in self.targets.items()}
        self.chars, self.counts, self.rejected = Counter(), Counter(), Counter()
        self.first_groups = defaultdict(set)
        self.finite_shortfalls = {}
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA cache_size=-16384")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.execute("""CREATE TABLE docs (
            id INTEGER PRIMARY KEY, category TEXT, source TEXT, source_id TEXT,
            group_id TEXT, text TEXT, sha TEXT UNIQUE, provenance TEXT,
            selection_key TEXT, split_key TEXT, output_key TEXT,
            tokens INTEGER DEFAULT 0, selected INTEGER DEFAULT 0, split TEXT)""")
        for name, columns in (("selection", "category, selection_key"),
                              ("splitting", "category, selected, split_key"),
                              ("output", "split, output_key")):
            self.db.execute(f"CREATE INDEX {name} ON docs ({columns})")
        self.near = corpus.ExactNearDeduper(
            threshold=cfg["near_dedup_threshold"], num_perm=cfg["near_dedup_num_perm"],
            max_shingles=cfg.get("near_dedup_max_shingles", 256))
        self.exact_rejected = 0
        self.seed = cfg["seed"]
        frozen = json.loads((ROOT / "config/ember_semantic_v1.json").read_text())
        # Exact benchmark prompts must not be copied wholesale into an external source.
        self.frozen_prompts = [" ".join(c["prompt"].split()).casefold() for c in frozen["cases"]
                               if isinstance(c.get("prompt"), str)]

    def close(self):
        self.db.close()

    def full(self, category: str) -> bool:
        return self.chars[category] >= self.char_goals[category] and len(self.first_groups[category]) >= 2

    def add(self, doc: dict) -> bool:
        category = doc["category"]
        if category not in self.targets:
            raise ValueError(f"unknown category: {category}")
        if self.full(category):
            return False
        text = doc["text"]
        try:
            validate_document(doc)
        except (ValueError, KeyError, TypeError):
            self.rejected[category] += 1
            return False
        if category != "envelope_copy":
            normalized = " ".join(text.split()).casefold()
            if any(prompt in normalized for prompt in self.frozen_prompts):
                self.rejected[category] += 1
                return False
        sha = digest(text)
        if self.db.execute("SELECT 1 FROM docs WHERE sha=?", (sha,)).fetchone():
            self.exact_rejected += 1
            return False
        # Case-sensitive exact dedup preserves distinct synthetic literal values.
        if category != "envelope_copy" and not self.near.accept(text):
            return False
        group = doc.get("group_id", doc["source_id"])
        self.db.execute("""INSERT INTO docs (category,source,source_id,group_id,text,sha,
            provenance,selection_key,split_key,output_key) VALUES (?,?,?,?,?,?,?,?,?,?)""", (
            category, doc["source"], doc["source_id"], group, text, sha,
            json.dumps(doc["provenance"], ensure_ascii=False),
            digest(f"select:{category}:{doc['source_id']}:{sha}:{self.seed}"),
            digest(f"split:{doc['source']}:{group}:{self.seed}"),
            digest(f"order:{category}:{doc['source_id']}:{sha}:{self.seed}")))
        self.chars[category] += len(text)
        self.counts[category] += 1
        if len(self.first_groups[category]) < 2:
            self.first_groups[category].add((doc["source"], group))
        if sum(self.counts.values()) % 1000 == 0:
            self.db.commit()
        return True

    def finish_category(self, category: str):
        if self.full(category):
            return
        minimum = int(self.targets[category] * self.cfg["candidate_chars_per_token"])
        if self.cfg["sources"][category].get("finite_source") and self.chars[category] >= minimum:
            self.finite_shortfalls[category] = {
                "candidate_chars": self.chars[category], "hard_min_chars": minimum,
                "preferred_chars": self.char_goals[category]}
            return
        raise RuntimeError(f"{category}: source exhausted with {self.chars[category]} candidate chars; "
                           f"required {minimum if self.cfg['sources'][category].get('finite_source') else self.char_goals[category]}")

    def documents(self, category: str, *, selected=False):
        where = "category=?" + (" AND selected=1" if selected else "")
        for source_id, text, provenance in self.db.execute(
                f"SELECT source_id,text,provenance FROM docs WHERE {where} ORDER BY selection_key", (category,)):
            yield {"id": source_id, "text": text, **json.loads(provenance).get("envelope_copy", {})}


def train_tokenizer(store: CandidateStore, output: Path):
    import sentencepiece as spm
    _, tokenizer_module = legacy()
    cfg = store.cfg
    seed_path = output / "tokenizer_seed.txt"
    caps = targets_from_mix(int(cfg["tokenizer_seed_max_chars"]), cfg["mix"])
    observed = {}
    with seed_path.open("w", encoding="utf-8") as seed_file:
        for category, cap in caps.items():
            count = 0
            for doc in store.documents(category):
                seed_file.write(doc["text"])
                count += len(doc["text"])
                if count >= cap:
                    break
            if count == 0:
                raise RuntimeError(f"tokenizer seed missing {category}")
            observed[category] = count
    prefix = output / "ember_tokenizer"
    spm.SentencePieceTrainer.train(
        input=str(seed_path), model_prefix=str(prefix), model_type="bpe",
        vocab_size=int(cfg["tokenizer"]["vocab_size"]),
        character_coverage=float(cfg["tokenizer"].get("character_coverage", 1.0)),
        byte_fallback=True, hard_vocab_limit=False, bos_id=-1, eos_id=-1, pad_id=-1,
        unk_id=0, unk_piece="<unk>", user_defined_symbols=tokenizer_module.SPECIAL_TOKENS,
        normalization_rule_name="identity", shuffle_input_sentence=False,
        num_threads=4, max_sentence_length=1_000_000, minloglevel=1)
    tokenizer = tokenizer_module.SentencePieceBPETokenizer.from_model_file(prefix.with_suffix(".model"))
    if tokenizer.vocab_size != cfg["tokenizer"]["vocab_size"]:
        raise RuntimeError(f"tokenizer has {tokenizer.vocab_size} pieces, expected {cfg['tokenizer']['vocab_size']}")
    if any(tokenizer._sp.id_to_piece(tokenizer._sp.piece_to_id(token)) != token
           for token in tokenizer_module.SPECIAL_TOKENS):
        raise RuntimeError("special token did not retain its atomic SentencePiece representation")
    return tokenizer, observed


def select_and_split(store: CandidateStore, tokenizer):
    selected_tokens, split_stats = {}, {}
    ratio = float(store.cfg["validation_ratio"])
    if not 0 < ratio < 1:
        raise ValueError("validation_ratio must be between zero and one")
    for category, target in store.targets.items():
        total = 0
        selected_groups = set()
        for row_id, text, group in store.db.execute(
                "SELECT id,text,split_key FROM docs WHERE category=? ORDER BY selection_key", (category,)):
            if total >= target and len(selected_groups) >= 2:
                break
            count = document_tokens(tokenizer, text)
            store.db.execute("UPDATE docs SET tokens=?, selected=1 WHERE id=?", (count, row_id))
            total += count
            if len(selected_groups) < 2:
                selected_groups.add(group)
        if total < target:
            raise RuntimeError(f"{category}: only {total} actual Ember tokens; target={target}")
        selected_tokens[category] = total
        val_target = max(1, int(total * ratio))
        group_count = store.db.execute(
            "SELECT COUNT(DISTINCT split_key) FROM docs WHERE category=? AND selected=1", (category,)).fetchone()[0]
        val_count, last_group, group_index = 0, None, 0
        category_stats = Counter()
        split = "val"
        for row_id, group, count in store.db.execute(
                "SELECT id,split_key,tokens FROM docs WHERE category=? AND selected=1 ORDER BY split_key,id", (category,)):
            if group != last_group:
                group_index += 1
                # Whole groups can exceed a small category's budget. Keep the
                # last group available for training even if validation is short.
                split = "val" if val_count < val_target and group_index < group_count else "train"
                last_group = group
            store.db.execute("UPDATE docs SET split=? WHERE id=?", (split, row_id))
            category_stats[split + "_docs"] += 1
            category_stats[split + "_tokens"] += count
            if split == "val":
                val_count += count
        if not category_stats["train_docs"] or not category_stats["val_docs"]:
            raise RuntimeError(f"{category}: need separate source groups in both train and validation")
        split_stats[category] = dict(category_stats)
    store.db.commit()
    overlap = store.db.execute("""SELECT source,group_id FROM docs WHERE selected=1
        GROUP BY source,group_id HAVING COUNT(DISTINCT split)>1 LIMIT 1""").fetchone()
    if overlap:
        raise RuntimeError(f"source group occurs in both training and validation: {overlap}")
    # Only the first document in each physical file receives an initial prefix.
    for split in ("train", "val"):
        row_id, category, text, count = store.db.execute(
            "SELECT id,category,text,tokens FROM docs WHERE split=? ORDER BY output_key LIMIT 1", (split,)).fetchone()
        delta = document_tokens(tokenizer, text, first=True) - count
        store.db.execute("UPDATE docs SET tokens=tokens+? WHERE id=?", (delta, row_id))
        selected_tokens[category] += delta
        split_stats[category][split + "_tokens"] += delta
    store.db.commit()
    return selected_tokens, split_stats


def save_outputs(store: CandidateStore, output: Path, tokenizer, revisions: dict, *, smoke: bool,
                 selected_tokens: dict, split_stats: dict, seed_chars: dict):
    cfg = store.cfg
    audit = envelope_copy.audit(store.documents("envelope_copy", selected=True), cfg)
    sample_cfg = copy.deepcopy(cfg)
    coverage = sample_cfg["sources"]["envelope_copy"]["coverage_requirements"]
    for key in coverage:
        coverage[key] = False if key == "every_tool_in_every_tool_shape" else 0
    document_audit = envelope_copy.audit(store.documents("envelope_copy", selected=True), sample_cfg)
    if document_audit["status"] != "PASS" or (not smoke and audit["status"] != "PASS"):
        raise RuntimeError(f"selected envelope_copy audit failed: {audit['failures']}")
    _, tok_module = legacy()
    present = {token: False for token in tok_module.SPECIAL_TOKENS}
    token_counts, doc_counts, artifacts = Counter(), Counter(), {}
    provenance_path = output / "provenance.json"
    with ExitStack() as stack:
        streams = {split: stack.enter_context((output / f"{split}.txt").open("w", encoding="utf-8"))
                   for split in ("train", "val")}
        provenance = stack.enter_context(provenance_path.open("w", encoding="utf-8"))
        header = {"ember_version": "0.1.0", "created_utc": datetime.now(timezone.utc).isoformat(),
                  "config": cfg, "dataset_revisions": revisions, "package_sha256": PACKAGE_SHA256}
        provenance.write(json.dumps(header, ensure_ascii=False)[:-1] + ',"documents":[')
        first = True
        for category, source, source_id, group, text, sha, raw_meta, tokens, split in store.db.execute(
                """SELECT category,source,source_id,group_id,text,sha,provenance,tokens,split FROM docs
                   WHERE selected=1 ORDER BY split,output_key"""):
            meta = json.loads(raw_meta)
            validate_document({"category": category, "text": text, "provenance": meta})
            streams[split].write(text)
            token_counts[split] += tokens
            doc_counts[split] += 1
            for token in present:
                present[token] |= token in text
            if not first:
                provenance.write(",")
            first = False
            provenance.write(json.dumps({"category": category, "source": source, "source_id": source_id,
                "group_id": group, "sha256": sha, "token_count": tokens, "split": split, **meta}, ensure_ascii=False))
        provenance.write("]}")
    if not all(present.values()):
        raise RuntimeError(f"missing special tokens: {[t for t, seen in present.items() if not seen]}")
    # Independently re-read the final serialized files with the persisted tokenizer.
    # Documents end at EOT; no candidate-only count is used as final evidence.
    reread = {}
    for split in ("train", "val"):
        count, parts, first_document = 0, [], True
        with (output / f"{split}.txt").open(encoding="utf-8") as stream:
            for line in stream:
                parts.append(line)
                if line == EOT + "\n":
                    count += document_tokens(tokenizer, "".join(parts), first=first_document)
                    first_document = False
                    parts.clear()
        if parts or count != token_counts[split]:
            raise RuntimeError(f"final {split} file does not reproduce selected token counts")
        reread[split] = count
    total = sum(reread.values())
    if not smoke and not cfg["min_total_tokens"] <= total <= cfg["max_total_tokens"]:
        raise RuntimeError(f"production token gate failed: {total}")
    for name in ("train.txt", "val.txt", "ember_tokenizer.model", "provenance.json"):
        path = output / name
        artifacts[name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    stats = {
        "status": "SMOKE_PASS" if smoke else "PASS", "production_ready": not smoke,
        "ember_version": "0.1.0", "scope": "small CPU pipeline smoke" if smoke else "full corpus",
        "actual_ember_tokens": total, "train_tokens": reread["train"], "val_tokens": reread["val"],
        "vocab_size": tokenizer.vocab_size, "tokenizer_model": "ember_tokenizer.model",
        "token_count_method": "saved SentencePiece model; EOT context carried across documents; final files re-read",
        "target_total_tokens": sum(store.targets.values()), "production_token_gate": [cfg["min_total_tokens"], cfg["max_total_tokens"]],
        "category_target_tokens": store.targets, "category_selected_tokens": selected_tokens,
        "split_by_category": split_stats, "source_groups_disjoint": True,
        "documents": dict(doc_counts), "candidate_documents": dict(store.counts),
        "candidate_characters": dict(store.chars), "rejected_invalid_or_benchmark": dict(store.rejected),
        "special_tokens_present": present, "tokenizer_seed_characters": seed_chars,
        "dedup": {**store.near.stats(), "global_exact_duplicates_rejected": store.exact_rejected,
                  "envelope_copy": "case-sensitive exact-only; no MinHash"},
        "finite_source_headroom_shortfalls": store.finite_shortfalls,
        "envelope_copy_audit": audit, "envelope_copy_document_audit": document_audit,
        "held_out_leakage": document_audit["held_out_leakage"],
        "held_out_leakage_scope": "synthetic target values; exact frozen semantic prompts filtered from external sources",
        "dataset_revisions": revisions, "artifacts": artifacts,
        "package_sha256": PACKAGE_SHA256, "gpu_training_launched": False,
    }
    (output / "corpus_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def build(cfg: dict, output: Path, *, smoke: bool = False, target_total: int | None = None,
          document_source=None, revisions=None):
    target_total = int(target_total or cfg["target_total_tokens"])
    if target_total != cfg["target_total_tokens"] and not smoke:
        raise ValueError("a different token budget requires --smoke; it cannot produce a production PASS")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty; prior corpus evidence is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(cfg)
    if smoke:
        cfg["tokenizer_seed_max_chars"] = min(cfg["tokenizer_seed_max_chars"], max(100_000, target_total * 4))
    store = CandidateStore(output / "candidates.sqlite3", cfg, target_total)
    try:
        if document_source is None:
            from ember_corpus_sources_v010 import collect_sources, resolve_revisions
            revisions = revisions or resolve_revisions(cfg)
            collect_sources(cfg, revisions, store)
        else:
            for doc in document_source:
                store.add(doc)
        for category in cfg["mix"]:
            store.finish_category(category)
        store.db.commit()
        print(json.dumps({"event": "candidates_ready", "counts": store.counts, "chars": store.chars}), flush=True)
        tokenizer, seed_chars = train_tokenizer(store, output)
        selected, splits = select_and_split(store, tokenizer)
        stats = save_outputs(store, output, tokenizer, revisions or {}, smoke=smoke,
                             selected_tokens=selected, split_stats=splits, seed_chars=seed_chars)
        print(json.dumps({"event": "corpus_complete", "status": stats["status"],
                          "actual_ember_tokens": stats["actual_ember_tokens"]}), flush=True)
        return stats
    except Exception as exc:
        (output / "corpus_stats.json").write_text(json.dumps({
            "status": "FAIL", "production_ready": False, "error": str(exc),
            "candidate_characters": dict(store.chars), "candidate_documents": dict(store.counts),
            "dataset_revisions": revisions or {}, "gpu_training_launched": False}, indent=2), encoding="utf-8")
        raise
    finally:
        store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/corpus_v0.1.0.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="small pipeline check, never a production PASS")
    parser.add_argument("--target-total-tokens", type=int)
    args = parser.parse_args(argv)
    build(json.loads(args.config.read_text()), args.output_dir, smoke=args.smoke,
          target_total=args.target_total_tokens)


if __name__ == "__main__":
    main()
