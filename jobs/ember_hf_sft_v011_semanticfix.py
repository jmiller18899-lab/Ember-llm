# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "huggingface-hub>=1.4",
#   "sentencepiece>=0.2",
#   "torch>=2.4",
# ]
# ///
"""Ember v0.0.11 corrective SFT launcher with robust semantic-span mapping.

The base trainer fails closed when a semantic focus term cannot be matched to
its target token IDs. SentencePiece can tokenize the same text differently at
completion, JSON, and punctuation boundaries, so standalone token needles are
not reliable. This launcher first locates each focus term in the decoded actual
completion, then maps that character span back onto the already-tokenized
completion. No missing semantic label is silently ignored.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import tempfile
import urllib.request

TRAINER_PIN = "df8ba9760624fe8d9c2fa4b66a8af63d1fa82c35"
TRAINER_URL = (
    "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/"
    f"{TRAINER_PIN}/jobs/ember_hf_sft_v011.py"
)


def load_trainer(path: Path):
    spec = importlib.util.spec_from_file_location("ember_hf_sft_v011_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned v0.0.11 trainer")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def align_focus_terms(rows):
    """Align labels only when the same text exists with different casing."""
    for row in rows:
        completion = str(row["completion"])
        aligned = []
        for term in row["focus_terms"]:
            raw = str(term)
            match = re.search(re.escape(raw), completion, flags=re.IGNORECASE)
            aligned.append(completion[match.start():match.end()] if match else raw)
        row["focus_terms"] = aligned
    return rows


def _decoded_prefix_lengths(tokenizer, ids):
    lengths = [0]
    for i in range(1, len(ids) + 1):
        lengths.append(len(tokenizer.decode(ids[:i])))
    return lengths


def fixed_encode_row(tokenizer, row, block_size, semantic_weight, eos_weight, torch):
    prompt_ids = list(tokenizer.encode(row["prompt"]))
    full_ids = list(tokenizer.encode(row["prompt"] + row["completion"]))
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError(f"tokenization changed at completion boundary: {row['id']}")
    if len(full_ids) < 2 or len(full_ids) > block_size + 1:
        raise RuntimeError(f"sequence length out of range: {row['id']}={len(full_ids)}")

    eot = list(tokenizer.encode("<|endoftext|>"))
    eot_id = int(eot[-1])
    x = torch.full((block_size,), eot_id, dtype=torch.long)
    y = torch.full((block_size,), -100, dtype=torch.long)
    w = torch.zeros((block_size,), dtype=torch.float32)
    seq_x = torch.tensor(full_ids[:-1], dtype=torch.long)
    seq_y = torch.tensor(full_ids[1:], dtype=torch.long)
    x[:len(seq_x)] = seq_x
    first_target = max(0, len(prompt_ids) - 1)
    y[first_target:len(seq_y)] = seq_y[first_target:]
    w[first_target:len(seq_y)] = 1.0

    completion_ids = full_ids[len(prompt_ids):]
    decoded_completion = tokenizer.decode(completion_ids)
    prefix_lengths = _decoded_prefix_lengths(tokenizer, completion_ids)
    semantic_positions = set()
    missing = []

    for term in row["focus_terms"]:
        raw = str(term)
        matches = list(re.finditer(re.escape(raw), decoded_completion))
        if not matches:
            matches = list(re.finditer(re.escape(raw), decoded_completion, flags=re.IGNORECASE))
        if not matches:
            missing.append(raw)
            continue

        for match in matches:
            char_start, char_end = match.span()
            token_start = None
            token_end = None
            for i in range(len(completion_ids)):
                left = prefix_lengths[i]
                right = prefix_lengths[i + 1]
                if token_start is None and right > char_start:
                    token_start = i
                if token_start is not None and right >= char_end:
                    token_end = i + 1
                    break
            if token_start is None:
                missing.append(raw)
                continue
            if token_end is None:
                token_end = len(completion_ids)
            for comp_pos in range(token_start, token_end):
                token_pos = len(prompt_ids) + comp_pos
                target_pos = token_pos - 1
                if first_target <= target_pos < len(seq_y):
                    semantic_positions.add(target_pos)

    if missing:
        raise RuntimeError(f"semantic text span not found: {row['id']} {sorted(set(missing))}")
    if not semantic_positions:
        raise RuntimeError(f"semantic token span empty after text mapping: {row['id']}")
    for pos in semantic_positions:
        w[pos] = max(float(w[pos]), float(semantic_weight))

    eos_positions = 0
    for token_pos in range(len(prompt_ids), len(full_ids)):
        if int(full_ids[token_pos]) == eot_id:
            target_pos = token_pos - 1
            if first_target <= target_pos < len(seq_y):
                w[target_pos] = max(float(w[target_pos]), float(eos_weight))
                eos_positions += 1
    if eos_positions < 1:
        raise RuntimeError(f"no trainable EOS target: {row['id']}")

    return {
        "x": x, "y": y, "w": w, "row": row,
        "semantic_positions": len(semantic_positions),
    }


def main():
    with tempfile.TemporaryDirectory(prefix="ember-v011-semanticfix-") as td:
        trainer_path = Path(td) / "ember_hf_sft_v011.py"
        urllib.request.urlretrieve(TRAINER_URL, trainer_path)
        trainer = load_trainer(trainer_path)

        original_load_module = trainer.load_module
        def fixed_load_module(path):
            data = original_load_module(path)
            original_build_examples = data.build_examples
            def build_examples(split, total):
                return align_focus_terms(original_build_examples(split, total))
            data.build_examples = build_examples
            return data

        trainer.load_module = fixed_load_module
        trainer.encode_row = fixed_encode_row
        trainer.main()


if __name__ == "__main__":
    main()
