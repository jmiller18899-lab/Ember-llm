# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "huggingface-hub>=1.4",
#   "sentencepiece>=0.2",
#   "torch>=2.4",
# ]
# ///
"""Ember v0.0.11 corrective SFT launcher with robust semantic-span matching.

The original weighted trainer correctly refuses to train if a declared focus
term cannot be found in the completion. Some direct-title examples preserve the
semantic term while changing presentation case (for example API -> Api), and
SentencePiece tokenization can differ at a completion/newline boundary. This
launcher aligns case to the target and derives token needles in realistic local
contexts while retaining the original fail-closed semantic validation.
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


def _lcp_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def fixed_term_sequences(tokenizer, term):
    raw = str(term)
    seqs = []

    def add(ids):
        ids = list(ids)
        if ids and ids not in seqs:
            seqs.append(ids)
        if len(ids) > 1 and ids[1:] not in seqs:
            seqs.append(ids[1:])

    # Standalone forms retained for normal in-text matches.
    add(tokenizer.encode(raw))
    add(tokenizer.encode(" " + raw))

    # Derive suffix tokens using realistic preceding contexts. This handles
    # SentencePiece boundary changes at assistant newlines and JSON values.
    for prefix in ("\n", " ", '"', ':', '":', '":"'):
        before = list(tokenizer.encode(prefix))
        combined = list(tokenizer.encode(prefix + raw))
        cut = _lcp_len(before, combined)
        add(combined[cut:])

    return sorted(seqs, key=len, reverse=True)


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
        trainer.term_sequences = fixed_term_sequences
        trainer.main()


if __name__ == "__main__":
    main()
