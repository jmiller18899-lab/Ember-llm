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
semantic term while changing presentation case (for example API -> Api). This
launcher keeps the fail-closed behavior while allowing only deterministic case
forms of the declared term to map to the actual target tokens.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
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


def fixed_term_sequences(tokenizer, term):
    raw = str(term)
    forms = []
    for value in (raw, raw.title(), raw.capitalize(), raw.lower(), raw.upper()):
        if value not in forms:
            forms.append(value)

    seqs = []
    for form in forms:
        for value in (form, " " + form):
            ids = list(tokenizer.encode(value))
            if ids and ids not in seqs:
                seqs.append(ids)
            # SentencePiece-style tokenizers may add a leading boundary token.
            if len(ids) > 1 and ids[1:] not in seqs:
                seqs.append(ids[1:])
    return sorted(seqs, key=len, reverse=True)


def main():
    with tempfile.TemporaryDirectory(prefix="ember-v011-semanticfix-") as td:
        trainer_path = Path(td) / "ember_hf_sft_v011.py"
        urllib.request.urlretrieve(TRAINER_URL, trainer_path)
        trainer = load_trainer(trainer_path)
        trainer.term_sequences = fixed_term_sequences
        trainer.main()


if __name__ == "__main__":
    main()
