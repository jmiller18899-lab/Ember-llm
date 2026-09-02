# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "huggingface-hub>=1.4",
#   "sentencepiece>=0.2",
#   "torch>=2.4",
# ]
# ///
"""Run v0.0.11 with exact-value data and character-aligned semantic weights."""
from __future__ import annotations

import urllib.request

TRAINER_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/df8ba9760624fe8d9c2fa4b66a8af63d1fa82c35/jobs/ember_hf_sft_v011.py"
DATA_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/34b8851b2e8d48e72ed0f512a37cdfa5b90081fd/jobs/ember_sft_data_v011_exact.py"
source = urllib.request.urlopen(TRAINER_URL, timeout=30).read().decode("utf-8")

old_data = 'DATA_URL = f"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{ASSET_PIN}/jobs/ember_sft_data_v011.py"'
new_data = f'DATA_URL = "{DATA_URL}"'
if old_data not in source:
    raise RuntimeError("pinned trainer DATA_URL line changed unexpectedly")
source = source.replace(old_data, new_data, 1)

old_span = '''    semantic_positions = set()\n    missing = []\n    for term in row["focus_terms"]:\n        found = False\n        for needle in term_sequences(tokenizer, term):\n            for a, b in subseq_hits(full_ids, needle, len(prompt_ids)):\n                found = True\n                for token_pos in range(a, b):\n                    target_pos = token_pos - 1\n                    if first_target <= target_pos < len(seq_y):\n                        semantic_positions.add(target_pos)\n        if not found:\n            missing.append(str(term))\n    if missing:\n        raise RuntimeError(f"semantic token span not found: {row['id']} {missing}")\n'''
new_span = '''    semantic_positions = set()\n    missing = []\n    completion_text = row["completion"]\n    for term in row["focus_terms"]:\n        term_text = str(term)\n        char_start = completion_text.find(term_text)\n        if char_start < 0:\n            missing.append(term_text)\n            continue\n        char_end = char_start + len(term_text)\n        before_ids = list(tokenizer.encode(row["prompt"] + completion_text[:char_start]))\n        through_ids = list(tokenizer.encode(row["prompt"] + completion_text[:char_end]))\n        token_start = max(len(prompt_ids), len(before_ids) - 1)\n        token_end = min(len(full_ids), max(token_start + 1, len(through_ids) + 1))\n        for token_pos in range(token_start, token_end):\n            if token_pos < len(full_ids):\n                target_pos = token_pos - 1\n                if first_target <= target_pos < len(seq_y):\n                    semantic_positions.add(target_pos)\n    if missing:\n        raise RuntimeError(f"semantic text absent from target: {row['id']} {missing}")\n'''
if old_span not in source:
    raise RuntimeError("pinned trainer semantic span block changed unexpectedly")
source = source.replace(old_span, new_span, 1)

exec(compile(source, TRAINER_URL, "exec"), {"__name__": "__main__", "__file__": TRAINER_URL})
