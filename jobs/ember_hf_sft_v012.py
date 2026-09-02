# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "huggingface-hub>=1.4",
#   "sentencepiece>=0.2",
#   "torch>=2.4",
# ]
# ///
"""Run Ember v0.0.12 copy-generalization canary from the saved v0.0.11 best checkpoint."""
from __future__ import annotations

import urllib.request

TRAINER_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/df8ba9760624fe8d9c2fa4b66a8af63d1fa82c35/jobs/ember_hf_sft_v011.py"
CONFIG_URL_FIXED = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/95ce9463eabb974f95c2dd4dd517a717c1a72b1e/config/ember_agent_copy_canary_v0.0.12.json"
DATA_URL_FIXED = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/3d8aba948670b8fb8e064609bcf72704568fa682/jobs/ember_sft_data_v012.py"
SOURCE_REPO_FIXED = "Jmiller18899/ember-v0.0.11-t4"
SOURCE_CHECKPOINT_FIXED = "checkpoints/ember-agent-v0.0.11-semantic-copy-sft-20260902T214935Z/best.pt"
SOURCE_SHA256_FIXED = "04a4868764ad1681d466c2fa3226418926fba44c777e1bca8e8133e832c0082e"

source = urllib.request.urlopen(TRAINER_URL, timeout=30).read().decode("utf-8")

# Advance the trainer's lifecycle/version markers without changing the model package.
source = source.replace("v0.0.11", "v0.0.12")
source = source.replace("V011", "V012")
source = source.replace("ember-v0.0.9-t4", "ember-v0.0.11-t4")
source = source.replace(
    "checkpoints/ember-agent-v0.0.9-tool-sft-20260828T142047Z/best.pt",
    SOURCE_CHECKPOINT_FIXED,
)
source = source.replace(
    "8299d52e8a852b9bd3e8403e086b48fd42d2babfd51c74eb94af29bd87ef2d13",
    SOURCE_SHA256_FIXED,
)

old_config = 'CONFIG_URL = f"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{ASSET_PIN}/config/ember_agent_semantic_sft_v0.0.12.json"'
old_data = 'DATA_URL = f"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{ASSET_PIN}/jobs/ember_sft_data_v012.py"'
if old_config not in source or old_data not in source:
    raise RuntimeError("pinned base trainer asset lines changed unexpectedly")
source = source.replace(old_config, f'CONFIG_URL = "{CONFIG_URL_FIXED}"', 1)
source = source.replace(old_data, f'DATA_URL = "{DATA_URL_FIXED}"', 1)
source = source.replace(
    'SOURCE_REPO = "Jmiller18899/ember-v0.0.11-t4"',
    f'SOURCE_REPO = "{SOURCE_REPO_FIXED}"',
    1,
)
source = source.replace(
    'SOURCE_CHECKPOINT = "' + SOURCE_CHECKPOINT_FIXED + '"',
    f'SOURCE_CHECKPOINT = "{SOURCE_CHECKPOINT_FIXED}"',
    1,
)
source = source.replace(
    'SOURCE_SHA256 = "' + SOURCE_SHA256_FIXED + '"',
    f'SOURCE_SHA256 = "{SOURCE_SHA256_FIXED}"',
    1,
)
source = source.replace(
    'frozen_config = work/"ember_agent_semantic_sft_v0.0.12.json"',
    'frozen_config = work/"ember_agent_copy_canary_v0.0.12.json"',
)
source = source.replace(
    '"config/ember_agent_semantic_sft_v0.0.12.json"',
    '"config/ember_agent_copy_canary_v0.0.12.json"',
)

# SentencePiece pieces can change at completion boundaries. Align semantic weights
# from exact target character spans back to token positions instead of matching
# independently-tokenized terms.
old_span = '''    semantic_positions = set()\n    missing = []\n    for term in row["focus_terms"]:\n        found = False\n        for needle in term_sequences(tokenizer, term):\n            for a, b in subseq_hits(full_ids, needle, len(prompt_ids)):\n                found = True\n                for token_pos in range(a, b):\n                    target_pos = token_pos - 1\n                    if first_target <= target_pos < len(seq_y):\n                        semantic_positions.add(target_pos)\n        if not found:\n            missing.append(str(term))\n    if missing:\n        raise RuntimeError(f"semantic token span not found: {row['id']} {missing}")\n'''
new_span = '''    semantic_positions = set()\n    missing = []\n    completion_text = row["completion"]\n    for term in row["focus_terms"]:\n        term_text = str(term)\n        char_start = completion_text.find(term_text)\n        if char_start < 0:\n            missing.append(term_text)\n            continue\n        char_end = char_start + len(term_text)\n        before_ids = list(tokenizer.encode(row["prompt"] + completion_text[:char_start]))\n        through_ids = list(tokenizer.encode(row["prompt"] + completion_text[:char_end]))\n        token_start = max(len(prompt_ids), len(before_ids) - 1)\n        token_end = min(len(full_ids), max(token_start + 1, len(through_ids) + 1))\n        for token_pos in range(token_start, token_end):\n            if token_pos < len(full_ids):\n                target_pos = token_pos - 1\n                if first_target <= target_pos < len(seq_y):\n                    semantic_positions.add(target_pos)\n    if missing:\n        raise RuntimeError(f"semantic text absent from target: {row['id']} {missing}")\n'''
if old_span not in source:
    raise RuntimeError("pinned base trainer semantic span block changed unexpectedly")
source = source.replace(old_span, new_span, 1)

exec(compile(source, TRAINER_URL, "exec"), {"__name__": "__main__", "__file__": TRAINER_URL})
