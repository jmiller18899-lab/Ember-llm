# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "huggingface-hub>=1.4",
#   "sentencepiece>=0.2",
#   "torch>=2.4",
# ]
# ///
"""Final CPU-gated wrapper for the v0.0.12 copy-generalization canary."""
from __future__ import annotations
import urllib.request

WRAPPER_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/286464152b8a2ae360b83f050f6364345c8923d6/jobs/ember_hf_sft_v012.py"
SAFE_DATA_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/6c5409a5777b83f1f1d44dcd5f0e76ce129c25cd/jobs/ember_sft_data_v012_safe.py"
source = urllib.request.urlopen(WRAPPER_URL, timeout=30).read().decode("utf-8")

old_data_guard = "old_data = 'DATA_URL = f\"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{ASSET_PIN}/jobs/ember_sft_data_v012.py\"'"
new_data_guard = "old_data = 'DATA_URL = f\"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{ASSET_PIN}/jobs/ember_sft_data_v011.py\"'"
if old_data_guard not in source:
    raise RuntimeError("v0.0.12 data guard changed unexpectedly")
source = source.replace(old_data_guard, new_data_guard, 1)

old_data_url = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/3d8aba948670b8fb8e064609bcf72704568fa682/jobs/ember_sft_data_v012.py"
if old_data_url not in source:
    raise RuntimeError("v0.0.12 pinned DATA_URL changed unexpectedly")
source = source.replace(old_data_url, SAFE_DATA_URL, 1)

exec(compile(source, WRAPPER_URL, "exec"), {"__name__": "__main__", "__file__": WRAPPER_URL})
