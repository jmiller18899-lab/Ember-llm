# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "huggingface-hub>=1.4",
#   "sentencepiece>=0.2",
#   "torch>=2.4",
# ]
# ///
"""Final pinned wrapper for Ember v0.0.12 copy-generalization canary."""
from __future__ import annotations
import urllib.request

WRAPPER_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/286464152b8a2ae360b83f050f6364345c8923d6/jobs/ember_hf_sft_v012.py"
FINAL_DATA_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/c263a5747833038d0876e0530aecddf523427e40/jobs/ember_sft_data_v012_final.py"
source = urllib.request.urlopen(WRAPPER_URL, timeout=30).read().decode("utf-8")

old_guard = "old_data = 'DATA_URL = f\"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{ASSET_PIN}/jobs/ember_sft_data_v012.py\"'"
new_guard = "old_data = 'DATA_URL = f\"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{ASSET_PIN}/jobs/ember_sft_data_v011.py\"'"
if old_guard not in source:
    raise RuntimeError("v0.0.12 data guard changed unexpectedly")
source = source.replace(old_guard, new_guard, 1)

old_url = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/3d8aba948670b8fb8e064609bcf72704568fa682/jobs/ember_sft_data_v012.py"
if old_url not in source:
    raise RuntimeError("v0.0.12 pinned data URL changed unexpectedly")
source = source.replace(old_url, FINAL_DATA_URL, 1)

exec(compile(source, WRAPPER_URL, "exec"), {"__name__": "__main__", "__file__": WRAPPER_URL})
