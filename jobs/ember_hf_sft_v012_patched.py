# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "huggingface-hub>=1.4",
#   "sentencepiece>=0.2",
#   "torch>=2.4",
# ]
# ///
"""Apply the exact v011 filename correction to the pinned v0.0.12 trainer wrapper."""
from __future__ import annotations
import urllib.request

WRAPPER_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/286464152b8a2ae360b83f050f6364345c8923d6/jobs/ember_hf_sft_v012.py"
source = urllib.request.urlopen(WRAPPER_URL, timeout=30).read().decode("utf-8")
old = "old_data = 'DATA_URL = f\"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{ASSET_PIN}/jobs/ember_sft_data_v012.py\"'"
new = "old_data = 'DATA_URL = f\"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{ASSET_PIN}/jobs/ember_sft_data_v011.py\"'"
if old not in source:
    raise RuntimeError("v0.0.12 wrapper pin-check line changed unexpectedly")
source = source.replace(old, new, 1)
exec(compile(source, WRAPPER_URL, "exec"), {"__name__": "__main__", "__file__": WRAPPER_URL})
