# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "huggingface-hub>=1.4",
#   "sentencepiece>=0.2",
#   "torch>=2.4",
# ]
# ///
"""Run the pinned v0.0.11 trainer with the exact-value curriculum adapter."""
from __future__ import annotations

import urllib.request

TRAINER_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/df8ba9760624fe8d9c2fa4b66a8af63d1fa82c35/jobs/ember_hf_sft_v011.py"
DATA_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/34b8851b2e8d48e72ed0f512a37cdfa5b90081fd/jobs/ember_sft_data_v011_exact.py"
source = urllib.request.urlopen(TRAINER_URL, timeout=30).read().decode("utf-8")
old = 'DATA_URL = f"https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/{ASSET_PIN}/jobs/ember_sft_data_v011.py"'
new = f'DATA_URL = "{DATA_URL}"'
if old not in source:
    raise RuntimeError("pinned trainer DATA_URL line changed unexpectedly")
source = source.replace(old, new, 1)
exec(compile(source, TRAINER_URL, "exec"), {"__name__": "__main__", "__file__": TRAINER_URL})
