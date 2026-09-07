# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["huggingface-hub>=1.4", "sentencepiece>=0.2", "torch>=2.4"]
# ///
"""Continue Ember sequence-copy training from the v0.0.16 best checkpoint.

This launcher deliberately reuses the tested v0.0.16 training implementation,
then applies the small version/source substitutions needed for v0.0.17. The
v0.0.17 config lowers learning rate, lengthens consolidation, and gives more
weight to continuation tokens while retaining the existing regression gates.
"""
from __future__ import annotations

import urllib.request

SOURCE_COMMIT = "8ccd8cf3cd6de9c1b92e9f9a8897772f67d14c15"
SOURCE_URL = (
    "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/"
    f"{SOURCE_COMMIT}/jobs/ember_hf_sft_v016.py"
)

with urllib.request.urlopen(SOURCE_URL) as response:
    source = response.read().decode("utf-8")

# Point the tested trainer at the new version/config and the actual v0.0.16 best
# checkpoint. Keep the v0.0.15 helper module because it supplies the shared data,
# loss, diagnostic, checkpoint, and upload utilities used by both phases.
source = source.replace(
    'CONFIG_PIN = "42ed91ac6e6b8ff22873fec6706c4f243f57f2fa"',
    f'CONFIG_PIN = "{SOURCE_COMMIT}"',
)
source = source.replace("0.0.16", "0.0.17")
source = source.replace(
    'SOURCE_REPO = "Jmiller18899/ember-v0.0.15-t4"',
    'SOURCE_REPO = "Jmiller18899/ember-v0.0.16-t4"',
)
source = source.replace(
    'cfg.get("source_model_name") != "ember-v0.0.15-t4"',
    'cfg.get("source_model_name") != "ember-v0.0.16-t4"',
)
source = source.replace("must start from v0.0.15", "must start from v0.0.16")
source = source.replace(
    'if str(source_cfg.get("version")) != "0.0.15":',
    'if str(source_cfg.get("version")) != "0.0.16":',
)
source = source.replace("expected a v0.0.15 checkpoint", "expected a v0.0.16 checkpoint")
source = source.replace("v0.0.15 checkpoint run_id", "v0.0.16 checkpoint run_id")
source = source.replace("v0.0.15 source state", "v0.0.16 source state")
source = source.replace("resolve_v015_source", "resolve_v016_source")
source = source.replace("v015_promotion", "v016_promotion")
source = source.replace("v015_best_step", "v016_best_step")

exec(compile(source, "ember_hf_sft_v017_runtime.py", "exec"), {"__name__": "__main__"})
