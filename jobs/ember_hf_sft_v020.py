# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["huggingface-hub>=1.4", "sentencepiece>=0.2", "torch>=2.4"]
# ///
"""Ember v0.0.20: concentrate hard-mined margin pressure on continuation tokens.

v0.0.19 made first-token selection effectively perfect but left continuation
TARGET top-1 unchanged. This phase resumes from v0.0.19 best.pt, excludes the
first TARGET token from margin/hardness scoring, lowers its CE weight, and
mines sequences with continuation failures more aggressively. Held-out
promotion diagnostics remain unchanged.
"""
from __future__ import annotations

import urllib.request

BASE_COMMIT = "0affb9360f8dfb58591b17f2ae077ec6ebd3e8bc"
CONFIG_COMMIT = "6da99687cbca0746b5352f2fd0ef12c49364dfff"
BASE_URL = (
    "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/"
    f"{BASE_COMMIT}/jobs/ember_hf_sft_v019.py"
)

with urllib.request.urlopen(BASE_URL) as response:
    source = response.read().decode("utf-8")


def replace_required(old: str, new: str, *, count: int = -1) -> None:
    global source
    if old not in source:
        raise RuntimeError(f"v0.0.20 transform target missing: {old!r}")
    source = source.replace(old, new, count)


# Repoint v0.0.19's tested hard-mining scaffold at v0.0.20 and v0.0.19 best.pt.
replace_required(
    'CONFIG_COMMIT = "c29d2a2f15e3bbbf478e0643dcad9151c3ede78f"',
    f'CONFIG_COMMIT = "{CONFIG_COMMIT}"',
    count=1,
)
source = source.replace("0.0.19", "0.0.20")
replace_required("ember_margin_copy_v0.0.20.json", "ember_continuation_margin_v0.0.20.json")
replace_required("hard-example-margin-copy", "continuation-margin-hard-mining")
replace_required(
    'SOURCE_REPO = "Jmiller18899/ember-v0.0.18-t4"',
    'SOURCE_REPO = "Jmiller18899/ember-v0.0.19-t4"',
)
replace_required(
    'cfg.get("source_model_name") != "ember-v0.0.18-t4"',
    'cfg.get("source_model_name") != "ember-v0.0.19-t4"',
)
replace_required("must start from v0.0.18", "must start from v0.0.19")
replace_required(
    'if str(source_cfg.get("version")) != "0.0.18":',
    'if str(source_cfg.get("version")) != "0.0.19":',
)
replace_required("expected a v0.0.18 checkpoint", "expected a v0.0.19 checkpoint")
replace_required("v0.0.18 checkpoint run_id", "v0.0.19 checkpoint run_id")
replace_required("v0.0.18 source state", "v0.0.19 source state")
replace_required("resolve_v018_source", "resolve_v019_source")
replace_required("v018_promotion", "v019_promotion")
replace_required("v018_best_step", "v019_best_step")
source = source.replace("EMBER_HF_V019_", "EMBER_HF_V020_")
source = source.replace("EMBER_V019_", "EMBER_V020_")
source = source.replace("ember_hf_sft_v019_runtime.py", "ember_hf_sft_v020_runtime.py")

# The v0.0.19 objective scored first TARGET and continuation TARGET positions.
# v0.0.20 deliberately excludes the already-solved first TARGET token from
# both hard-example selection and the argmax-margin loss.
old_hard_mask = '''target_mask = y.ne(-100) & (\n        weights.ge(copy_weight - 1e-6) | (weights - first_weight).abs().lt(1e-6)\n    )'''
new_hard_mask = '''target_mask = y.ne(-100) & weights.ge(copy_weight - 1e-6)'''
replace_required(old_hard_mask, new_hard_mask, count=1)

old_loss_mask = '''target_mask = active & (\n        weights.ge(copy_weight - 1e-6) | (weights - first_weight).abs().lt(1e-6)\n    )'''
new_loss_mask = '''target_mask = active & weights.ge(copy_weight - 1e-6)'''
replace_required(old_loss_mask, new_loss_mask, count=1)

# Prioritize rows where any continuation token is currently wrong, while still
# ranking by continuation error rate and logit-margin deficit.
replace_required(
    'hardness = error_rate * 2.0 + deficit_mean',
    'row_failed = wrong.any(dim=1).float()\n'
    '        sequence_boost = 1.0 + row_failed * (float(cfg["sequence_failure_multiplier"]) - 1.0)\n'
    '        hardness = (error_rate * 2.0 + deficit_mean) * sequence_boost',
    count=1,
)

# Extend CPU preflight so it proves that first-token and continuation masks are
# distinct and that the continuation-only mask is non-empty before GPU spend.
replace_required(
    'print("EMBER_V020_HARD_MINING_PREFLIGHT=PASS", flush=True)\n'
    '            print("EMBER_HF_V020_PREFLIGHT=PASS", flush=True)',
    'probe_item = train[probe[0]]\n'
    '            probe_y = probe_item["y"]\n'
    '            probe_w = probe_item["w"]\n'
    '            probe_active = probe_y.ne(-100)\n'
    '            probe_first = probe_active & (probe_w - float(cfg["first_token_weight"])).abs().lt(1e-6)\n'
    '            probe_cont = probe_active & probe_w.ge(float(cfg["copy_token_weight"]) - 1e-6)\n'
    '            if not bool(probe_first.any().item()) or not bool(probe_cont.any().item()):\n'
    '                raise RuntimeError("v0.0.20 preflight could not identify both first-token and continuation positions")\n'
    '            if bool((probe_first & probe_cont).any().item()):\n'
    '                raise RuntimeError("v0.0.20 first-token and continuation masks overlap")\n'
    '            print("EMBER_V020_CONTINUATION_ONLY_PREFLIGHT=PASS", flush=True)\n'
    '            print("EMBER_V020_HARD_MINING_PREFLIGHT=PASS", flush=True)\n'
    '            print("EMBER_HF_V020_PREFLIGHT=PASS", flush=True)',
    count=1,
)

exec(compile(source, "ember_hf_sft_v020_wrapper_runtime.py", "exec"), {"__name__": "__main__"})
