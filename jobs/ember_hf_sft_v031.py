# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["huggingface-hub>=1.4", "sentencepiece>=0.2", "torch>=2.4"]
# ///
"""Ember v0.0.31: guarded continuation from the v0.0.30 best checkpoint.

This phase intentionally keeps the v0.0.30 k=2 boundary-focused objective,
mining policy, curriculum, learning rate, schedule, and reporter contract intact.
Only the source/output model identity, seed, protection floors, and versioned
markers advance. The goal is to test whether a fresh schedule from the stronger
v0.0.30 step-499 checkpoint can close the remaining 2 exact cases and 6
continuation-token decisions without sacrificing the already-passed legacy gates.
"""
from __future__ import annotations

import urllib.request

V030_COMMIT = "09d6b15a04c7511993e963feea1d9456c60ec2df"
V031_CONFIG_COMMIT = "a5ef92fe52ea2c91008541fdb76f657a3df42a47"
V030_URL = (
    "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/"
    f"{V030_COMMIT}/jobs/ember_hf_sft_v030.py"
)


def build_v031_source(source: str) -> str:
    required = [
        'CONFIG_COMMIT = "4c3b3764abd46a876bdd59c4a4ac5c8086dc5e6f"',
        'SOURCE_REPO = "Jmiller18899/ember-v0.0.29-t4"',
        'resolve_v029_source',
        'v029_best_step',
        'v030_progress',
        'EMBER_HF_V030_',
        'EMBER_V030_',
        '0.0.30',
    ]
    missing = [token for token in required if token not in source]
    if missing:
        raise RuntimeError(f"v0.0.31 source contract changed; missing: {missing}")

    # First advance the current phase's own identifiers.
    source = source.replace("v030", "v031").replace("V030", "V031")
    source = source.replace("0.0.30", "0.0.31")

    # Then advance the source-model target from v0.0.29 -> v0.0.30.  These
    # tokens intentionally remain one version behind the phase itself.  Include
    # both the prose form (v0.0.29) and the bare version string (0.0.29), because
    # the checkpoint resolver compares train_config['version'] without a leading v.
    source = source.replace("ember-v0.0.29-t4", "ember-v0.0.30-t4")
    source = source.replace("v0.0.29", "v0.0.30")
    source = source.replace("0.0.29", "0.0.30")
    source = source.replace("v029", "v030").replace("V029", "V030")

    # Pin the immutable commit that contains the v0.0.31 config.
    source = source.replace(
        'CONFIG_COMMIT = "4c3b3764abd46a876bdd59c4a4ac5c8086dc5e6f"',
        f'CONFIG_COMMIT = "{V031_CONFIG_COMMIT}"',
    )

    # Hard assertions prevent a silent fallback to the wrong source/config.
    for forbidden in (
        'ember_multi_position_v0.0.30.json',
        'SOURCE_REPO = "Jmiller18899/ember-v0.0.29-t4"',
        'resolve_v029_source',
        'EMBER_HF_V030_',
        'EMBER_V030_',
        'source_cfg.get("version")) != "0.0.29"',
    ):
        if forbidden in source:
            raise RuntimeError(f"v0.0.31 transform left stale token: {forbidden}")
    for expected in (
        'ember_multi_position_v0.0.31.json',
        'SOURCE_REPO = "Jmiller18899/ember-v0.0.30-t4"',
        'resolve_v030_source',
        'source_cfg.get("version")) != "0.0.30"',
        'EMBER_HF_V031_',
        'EMBER_V031_',
        V031_CONFIG_COMMIT,
    ):
        if expected not in source:
            raise RuntimeError(f"v0.0.31 transform missing expected token: {expected}")
    return source


def main() -> None:
    with urllib.request.urlopen(V030_URL) as response:
        source = response.read().decode("utf-8")
    source = build_v031_source(source)
    print("EMBER_V031_SOURCE_ADVANCE_TRANSFORM=PASS", flush=True)
    exec(compile(source, "ember_hf_sft_v031_wrapper_runtime.py", "exec"), {"__name__": "__main__"})


if __name__ == "__main__":
    main()
