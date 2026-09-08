"""CPU-only known-tool envelope baseline preflight for Ember v0.0.35.

v0.0.34 showed that introducing lookup/fetch_url/read_file made the baseline
itself unstable. This revision changes only the prompt/tool vocabulary: every
90-case request is expressed with the exact v0.0.8-style system instruction for
one of weather, calculator, or web_search. No optimizer, training, GPU job,
promotion, deployment, or model write is present here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import ember_envelope_preflight_v034 as prior

base = prior.base
copy_data = prior.copy_data
EOT = prior.EOT
TOOL = prior.TOOL
DEFAULT_CONFIG = ROOT / "config/ember_envelope_preflight_v0.0.35.json"

TOOL_BY_KIND = {
    "short_code": ("web_search", "query"),
    "long_code": ("web_search", "query"),
    "digits": ("calculator", "expression"),
    "model_id": ("web_search", "query"),
    "url": ("web_search", "query"),
    "path": ("web_search", "query"),
    "entity": ("weather", "location"),
    "expression": ("calculator", "expression"),
    "mixed": ("web_search", "query"),
}

# Keep these byte-for-byte aligned with the demonstrated v0.0.8 tool prompts.
SYSTEM_BY_TOOL = {
    "weather": "You are Ember. When current weather is requested, call the weather tool with JSON arguments.",
    "calculator": "You are Ember. Use the calculator tool for arithmetic and provide JSON arguments.",
    "web_search": "You are Ember. Use web_search when the user requests current information.",
}


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if path.resolve() != DEFAULT_CONFIG.resolve() or cfg.get("version") != "0.0.35":
        raise ValueError("unsupported v0.0.35 envelope preflight configuration")
    if any(cfg.get(key) is not False for key in (
        "training_authorized", "gpu_training_authorized", "production_authorized"
    )):
        raise ValueError("v0.0.35 is preflight-only; all training and production authorization must be false")
    if int(cfg.get("expected_cases", 0)) != 90:
        raise ValueError("v0.0.35 requires all 90 held-out cases")
    for key in ("minimum_envelope_json_valid_rate", "minimum_envelope_tool_name_rate"):
        threshold = float(cfg.get("baseline_gate", {}).get(key, -1))
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"invalid baseline threshold: {key}")
    ref = ROOT / str(cfg.get("prompt_reference", ""))
    reference = json.loads(ref.read_text(encoding="utf-8"))
    expected = {
        "weather": SYSTEM_BY_TOOL["weather"],
        "calculator": SYSTEM_BY_TOOL["calculator"],
        "web_search": SYSTEM_BY_TOOL["web_search"],
    }
    found = {}
    for case in reference.get("cases", []):
        if case.get("kind") != "tool_call":
            continue
        tool = case.get("expected_tool")
        if tool in expected:
            prompt = case.get("prompt", "")
            system = prompt.split("<|system|>\n", 1)[1].split("\n<|user|>", 1)[0]
            found[tool] = system
    if found != expected:
        raise ValueError("v0.0.8 demonstrated system prompts changed or are incomplete")
    return cfg


def _user_request(tool: str, target: str, old: str, fallback: str) -> str:
    if tool == "weather":
        return (
            f"What is the weather in {target} right now? "
            f"Ignore {old} and {fallback}; they are not the requested location."
        )
    if tool == "calculator":
        return (
            f"Calculate {target}. "
            f"Ignore {old} and {fallback}; they are not the requested expression."
        )
    if tool == "web_search":
        return (
            f"Find current information about {target}. "
            f"Ignore {old} and {fallback}; they are not the requested query."
        )
    raise ValueError(tool)


def build_cases() -> list[dict]:
    diagnostics = list(copy_data.DIAGNOSTICS)
    if len(diagnostics) != 90:
        raise ValueError(f"expected 90 held-out values, got {len(diagnostics)}")
    by_kind: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for kind, value, corrupt in diagnostics:
        by_kind[kind].append((value, corrupt))
    if set(by_kind) != set(copy_data.KINDS) or any(len(rows) != 10 for rows in by_kind.values()):
        raise ValueError("held-out battery must remain balanced at ten cases per kind")

    counters = defaultdict(int)
    cases = []
    for kind, value, corrupt in diagnostics:
        tool, argument_key = TOOL_BY_KIND[kind]
        index = counters[kind]
        counters[kind] += 1
        rows = by_kind[kind]
        fallback = rows[(index + 1) % len(rows)][0]
        if fallback == value:
            fallback = rows[(index + 1) % len(rows)][1]
        prompt = (
            f"<|system|>\n{SYSTEM_BY_TOOL[tool]}\n"
            f"<|user|>\n{_user_request(tool, value, corrupt, fallback)}\n"
            "<|assistant|>\n"
        )
        cases.append({
            "id": f"known_tool_{kind}_{index:02d}",
            "kind": kind,
            "target": value,
            "old": corrupt,
            "fallback": fallback,
            "expected_tool": tool,
            "argument_key": argument_key,
            "prompt": prompt,
        })
    return cases


def score_case(case: dict, completion: str) -> dict:
    return prior.score_case(case, completion)


def summarize(rows: list[dict], cfg: dict) -> dict:
    return prior.summarize(rows, cfg)


def summary_markdown(report: dict) -> str:
    result = report["baseline_gate"]
    m = result["metrics"]
    lines = [
        f"# Ember v0.0.35 known-tool envelope preflight: {report['status']}", "",
        "CPU baseline only. No optimizer was created; no training, GPU submission, promotion, or production integration occurred.", "",
        "| Metric | Result |", "| --- | ---: |",
        f"| Envelope JSON valid | {m['envelope_json_valid']}/{m['cases']} ({m['envelope_json_valid_rate']:.1%}) |",
        f"| Correct tool name | {m['envelope_tool_name_correct']}/{m['cases']} ({m['envelope_tool_name_rate']:.1%}) |",
        f"| Slot evaluable | {m['slot_evaluable']}/{m['cases']} ({m['slot_evaluable_rate']:.1%}) |",
        (
            f"| Slot exact | {m['slot_exact']}/{m['slot_evaluable']} ({m['slot_exact_rate']:.1%}) |"
            if m["slot_exact_rate"] is not None
            else "| Slot exact | undefined (no evaluable envelopes) |"
        ),
        f"| Right envelope + tool, wrong value | {m['right_envelope_tool_wrong_value']}/{m['cases']} ({m['right_envelope_tool_wrong_value_rate']:.1%}) |",
        "",
        "Baseline checks: " + ", ".join(
            f"`{key}`={'PASS' if value else 'FAIL'}" for key, value in result["checks"].items()
        ) + ".",
    ]
    if result["passed"]:
        lines += [
            "",
            "The baseline envelope is stable enough for slot_exact to measure placement on this 90-case prompt family. This PASS does not authorize training.",
        ]
    else:
        lines += [
            "",
            "The baseline envelope is still not stable enough. Stop here; no future learning phase may consume this report as a valid baseline.",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("v035-results"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not os.environ.get("HF_TOKEN", "").strip():
        raise ValueError("HF_TOKEN is required to read the pinned private checkpoint")

    import torch

    torch.set_num_threads(2)
    torch.manual_seed(int(cfg["seed"]))
    torch.use_deterministic_algorithms(True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "version": "0.0.35",
        "phase": cfg["phase"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ERROR",
        "code_commit": os.environ.get("GITHUB_SHA"),
        "training_authorized": False,
        "gpu_training_authorized": False,
        "production_authorized": False,
        "prompt_reference": cfg["prompt_reference"],
        "tool_mapping": {kind: {"name": tool, "argument_key": key} for kind, (tool, key) in TOOL_BY_KIND.items()},
    }
    try:
        source_cfg = json.loads(base.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        if source_cfg.get("version") != "0.0.32":
            raise ValueError("unexpected source-loader configuration")
        with tempfile.TemporaryDirectory(prefix="ember-v035-") as td:
            model, tokenizer, source, _splits, reference = base.load_inputs(source_cfg, Path(td), torch)
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != cfg["source_version"]:
                raise ValueError("v0.0.35 must measure the pinned v0.0.31 step-479 baseline")
            rows = []
            for case in build_cases():
                generation = base.semantic_gate.generate_completion(
                    model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"])
                )
                score = score_case(case, generation["completion"])
                rows.append({**case, **generation, "score": score})
                print(json.dumps({
                    "event": "case", "id": case["id"], "kind": case["kind"],
                    "json_valid": score["envelope_json_valid"],
                    "tool_name_correct": score["tool_name_correct"],
                    "slot_exact": score["slot_exact"],
                }), flush=True)
            gate = summarize(rows, cfg)
            report.update(
                source={
                    "repo_id": reference["repo_id"],
                    "checkpoint_path": reference["checkpoint_path"],
                    "revision": reference["revision"],
                    "checkpoint_sha256": reference["checkpoint_sha256"],
                    "step": source["step"],
                    "version": source["train_config"]["version"],
                },
                cases=rows,
                baseline_gate=gate,
                status="PASS" if gate["passed"] else "FAIL",
                meaning="Known v0.0.8-style tool-envelope baseline only; slot_exact is diagnostic and no training is authorized.",
            )
    except Exception as exc:
        report["status"] = "ERROR"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        (output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        if "baseline_gate" in report:
            (output / "summary.md").write_text(summary_markdown(report), encoding="utf-8")

    print(json.dumps({"event": "complete", "status": report["status"], **report["baseline_gate"]["metrics"]}), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
