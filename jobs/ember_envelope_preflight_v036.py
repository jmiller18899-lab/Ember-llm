"""CPU-only single-target envelope baseline preflight for Ember v0.0.36.

Two measurements run against the pinned v0.0.31 step-479 checkpoint:
1. the four exact tool-call prompts from config/ember_v0.0.8_eval.json; and
2. the 90 historical held-out values expressed as one natural requested value.

There is no optimizer, model write, GPU submission, promotion, or deployment.
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

from jobs import ember_envelope_preflight_v035 as prior

base = prior.base
copy_data = prior.copy_data
EOT = prior.EOT
TOOL = prior.TOOL
DEFAULT_CONFIG = ROOT / "config/ember_envelope_preflight_v0.0.36.json"

TOOL_BY_KIND = dict(prior.TOOL_BY_KIND)
SYSTEM_BY_TOOL = dict(prior.SYSTEM_BY_TOOL)


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if path.resolve() != DEFAULT_CONFIG.resolve() or cfg.get("version") != "0.0.36":
        raise ValueError("unsupported v0.0.36 envelope preflight configuration")
    if any(cfg.get(key) is not False for key in (
        "training_authorized", "gpu_training_authorized", "production_authorized"
    )):
        raise ValueError("v0.0.36 is preflight-only; training and production authorization must be false")
    if int(cfg.get("expected_cases", 0)) != 90 or int(cfg.get("expected_reference_cases", 0)) != 4:
        raise ValueError("v0.0.36 requires 90 held-out cases and four exact reference tool cases")
    for group in ("reference_gate", "baseline_gate"):
        for key in ("minimum_envelope_json_valid_rate", "minimum_envelope_tool_name_rate"):
            threshold = float(cfg.get(group, {}).get(key, -1))
            if not 0.0 < threshold <= 1.0:
                raise ValueError(f"invalid {group} threshold: {key}")
    return cfg


def reference_cases(cfg: dict) -> list[dict]:
    reference = json.loads((ROOT / cfg["prompt_reference"]).read_text(encoding="utf-8"))
    cases = [
        {"id": case["id"], "prompt": case["prompt"], "expected_tool": case["expected_tool"]}
        for case in reference.get("cases", []) if case.get("kind") == "tool_call"
    ]
    if len(cases) != int(cfg["expected_reference_cases"]):
        raise ValueError("reference prompt no longer contains exactly four tool-call cases")
    expected_prompts = {
        "tool_weather": (
            "weather",
            "<|system|>\nYou are Ember. When current weather is requested, call the weather tool with JSON arguments.\n"
            "<|user|>\nWhat is the weather in Detroit right now?\n<|assistant|>\n",
        ),
        "tool_calculator": (
            "calculator",
            "<|system|>\nYou are Ember. Use the calculator tool for arithmetic and provide JSON arguments.\n"
            "<|user|>\nCalculate 347 multiplied by 28.\n<|assistant|>\n",
        ),
        "tool_web_search": (
            "web_search",
            "<|system|>\nYou are Ember. Use web_search when the user requests current information.\n"
            "<|user|>\nFind the latest published release of Python.\n<|assistant|>\n",
        ),
        "tool_get_time": (
            "get_time",
            "<|system|>\nYou are Ember. Use get_time for the current time in a requested timezone.\n"
            "<|user|>\nWhat time is it in Tokyo?\n<|assistant|>\n",
        ),
    }
    actual = {case["id"]: (case["expected_tool"], case["prompt"]) for case in cases}
    if actual != expected_prompts:
        raise ValueError("v0.0.8 reference tool prompts changed")
    return cases


def _user_request(tool: str, target: str) -> str:
    if tool == "weather":
        return f"What is the weather in {target} right now?"
    if tool == "calculator":
        return f"Calculate {target}."
    if tool == "web_search":
        return f"Find current information about {target}."
    raise ValueError(tool)


def build_cases() -> list[dict]:
    diagnostics = list(copy_data.DIAGNOSTICS)
    if len(diagnostics) != 90:
        raise ValueError(f"expected 90 held-out values, got {len(diagnostics)}")
    by_kind = defaultdict(int)
    cases = []
    for kind, value, _corrupt in diagnostics:
        index = by_kind[kind]
        by_kind[kind] += 1
        tool, argument_key = TOOL_BY_KIND[kind]
        prompt = (
            f"<|system|>\n{SYSTEM_BY_TOOL[tool]}\n"
            f"<|user|>\n{_user_request(tool, value)}\n"
            "<|assistant|>\n"
        )
        cases.append({
            "id": f"single_target_{kind}_{index:02d}",
            "kind": kind,
            "target": value,
            "expected_tool": tool,
            "argument_key": argument_key,
            "prompt": prompt,
        })
    if set(by_kind) != set(copy_data.KINDS) or any(by_kind[k] != 10 for k in copy_data.KINDS):
        raise ValueError("held-out battery must remain balanced at ten cases per kind")
    return cases


def _strict_envelope(completion: str) -> tuple[dict | None, str | None, bool]:
    body, separator, tail = completion.partition(EOT)
    body = body.strip()
    clean_stop = bool(separator) and completion.count(EOT) == 1 and not tail.strip()
    try:
        if not body.startswith(TOOL):
            raise ValueError("tool envelope must start at the beginning of the answer")
        payload = base.semantic_gate.strict_json(body[len(TOOL):].strip())
        if not isinstance(payload, dict) or set(payload) != {"name", "arguments"}:
            raise ValueError("tool envelope must contain only name and arguments")
        if not isinstance(payload.get("name"), str) or not isinstance(payload.get("arguments"), dict):
            raise ValueError("tool envelope fields have invalid types")
        return payload, None, clean_stop
    except (ValueError, RecursionError) as exc:
        return None, str(exc), clean_stop


def score_reference(case: dict, completion: str) -> dict:
    payload, error, clean_stop = _strict_envelope(completion)
    valid = payload is not None
    return {
        "envelope_json_valid": valid,
        "tool_name_correct": valid and payload["name"] == case["expected_tool"],
        "clean_stop": clean_stop,
        "json_error": error,
    }


def score_case(case: dict, completion: str) -> dict:
    return prior.score_case(case, completion)


def reference_summary(rows: list[dict], cfg: dict) -> dict:
    total = len(rows)
    if total != int(cfg["expected_reference_cases"]):
        raise ValueError("incomplete reference probe")
    valid = sum(bool(row["score"]["envelope_json_valid"]) for row in rows)
    tool = sum(bool(row["score"]["tool_name_correct"]) for row in rows)
    metrics = {
        "cases": total,
        "envelope_json_valid": valid,
        "envelope_json_valid_rate": valid / total,
        "envelope_tool_name_correct": tool,
        "envelope_tool_name_rate": tool / total,
        "clean_stop_rate": sum(bool(row["score"]["clean_stop"]) for row in rows) / total,
    }
    thresholds = cfg["reference_gate"]
    checks = {
        "reference_envelope_json_valid": metrics["envelope_json_valid_rate"] >= float(thresholds["minimum_envelope_json_valid_rate"]),
        "reference_envelope_tool_name": metrics["envelope_tool_name_rate"] >= float(thresholds["minimum_envelope_tool_name_rate"]),
    }
    return {"passed": all(checks.values()), "checks": checks, "metrics": metrics}


def baseline_summary(rows: list[dict], cfg: dict) -> dict:
    return prior.summarize(rows, cfg)


def summary_markdown(report: dict) -> str:
    ref = report["reference_gate"]
    base_gate = report["baseline_gate"]
    r, m = ref["metrics"], base_gate["metrics"]
    return "\n".join([
        f"# Ember v0.0.36 single-target envelope preflight: {report['status']}", "",
        "CPU baseline only. No optimizer, training, GPU submission, promotion, or production integration occurred.", "",
        "## Exact v0.0.8 reference", "",
        f"- JSON-valid envelope: {r['envelope_json_valid']}/{r['cases']} ({r['envelope_json_valid_rate']:.1%})",
        f"- Correct tool name: {r['envelope_tool_name_correct']}/{r['cases']} ({r['envelope_tool_name_rate']:.1%})",
        f"- Reference gate: {'PASS' if ref['passed'] else 'FAIL'}", "",
        "## 90-case single-target baseline", "",
        f"- JSON-valid envelope: {m['envelope_json_valid']}/{m['cases']} ({m['envelope_json_valid_rate']:.1%})",
        f"- Correct tool name: {m['envelope_tool_name_correct']}/{m['cases']} ({m['envelope_tool_name_rate']:.1%})",
        f"- Slot evaluable: {m['slot_evaluable']}/{m['cases']} ({m['slot_evaluable_rate']:.1%})",
        (f"- Slot exact: {m['slot_exact']}/{m['slot_evaluable']} ({m['slot_exact_rate']:.1%})"
         if m["slot_exact_rate"] is not None else "- Slot exact: undefined"),
        f"- Right envelope + tool, wrong value: {m['right_envelope_tool_wrong_value']}/{m['cases']} ({m['right_envelope_tool_wrong_value_rate']:.1%})",
        f"- 90-case gate: {'PASS' if base_gate['passed'] else 'FAIL'}", "",
        ("Both envelope gates passed; slot_exact is now interpretable as the placement measurement. This still does not authorize training."
         if report["status"] == "PASS"
         else "At least one envelope gate failed. Stop here; no learning phase may consume this baseline and no GPU should be spent."),
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("v036-results"))
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
        "version": "0.0.36",
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
        with tempfile.TemporaryDirectory(prefix="ember-v036-") as td:
            model, tokenizer, source, _splits, source_ref = base.load_inputs(source_cfg, Path(td), torch)
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != cfg["source_version"]:
                raise ValueError("v0.0.36 must measure the pinned v0.0.31 step-479 baseline")

            reference_rows = []
            for case in reference_cases(cfg):
                generation = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"]))
                score = score_reference(case, generation["completion"])
                reference_rows.append({**case, **generation, "score": score})
                print(json.dumps({"event": "reference_case", "id": case["id"], **{k: score[k] for k in ("envelope_json_valid", "tool_name_correct")}}), flush=True)
            ref_gate = reference_summary(reference_rows, cfg)

            rows = []
            for case in build_cases():
                generation = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"]))
                score = score_case(case, generation["completion"])
                rows.append({**case, **generation, "score": score})
                print(json.dumps({
                    "event": "case", "id": case["id"], "kind": case["kind"],
                    "json_valid": score["envelope_json_valid"],
                    "tool_name_correct": score["tool_name_correct"],
                    "slot_exact": score["slot_exact"],
                }), flush=True)
            gate = baseline_summary(rows, cfg)
            passed = ref_gate["passed"] and gate["passed"]
            report.update(
                source={
                    "repo_id": source_ref["repo_id"], "checkpoint_path": source_ref["checkpoint_path"],
                    "revision": source_ref["revision"], "checkpoint_sha256": source_ref["checkpoint_sha256"],
                    "step": source["step"], "version": source["train_config"]["version"],
                },
                reference_cases=reference_rows,
                reference_gate=ref_gate,
                cases=rows,
                baseline_gate=gate,
                status="PASS" if passed else "FAIL",
                meaning="Exact v0.0.8 reference plus 90-case single-target envelope baseline only; no training is authorized.",
            )
    except Exception as exc:
        report["status"] = "ERROR"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        if "reference_gate" in report and "baseline_gate" in report:
            (output / "summary.md").write_text(summary_markdown(report), encoding="utf-8")

    print(json.dumps({
        "event": "complete", "status": report["status"],
        "reference": report["reference_gate"]["metrics"],
        "baseline": report["baseline_gate"]["metrics"],
    }), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
