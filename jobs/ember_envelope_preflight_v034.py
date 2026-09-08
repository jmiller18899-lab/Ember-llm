"""CPU-only baseline envelope preflight for Ember v0.0.34.

This phase does not train, optimize, promote, deploy, or submit a GPU job. It
asks the pinned v0.0.31 step-479 model to place each of the 90 historical held-
out copy values into a semantically appropriate tool-call argument. Prompts use
the agent style established by config/ember_v0.0.8_eval.json: a short task-
shaped system instruction followed by a natural user request.

The baseline must first demonstrate that the tool envelope itself is a stable
behavior. Only then is slot_exact a meaningful placement diagnostic.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import ember_semantic_train_v032 as base
from jobs import ember_sft_data_v026 as copy_data

DEFAULT_CONFIG = ROOT / "config/ember_envelope_preflight_v0.0.34.json"
EOT = base.semantic_data.EOT
TOOL = base.semantic_data.TOOL

# Every historical copy kind gets a tool whose argument has a natural semantic
# relationship to the value. The mapping is deliberately simple and fixed so
# routing competence is not entangled with the placement measurement.
TOOL_BY_KIND = {
    "short_code": ("web_search", "query"),
    "long_code": ("lookup", "key"),
    "digits": ("calculator", "expression"),
    "model_id": ("web_search", "query"),
    "url": ("fetch_url", "url"),
    "path": ("read_file", "path"),
    "entity": ("weather", "location"),
    "expression": ("calculator", "expression"),
    "mixed": ("lookup", "key"),
}

SYSTEM_BY_TOOL = {
    "weather": "You are Ember. When current weather is requested, call the weather tool with JSON arguments.",
    "calculator": "You are Ember. Use the calculator tool for arithmetic and provide JSON arguments.",
    "web_search": "You are Ember. Use web_search when the user asks you to look up information.",
    "lookup": "You are Ember. Use lookup when the user asks you to retrieve a record by key.",
    "fetch_url": "You are Ember. Use fetch_url when the user asks you to fetch a URL.",
    "read_file": "You are Ember. Use read_file when the user asks you to read a file path.",
}


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if path.resolve() != DEFAULT_CONFIG.resolve() or cfg.get("version") != "0.0.34":
        raise ValueError("unsupported envelope preflight configuration")
    if any(cfg.get(key) is not False for key in (
        "training_authorized", "gpu_training_authorized", "production_authorized"
    )):
        raise ValueError("v0.0.34 is preflight-only; all training and production authorization must be false")
    gate = cfg.get("baseline_gate", {})
    for key in ("minimum_envelope_json_valid_rate", "minimum_envelope_tool_name_rate"):
        value = float(gate.get(key, -1))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"invalid baseline threshold: {key}")
    if int(cfg.get("expected_cases", 0)) != 90:
        raise ValueError("v0.0.34 requires the full 90-case held-out battery")
    ref = ROOT / str(cfg.get("prompt_reference", ""))
    prompt_spec = json.loads(ref.read_text(encoding="utf-8"))
    reference_tools = [case for case in prompt_spec.get("cases", []) if case.get("kind") == "tool_call"]
    if len(reference_tools) < 4 or any(
        "<|system|>\nYou are Ember." not in case.get("prompt", "")
        or "<|user|>\n" not in case.get("prompt", "")
        for case in reference_tools
    ):
        raise ValueError("v0.0.8 prompt reference no longer has the expected agent-style tool cases")
    return cfg


def _user_request(tool: str, target: str, old: str, fallback: str) -> str:
    if tool == "weather":
        return f"What is the current weather in {target}? Ignore the earlier location {old} and the fallback {fallback}."
    if tool == "calculator":
        return f"Calculate {target}. Ignore the earlier expression {old} and the fallback {fallback}."
    if tool == "web_search":
        return f"Search for {target}. Ignore the earlier query {old} and the fallback {fallback}."
    if tool == "lookup":
        return f"Look up the record {target}. Ignore the old key {old} and the fallback key {fallback}."
    if tool == "fetch_url":
        return f"Fetch {target}. Ignore the earlier URL {old} and the fallback URL {fallback}."
    if tool == "read_file":
        return f"Read the file at {target}. Ignore the old path {old} and the fallback path {fallback}."
    raise ValueError(f"unsupported tool: {tool}")


def build_cases() -> list[dict]:
    diagnostics = list(copy_data.DIAGNOSTICS)
    if len(diagnostics) != 90:
        raise ValueError(f"expected 90 held-out values, got {len(diagnostics)}")
    by_kind: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for kind, value, corrupt in diagnostics:
        by_kind[kind].append((value, corrupt))
    if set(by_kind) != set(copy_data.KINDS) or any(len(rows) != 10 for rows in by_kind.values()):
        raise ValueError("held-out battery is no longer balanced at ten cases per kind")
    cases = []
    counters = defaultdict(int)
    for kind, value, corrupt in diagnostics:
        tool, argument_key = TOOL_BY_KIND[kind]
        rows = by_kind[kind]
        index = counters[kind]
        counters[kind] += 1
        fallback = rows[(index + 1) % len(rows)][0]
        if fallback == value:
            fallback = rows[(index + 1) % len(rows)][1]
        user = _user_request(tool, value, corrupt, fallback)
        prompt = (
            f"<|system|>\n{SYSTEM_BY_TOOL[tool]}\n"
            f"<|user|>\n{user}\n<|assistant|>\n"
        )
        cases.append({
            "id": f"envelope_{kind}_{index:02d}",
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
    body, separator, tail = completion.partition(EOT)
    body = body.strip()
    clean_stop = bool(separator) and completion.count(EOT) == 1 and not tail.strip()
    payload = None
    error = None
    try:
        if not body.startswith(TOOL):
            raise ValueError("tool envelope must start at the beginning of the answer")
        payload = base.semantic_gate.strict_json(body[len(TOOL):].strip())
    except (ValueError, RecursionError) as exc:
        error = str(exc)
    envelope_json_valid = (
        isinstance(payload, dict)
        and set(payload) == {"name", "arguments"}
        and isinstance(payload.get("name"), str)
        and isinstance(payload.get("arguments"), dict)
    )
    tool_name_correct = envelope_json_valid and payload["name"] == case["expected_tool"]
    arguments = payload["arguments"] if envelope_json_valid else {}
    slot_exact = (
        tool_name_correct
        and set(arguments) == {case["argument_key"]}
        and arguments.get(case["argument_key"]) == case["target"]
    )
    right_envelope_tool_wrong_value = (
        tool_name_correct
        and set(arguments) == {case["argument_key"]}
        and arguments.get(case["argument_key"]) != case["target"]
    )
    return {
        "envelope_json_valid": envelope_json_valid,
        "tool_name_correct": tool_name_correct,
        "slot_exact": slot_exact,
        "right_envelope_tool_wrong_value": right_envelope_tool_wrong_value,
        "clean_stop": clean_stop,
        "argument_value": arguments.get(case["argument_key"]) if isinstance(arguments, dict) else None,
        "json_error": error,
    }


def summarize(rows: list[dict], cfg: dict) -> dict:
    total = len(rows)
    if total != int(cfg["expected_cases"]):
        raise ValueError("incomplete envelope preflight")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate envelope case IDs")
    json_valid = sum(bool(row["score"]["envelope_json_valid"]) for row in rows)
    tool_correct = sum(bool(row["score"]["tool_name_correct"]) for row in rows)
    slot_exact = sum(bool(row["score"]["slot_exact"]) for row in rows)
    wrong_value = sum(bool(row["score"]["right_envelope_tool_wrong_value"]) for row in rows)
    slot_evaluable = tool_correct
    metrics = {
        "cases": total,
        "envelope_json_valid": json_valid,
        "envelope_json_valid_rate": json_valid / total,
        "envelope_tool_name_correct": tool_correct,
        "envelope_tool_name_rate": tool_correct / total,
        "slot_evaluable": slot_evaluable,
        "slot_evaluable_rate": slot_evaluable / total,
        "slot_exact": slot_exact,
        "slot_exact_rate": slot_exact / slot_evaluable if slot_evaluable else None,
        "slot_exact_overall_rate": slot_exact / total,
        "right_envelope_tool_wrong_value": wrong_value,
        "right_envelope_tool_wrong_value_rate": wrong_value / total,
        "clean_stop_rate": sum(bool(row["score"]["clean_stop"]) for row in rows) / total,
    }
    thresholds = cfg["baseline_gate"]
    checks = {
        "baseline_envelope_json_valid": metrics["envelope_json_valid_rate"] >= float(
            thresholds["minimum_envelope_json_valid_rate"]
        ),
        "baseline_envelope_tool_name": metrics["envelope_tool_name_rate"] >= float(
            thresholds["minimum_envelope_tool_name_rate"]
        ),
    }
    by_kind = {}
    for kind in copy_data.KINDS:
        selected = [row for row in rows if row["kind"] == kind]
        by_kind[kind] = {
            "cases": len(selected),
            "json_valid": sum(bool(row["score"]["envelope_json_valid"]) for row in selected),
            "tool_name_correct": sum(bool(row["score"]["tool_name_correct"]) for row in selected),
            "slot_exact": sum(bool(row["score"]["slot_exact"]) for row in selected),
        }
    return {"passed": all(checks.values()), "checks": checks, "metrics": metrics, "by_kind": by_kind}


def summary_markdown(report: dict) -> str:
    result = report["baseline_gate"]
    m = result["metrics"]
    lines = [
        f"# Ember v0.0.34 envelope preflight: {report['status']}", "",
        "CPU baseline only. No optimizer was created; no training, GPU submission, promotion, or production integration occurred.", "",
        "| Metric | Result |", "| --- | ---: |",
        f"| Envelope JSON valid | {m['envelope_json_valid']}/{m['cases']} ({m['envelope_json_valid_rate']:.1%}) |",
        f"| Correct tool name | {m['envelope_tool_name_correct']}/{m['cases']} ({m['envelope_tool_name_rate']:.1%}) |",
        f"| Slot evaluable | {m['slot_evaluable']}/{m['cases']} ({m['slot_evaluable_rate']:.1%}) |",
        f"| Slot exact | {m['slot_exact']}/{m['slot_evaluable']} ({m['slot_exact_rate']:.1%}) |" if m["slot_exact_rate"] is not None else "| Slot exact | undefined (no evaluable envelopes) |",
        f"| Right envelope + tool, wrong value | {m['right_envelope_tool_wrong_value']}/{m['cases']} ({m['right_envelope_tool_wrong_value_rate']:.1%}) |",
        "",
        "Baseline checks: " + ", ".join(
            f"`{key}`={'PASS' if value else 'FAIL'}" for key, value in result["checks"].items()
        ) + ".",
    ]
    if result["passed"]:
        lines += ["", "The envelope baseline is stable enough for slot_exact to be interpreted as a placement measurement. This does not authorize a training phase."]
    else:
        lines += ["", "The envelope prompt/baseline is not stable enough. Stop here; a future training phase must not start from this report."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("v034-results"))
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
        "version": "0.0.34",
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
        with tempfile.TemporaryDirectory(prefix="ember-v034-") as td:
            model, tokenizer, source, _splits, reference = base.load_inputs(source_cfg, Path(td), torch)
            if source.get("step") != 479 or source.get("train_config", {}).get("version") != cfg["source_version"]:
                raise ValueError("v0.0.34 must measure the pinned v0.0.31 step-479 baseline")
            cases = build_cases()
            rows = []
            for case in cases:
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
                meaning="Agent-style envelope baseline preflight only; slot_exact is diagnostic and no training is authorized.",
            )
    except Exception as exc:
        report["status"] = "ERROR"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        if "baseline_gate" in report:
            (output / "summary.md").write_text(summary_markdown(report), encoding="utf-8")

    print(json.dumps({"event": "complete", "status": report["status"], **report["baseline_gate"]["metrics"]}), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
