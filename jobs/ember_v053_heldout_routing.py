"""Held-out routing robustness challenge for saved Ember v0.0.53 step 9.

Read-only, CPU-only evaluation of the exact saved v0.0.53 step-9 candidate.
Uses 20 new prompts not present in the fixed promotion eval or routing-anchor
training set: 12 direct-response cases and 8 tool-required cases. Several direct
cases deliberately contain tool-adjacent words (weather, calculator, time zone,
web search, latest/current) without requiring a tool, to expose keyword-triggered
routing regressions.

The challenge evaluates both the full checkpoint and its saved INT4 companion.
No weights are changed, saved, promoted, or integrated.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

from huggingface_hub import HfApi, hf_hub_download
import torch

from jobs import ember_hf_eval as ev
from jobs import ember_v052_first_token_logits as ft

OUT = Path("v053-heldout-routing")
MODEL_NAME = "ember-v0.0.53-t4"
RUN_ID = "ember-v053-step9-20260910T165742Z"
BEST_PATH = f"checkpoints/{RUN_ID}/best.pt"
INT4_PATH = f"checkpoints/{RUN_ID}/best.int4.pt"
BEST_SHA256 = "700e4259b2e724c8fa384e23ba1a6e67636c4f110863e887499f06951622c9ad"
INT4_SHA256 = "d77c466fb8a9e602796aed3dffa4070b55cae8b492057b8f59a672ab6a1c1c78"

GENERATION = {
    "max_new_tokens": 64,
    "temperature": 1.0,
    "top_k": 1,
    "seed": 20260910,
}

SYSTEM_A = (
    "You are Ember. You may answer normally or use one of these tools: weather, "
    "calculator, web_search, get_time. Use a tool only when live weather, arithmetic, "
    "current web information, or current local time is actually needed. Otherwise "
    "answer directly. Tool calls use JSON arguments."
)
SYSTEM_B = (
    "You are Ember. Available tools are weather, calculator, web_search, and get_time. "
    "Choose a tool only when the request truly needs one; otherwise respond in plain "
    "language. Return JSON arguments for tool calls."
)


def prompt(system: str, user: str) -> str:
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"


def case(case_id: str, kind: str, user: str, *, system: str = SYSTEM_A, expected_tool: str | None = None) -> dict:
    row = {"id": case_id, "kind": kind, "prompt": prompt(system, user), "user": user}
    if expected_tool is not None:
        row["expected_tool"] = expected_tool
    return row


CASES = [
    # Direct responses: intentionally include tool-adjacent lexical lures.
    case("direct_welcome", "direct_response", "Write a friendly one-sentence welcome for a new teammate."),
    case("direct_rewrite_bug", "direct_response", "Rewrite this clearly as a bug title: button thing opens wrong page.", system=SYSTEM_B),
    case("direct_weather_concept", "direct_response", "Explain in one sentence why weather forecasts can change from day to day."),
    case("direct_time_concept", "direct_response", "In one sentence, explain what a time zone is.", system=SYSTEM_B),
    case("direct_calculator_concept", "direct_response", "Give two short reasons a calculator is useful in everyday work."),
    case("direct_websearch_concept", "direct_response", "Explain what web search does without searching the web.", system=SYSTEM_B),
    case("direct_latest_rewrite", "direct_response", "Rewrite this phrase more clearly: latest update stuff."),
    case("direct_current_status_rewrite", "direct_response", "Make this message friendlier: Current status is unknown; check again later.", system=SYSTEM_B),
    case("direct_plan", "direct_response", "Give me three short steps for checking a broken navigation link."),
    case("direct_sentiment", "direct_response", "Label the sentiment as positive, negative, or neutral: The deployment finally worked.", system=SYSTEM_B),
    case("direct_summary", "direct_response", "Summarize in one sentence: The tests passed, the build succeeded, and deployment is ready."),
    case("direct_compare", "direct_response", "Explain the difference between a cache and a database in two short sentences.", system=SYSTEM_B),

    # Tool-required: phrased differently from the original fixed eval.
    case("tool_weather_umbrella", "tool_call", "Should I bring an umbrella in Portland, Oregon right now?", expected_tool="weather"),
    case("tool_weather_temperature", "tool_call", "What is the temperature in Miami at the moment?", system=SYSTEM_B, expected_tool="weather"),
    case("tool_calculator_divide", "tool_call", "What is 9187 divided by 43?", expected_tool="calculator"),
    case("tool_calculator_percent", "tool_call", "Calculate 17.5 percent of 864.", system=SYSTEM_B, expected_tool="calculator"),
    case("tool_search_node", "tool_call", "What is the newest stable Node.js release?", expected_tool="web_search"),
    case("tool_search_python", "tool_call", "Find the most recent official Python security release.", system=SYSTEM_B, expected_tool="web_search"),
    case("tool_time_lisbon", "tool_call", "What time is it in Lisbon right now?", expected_tool="get_time"),
    case("tool_time_seoul", "tool_call", "Tell me the current local time in Seoul.", system=SYSTEM_B, expected_tool="get_time"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_full(repo: str, work: Path, token: str):
    local = Path(hf_hub_download(
        repo_id=repo,
        repo_type="model",
        filename=BEST_PATH,
        token=token,
        local_dir=work / "full",
    ))
    if sha256_file(local) != BEST_SHA256:
        raise RuntimeError("full v0.0.53 checkpoint SHA256 mismatch")
    model, tokenizer, checkpoint = ft.load_model(repo, BEST_PATH, work / "loaded-full", work)
    cfg = checkpoint.get("train_config", {})
    if cfg.get("version") != "0.0.53" or int(cfg.get("anchor_steps", -1)) != 9:
        raise RuntimeError(f"checkpoint metadata mismatch: {cfg}")
    return model, tokenizer, checkpoint


def load_int4(repo: str, work: Path, token: str):
    from src.model import EmberGPT, ModelConfig
    from src.quantize_int4 import dequantize_tensor
    from src.tokenizer import tokenizer_from_state_dict

    local = Path(hf_hub_download(
        repo_id=repo,
        repo_type="model",
        filename=INT4_PATH,
        token=token,
        local_dir=work / "int4",
    ))
    if sha256_file(local) != INT4_SHA256:
        raise RuntimeError("INT4 v0.0.53 checkpoint SHA256 mismatch")
    payload = torch.load(local, map_location="cpu", weights_only=False)
    state = {name: dequantize_tensor(value) for name, value in payload["quantized_state"].items()}
    state.update(payload.get("passthrough_state", {}))
    model = EmberGPT(ModelConfig(**payload["model_config"]))
    model.load_state_dict(state)
    model.to("cpu").eval()
    tokenizer = tokenizer_from_state_dict(payload["tokenizer"])
    return model, tokenizer


def evaluate(model, tokenizer, label: str) -> dict:
    torch.manual_seed(int(GENERATION["seed"]))
    rows = []
    direct_first = direct_generation = tool_first = tool_generation = 0
    direct_total = tool_total = 0
    tool_name_correct = 0

    for c in CASES:
        first = ft.inspect_case(model, tokenizer, c)
        completion = ev.generate_completion(model, tokenizer, torch, c["prompt"], GENERATION)
        score = ev.score_case(c, completion)
        if c["kind"] == "direct_response":
            direct_total += 1
            first_ok = not bool(first["argmax_is_tool"])
            gen_ok = bool(score["passed"])
            direct_first += int(first_ok)
            direct_generation += int(gen_ok)
        else:
            tool_total += 1
            first_ok = bool(first["argmax_is_tool"])
            gen_ok = bool(score["passed"])
            tool_first += int(first_ok)
            tool_generation += int(gen_ok)
            tool_name_correct += int(bool(score.get("tool_name_matches")))

        row = {
            "id": c["id"],
            "kind": c["kind"],
            "user": c["user"],
            "expected_tool": c.get("expected_tool"),
            "first_token": {
                "argmax": first["argmax_token"],
                "argmax_is_tool": first["argmax_is_tool"],
                "tool_probability": first["tool_probability"],
                "tool_rank": first["tool_rank"],
                "tool_margin": first["tool_minus_best_non_special_logit"],
                "best_lexical": first["best_non_special_token"],
            },
            "completion": completion,
            "score": score,
            "first_token_pass": first_ok,
            "generation_pass": gen_ok,
        }
        rows.append(row)
        print(json.dumps({
            "event": "heldout_case",
            "model": label,
            "id": c["id"],
            "kind": c["kind"],
            "expected_tool": c.get("expected_tool"),
            "argmax": first["argmax_token"],
            "tool_probability": first["tool_probability"],
            "tool_rank": first["tool_rank"],
            "first_pass": first_ok,
            "generation_pass": gen_ok,
            "score": score,
            "completion_prefix": completion[:180].replace("\n", "\\n"),
        }), flush=True)

    metrics = {
        "direct_first_token_pass": direct_first,
        "direct_total": direct_total,
        "tool_first_token_pass": tool_first,
        "tool_total": tool_total,
        "direct_generation_pass": direct_generation,
        "tool_generation_pass": tool_generation,
        "tool_name_correct": tool_name_correct,
    }
    metrics["strict_pass"] = (
        direct_first == direct_total
        and tool_first == tool_total
        and direct_generation == direct_total
        and tool_generation == tool_total
        and tool_name_correct == tool_total
    )
    return {"label": label, "metrics": metrics, "cases": rows}


def summary_markdown(report: dict) -> str:
    lines = [
        "# Ember v0.0.53 held-out routing robustness challenge",
        "",
        "Exact saved step-9 candidate; 20 new prompts (12 direct, 8 tool-required).",
        "Direct cases include tool-adjacent lexical lures. Both full and INT4 checkpoints are tested.",
        "",
        "| Model | Direct first-token | Tool first-token | Direct generation | Tool generation | Tool name | Strict pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in ("full", "int4"):
        m = report[key]["metrics"]
        lines.append(
            f"| {key.upper()} | {m['direct_first_token_pass']}/{m['direct_total']} | "
            f"{m['tool_first_token_pass']}/{m['tool_total']} | "
            f"{m['direct_generation_pass']}/{m['direct_total']} | "
            f"{m['tool_generation_pass']}/{m['tool_total']} | "
            f"{m['tool_name_correct']}/{m['tool_total']} | {m['strict_pass']} |"
        )

    failures = []
    for key in ("full", "int4"):
        for row in report[key]["cases"]:
            if not row["first_token_pass"] or not row["generation_pass"]:
                failures.append((key, row))
    lines += ["", "## Failures", ""]
    if not failures:
        lines.append("None.")
    else:
        for key, row in failures:
            f = row["first_token"]
            lines.append(
                f"- **{key.upper()} / {row['id']}**: argmax `{f['argmax']}`, "
                f"P(tool)={f['tool_probability']:.6f}, rank={f['tool_rank']}, "
                f"margin={f['tool_margin']:+.4f}; first={row['first_token_pass']}, generation={row['generation_pass']}."
            )
    lines += [
        "",
        f"Overall strict pass: **{report['overall_strict_pass']}**",
        "",
        "No training, checkpoint write, promotion, or production integration occurred.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    api = HfApi(token=token)
    owner = api.whoami()["name"]
    repo = f"{owner}/{MODEL_NAME}"

    with tempfile.TemporaryDirectory(prefix="ember-v053-heldout-") as td:
        work = Path(td)
        archive = ev.download_verified(ev.PACKAGE_URL, work / "ember.zip", ev.PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        import sys
        sys.path.insert(0, str(work / "src" / "ember"))

        full_model, full_tok, full_checkpoint = load_full(repo, work, token)
        full = evaluate(full_model, full_tok, "full")
        del full_model, full_checkpoint

        int4_model, int4_tok = load_int4(repo, work, token)
        int4 = evaluate(int4_model, int4_tok, "int4")
        del int4_model

    report = {
        "schema_version": 1,
        "challenge": "ember-v053-heldout-routing-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_repo": repo,
        "run_id": RUN_ID,
        "checkpoint": BEST_PATH,
        "checkpoint_sha256": BEST_SHA256,
        "int4": INT4_PATH,
        "int4_sha256": INT4_SHA256,
        "generation": GENERATION,
        "case_count": len(CASES),
        "direct_cases": sum(c["kind"] == "direct_response" for c in CASES),
        "tool_cases": sum(c["kind"] == "tool_call" for c in CASES),
        "full": full,
        "int4": int4,
        "overall_strict_pass": bool(full["metrics"]["strict_pass"] and int4["metrics"]["strict_pass"]),
        "read_only": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
    print(json.dumps({
        "event": "heldout_complete",
        "full": full["metrics"],
        "int4": int4["metrics"],
        "overall_strict_pass": report["overall_strict_pass"],
        "elapsed_seconds": report["elapsed_seconds"],
    }), flush=True)


if __name__ == "__main__":
    main()
