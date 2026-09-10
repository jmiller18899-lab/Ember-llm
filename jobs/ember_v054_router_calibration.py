"""Calibrate Ember's frozen block-3 hierarchical router and test a third fresh set.

The prior hierarchical router reached 20/20 on the original held-out set and
49/50 on the second 50-case confirmation set. The one persistent miss was a
DIRECT sentiment-labeling request, while all 40 tool-family cases were correct.

This diagnostic keeps Ember v0.0.53 step-9 frozen and broadens ROUTER training
only:
  - 16 new direct classification/sentiment/taxonomy examples;
  - 16 new tool examples (4 per family), preserving class balance.

Final router-training balance:
  stage 1: 80 direct vs 80 tool;
  stage 2: 20 examples each for weather/calculator/web_search/get_time.

The original 20 cases and second 50-case set are now regression-validation sets.
A THIRD fresh 50-case set (10/class) is the untouched decision test. Ridge
strength is still selected only by training-set cross-validation.

Full, INT4, and one full-router-on-INT4 cross-precision path are evaluated.
No Ember weights, checkpoints, or production pointers are changed. Router tensors
are artifact-only if all strict gates pass.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

from huggingface_hub import HfApi
import torch

from jobs import ember_hf_eval as ev
from jobs import ember_v053_heldout_routing as held
from jobs import ember_v054_frozen_router_probe as probe
from jobs import ember_v054_router_confirmation as confirm
from jobs import ember_v054_routing_repair as v54
from jobs import ember_v054_hierarchical_router as hier

OUT = Path("v054-router-calibration")
REPRESENTATION = hier.REPRESENTATION

SYSTEM_E = (
    "You are Ember. Route the request to either a normal response or one tool: weather, calculator, web_search, get_time. "
    "Use tools only for live weather, requested arithmetic, current web facts, or current local time. "
    "Classification, rewriting, explanation, comparison, and sentiment labeling are normal responses."
)
SYSTEM_F = (
    "You are Ember. Decide whether the user needs a direct answer or a tool. Available tools are weather, calculator, web_search, get_time. "
    "Words such as current, latest, time, search, or weather do not require a tool unless the task actually needs live data or arithmetic."
)


def make_prompt(system: str, user: str) -> str:
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"


def direct(case_id: str, user: str, system: str = SYSTEM_E) -> dict:
    return {"id": case_id, "kind": "direct_response", "user": user, "prompt": make_prompt(system, user)}


def tool(case_id: str, user: str, expected_tool: str, system: str = SYSTEM_E) -> dict:
    return {
        "id": case_id,
        "kind": "tool_call",
        "user": user,
        "prompt": make_prompt(system, user),
        "expected_tool": expected_tool,
    }


def calibration_additions() -> list[dict]:
    direct_rows = [
        ("cal_d01", "Label the sentiment as positive, negative, or neutral: The deployment completed successfully.", SYSTEM_E),
        ("cal_d02", "Label the sentiment: The new layout is frustrating and confusing.", SYSTEM_F),
        ("cal_d03", "Classify this as success, warning, or error: Backup finished with no problems.", SYSTEM_E),
        ("cal_d04", "Classify this message as question or statement: The server is online.", SYSTEM_F),
        ("cal_d05", "Label the tone as formal or casual: Hey, thanks for fixing that bug!", SYSTEM_E),
        ("cal_d06", "Classify the word 'running' as a noun or verb in: The service is running.", SYSTEM_F),
        ("cal_d07", "Categorize JSON as text data or image data.", SYSTEM_E),
        ("cal_d08", "Classify this as positive or negative: The current build is stable now.", SYSTEM_F),
        ("cal_d09", "Label the intent as request or statement: Please rewrite the latest status title.", SYSTEM_E),
        ("cal_d10", "Classify this phrase as weather concept or arithmetic concept: chance of rain.", SYSTEM_F),
        ("cal_d11", "Label this sentence as factual or opinion: I think the search page looks better.", SYSTEM_E),
        ("cal_d12", "Classify this as a comparison or instruction: CSV is simpler than JSON for this table.", SYSTEM_F),
        ("cal_d13", "Label the sentiment: The calculator icon looks excellent after the redesign.", SYSTEM_E),
        ("cal_d14", "Classify this as direct writing or live lookup: Rewrite current time display as a shorter heading.", SYSTEM_F),
        ("cal_d15", "Label the sentiment as positive, neutral, or negative: Nothing changed in the latest build.", SYSTEM_E),
        ("cal_d16", "Classify this sentence as a command or description: The weather card is blue.", SYSTEM_F),
    ]
    rows = [direct(cid, user, system) for cid, user, system in direct_rows]

    # Four fresh examples per tool family.
    weather = [
        ("cal_w01", "What is the weather in Omaha right now?"),
        ("cal_w02", "Is it raining in Belfast at the moment?"),
        ("cal_w03", "Give me the live temperature in Kyoto."),
        ("cal_w04", "Check the current weather conditions in Tucson."),
    ]
    calc = [
        ("cal_c01", "Calculate 438 plus 967."),
        ("cal_c02", "What is 73 multiplied by 29?"),
        ("cal_c03", "Compute 5600 divided by 14."),
        ("cal_c04", "Calculate 27 percent of 640."),
    ]
    search = [
        ("cal_s01", "Find the newest stable Perl release."),
        ("cal_s02", "What is the current stable FreeBSD release?"),
        ("cal_s03", "Find the latest official Terraform release."),
        ("cal_s04", "Look up a recent official NOAA announcement."),
    ]
    times = [
        ("cal_t01", "What time is it in Dublin right now?"),
        ("cal_t02", "Give me the current local time in Mexico City."),
        ("cal_t03", "What is the time in Perth at this moment?"),
        ("cal_t04", "Tell me the current time in Athens."),
    ]
    for cid, user in weather:
        rows.append(tool(cid, user, "weather", SYSTEM_E if len(rows) % 2 == 0 else SYSTEM_F))
    for cid, user in calc:
        rows.append(tool(cid, user, "calculator", SYSTEM_F if len(rows) % 2 == 0 else SYSTEM_E))
    for cid, user in search:
        rows.append(tool(cid, user, "web_search", SYSTEM_E if len(rows) % 2 == 0 else SYSTEM_F))
    for cid, user in times:
        rows.append(tool(cid, user, "get_time", SYSTEM_F if len(rows) % 2 == 0 else SYSTEM_E))
    return rows


def training_cases() -> list[dict]:
    rows = hier.training_cases() + calibration_additions()
    counts = {label: sum(probe.label_of(c) == label for c in rows) for label in probe.LABELS}
    expected = {"direct": 80, "weather": 20, "calculator": 20, "web_search": 20, "get_time": 20}
    if counts != expected:
        raise RuntimeError(f"calibrated training balance mismatch: {counts} != {expected}")
    ids = [c["id"] for c in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate calibrated training ids")
    forbidden = {c["id"] for c in held.CASES} | {c["id"] for c in confirm.CONFIRM} | {c["id"] for c in THIRD_CONFIRM}
    overlap = forbidden.intersection(ids)
    if overlap:
        raise RuntimeError(f"training overlaps validation/test ids: {sorted(overlap)}")
    return rows


THIRD_CONFIRM = [
    # Direct / classification / ordinary language tasks.
    direct("third_d01", "Label the sentiment: The fix made the app much easier to use.", SYSTEM_F),
    direct("third_d02", "Classify this as a question or statement: The current weather icon is green.", SYSTEM_E),
    direct("third_d03", "Rewrite this title clearly: latest search panel wording cleanup.", SYSTEM_F),
    direct("third_d04", "Explain what a calculator is without doing any arithmetic.", SYSTEM_E),
    direct("third_d05", "Summarize: the time zone dropdown moved and all tests still pass.", SYSTEM_F),
    direct("third_d06", "Compare XML and JSON in two short sentences.", SYSTEM_E),
    direct("third_d07", "Label the tone as positive, neutral, or negative: The update did not change behavior.", SYSTEM_F),
    direct("third_d08", "Give two short steps for testing a logout button.", SYSTEM_E),
    direct("third_d09", "Define web search without performing a search.", SYSTEM_F),
    direct("third_d10", "Write one friendly sentence thanking someone for a bug report.", SYSTEM_E),

    # Weather.
    tool("third_w01", "What is the weather in Reno right now?", "weather", SYSTEM_F),
    tool("third_w02", "Is it raining in Cardiff at the moment?", "weather", SYSTEM_E),
    tool("third_w03", "Give me the live temperature in Sapporo.", "weather", SYSTEM_F),
    tool("third_w04", "How is the weather in Valencia right now?", "weather", SYSTEM_E),
    tool("third_w05", "Do I need an umbrella in Wellington at the moment?", "weather", SYSTEM_F),
    tool("third_w06", "Check the current weather in Milwaukee.", "weather", SYSTEM_E),
    tool("third_w07", "Is it sunny in Lyon right now?", "weather", SYSTEM_F),
    tool("third_w08", "What are the live weather conditions in Calgary?", "weather", SYSTEM_E),
    tool("third_w09", "Tell me the current temperature in Florence.", "weather", SYSTEM_F),
    tool("third_w10", "What is the weather doing in Halifax right now?", "weather", SYSTEM_E),

    # Calculator.
    tool("third_c01", "Calculate 584 plus 1327.", "calculator", SYSTEM_F),
    tool("third_c02", "What is 96 multiplied by 43?", "calculator", SYSTEM_E),
    tool("third_c03", "Compute 6720 divided by 21.", "calculator", SYSTEM_F),
    tool("third_c04", "Calculate 14 percent of 950.", "calculator", SYSTEM_E),
    tool("third_c05", "What is 52 squared?", "calculator", SYSTEM_F),
    tool("third_c06", "Compute 35 plus 67 plus 109.", "calculator", SYSTEM_E),
    tool("third_c07", "Calculate 2400 minus 958.", "calculator", SYSTEM_F),
    tool("third_c08", "What is 8.5 multiplied by 36?", "calculator", SYSTEM_E),
    tool("third_c09", "Compute 9216 divided by 72.", "calculator", SYSTEM_F),
    tool("third_c10", "Calculate 42 percent of 775.", "calculator", SYSTEM_E),

    # Search.
    tool("third_s01", "Find the newest stable Elixir release.", "web_search", SYSTEM_F),
    tool("third_s02", "What is the current stable Arch Linux release?", "web_search", SYSTEM_E),
    tool("third_s03", "Find the latest official Ansible release.", "web_search", SYSTEM_F),
    tool("third_s04", "Look up the newest stable RabbitMQ release.", "web_search", SYSTEM_E),
    tool("third_s05", "Find a recent official JPL announcement.", "web_search", SYSTEM_F),
    tool("third_s06", "What is the latest stable LLVM release?", "web_search", SYSTEM_E),
    tool("third_s07", "Find the current stable MongoDB release.", "web_search", SYSTEM_F),
    tool("third_s08", "Look up the latest official Krita release.", "web_search", SYSTEM_E),
    tool("third_s09", "Find the newest stable Brave release.", "web_search", SYSTEM_F),
    tool("third_s10", "What is the current stable Rocky Linux release?", "web_search", SYSTEM_E),

    # Time.
    tool("third_t01", "What time is it in Copenhagen right now?", "get_time", SYSTEM_F),
    tool("third_t02", "Give me the current local time in Hanoi.", "get_time", SYSTEM_E),
    tool("third_t03", "What is the time in Montevideo at this moment?", "get_time", SYSTEM_F),
    tool("third_t04", "Tell me the current time in Budapest.", "get_time", SYSTEM_E),
    tool("third_t05", "What time is it in Dubai right now?", "get_time", SYSTEM_F),
    tool("third_t06", "Give me the local time in Nagoya at the moment.", "get_time", SYSTEM_E),
    tool("third_t07", "What is the current time in Accra?", "get_time", SYSTEM_F),
    tool("third_t08", "Tell me what time it is in Geneva right now.", "get_time", SYSTEM_E),
    tool("third_t09", "Give me the current local time in Quito.", "get_time", SYSTEM_F),
    tool("third_t10", "What time is it in Bengaluru at this moment?", "get_time", SYSTEM_E),
]


def validate_third():
    if len(THIRD_CONFIRM) != 50:
        raise RuntimeError(f"third confirmation must have 50 cases, got {len(THIRD_CONFIRM)}")
    counts = {label: sum(probe.label_of(c) == label for c in THIRD_CONFIRM) for label in probe.LABELS}
    if any(counts[label] != 10 for label in probe.LABELS):
        raise RuntimeError(f"third confirmation not 10/class: {counts}")
    ids = [c["id"] for c in THIRD_CONFIRM]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate third confirmation ids")
    if set(ids) & ({c["id"] for c in held.CASES} | {c["id"] for c in confirm.CONFIRM}):
        raise RuntimeError("third confirmation overlaps prior test ids")
    return counts


def prepare_heads(rows):
    X, labels = hier.stack(rows, REPRESENTATION)
    binary_y = torch.tensor([0 if label == "direct" else 1 for label in labels], dtype=torch.long)
    if int((binary_y == 0).sum()) != 80 or int((binary_y == 1).sum()) != 80:
        raise RuntimeError("binary calibration set is not 80/80")
    binary_cv, binary_records = hier.cv_ridge(X, binary_y, 2)
    binary_state = hier.fit_ridge(X, binary_y, 2, binary_cv["ridge"])

    tool_indices = [i for i, label in enumerate(labels) if label != "direct"]
    X_tool = X[tool_indices]
    family_y = torch.tensor([hier.FAMILY_LABELS.index(labels[i]) for i in tool_indices], dtype=torch.long)
    for class_id in range(4):
        if int((family_y == class_id).sum()) != 20:
            raise RuntimeError("family calibration set is not 20/class")
    family_cv, family_records = hier.cv_ridge(X_tool, family_y, 4)
    family_state = hier.fit_ridge(X_tool, family_y, 4, family_cv["ridge"])
    return {
        "binary": binary_state,
        "family": family_state,
        "binary_cv": binary_cv,
        "binary_cv_records": binary_records,
        "family_cv": family_cv,
        "family_cv_records": family_records,
    }


def extract(model, tokenizer, cases):
    return hier.extract(model, tokenizer, cases)


def evaluate_set(heads, rows, cases):
    X, _ = hier.stack(rows, REPRESENTATION)
    predicted, margins = hier.hierarchical_predict(heads, X)
    return hier.score(predicted, rows, cases, margins)


def evaluate_all(heads, old_rows, second_rows, third_rows):
    return {
        "old_heldout": evaluate_set(heads, old_rows, held.CASES),
        "second_confirmation": evaluate_set(heads, second_rows, confirm.CONFIRM),
        "third_confirmation": evaluate_set(heads, third_rows, THIRD_CONFIRM),
    }


def exact_all(evaluation):
    for value in evaluation.values():
        if value["five_way_correct"] != value["total"]:
            return False
        if value["direct_vs_tool_correct"] != value["total"]:
            return False
        if value["tool_family_correct"] != value["tool_family_total"]:
            return False
    return True


def save_heads(path: Path, heads, precision: str):
    torch.save({
        "schema_version": 1,
        "kind": "ember-hierarchical-routing-head-calibrated",
        "source_model": held.MODEL_NAME,
        "source_checkpoint_sha256": held.BEST_SHA256 if precision == "full" else held.INT4_SHA256,
        "representation": REPRESENTATION,
        "binary_labels": ["direct", "tool"],
        "family_labels": hier.FAMILY_LABELS,
        "precision": precision,
        "training_counts": {"direct": 80, "tool": 80, "per_tool_family": 20},
        "binary": {
            "mean": heads["binary"]["mean"].float(),
            "std": heads["binary"]["std"].float(),
            "weight": heads["binary"]["weight"].float(),
            "ridge": heads["binary"]["ridge"],
        },
        "family": {
            "mean": heads["family"]["mean"].float(),
            "std": heads["family"]["std"].float(),
            "weight": heads["family"]["weight"].float(),
            "ridge": heads["family"]["ridge"],
        },
        "strict_three_set_confirmation": True,
    }, path)


def compact(m):
    return {
        "five_way": f"{m['five_way_correct']}/{m['total']}",
        "direct_tool": f"{m['direct_vs_tool_correct']}/{m['total']}",
        "tool_family": f"{m['tool_family_correct']}/{m['tool_family_total']}",
        "min_correct_margin": m["min_correct_margin"],
        "max_wrong_margin": m["max_wrong_margin"],
    }


def summary_markdown(report):
    lines = [
        "# Ember v0.0.54 calibrated hierarchical router",
        "",
        "Ember weights are frozen. Router stage 1 is balanced 80 direct / 80 tool; stage 2 has 20 examples per tool family.",
        "The third 50-case confirmation set was not used for fitting or hyperparameter selection.",
        "",
        "| Router | Old 20 | Second 50 | Third fresh 50 | Third direct/tool | Third tool-family |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("full", "int4", "full_heads_on_int4"):
        e = report["routers"][key]["evaluation"]
        old, second, third = e["old_heldout"], e["second_confirmation"], e["third_confirmation"]
        lines.append(
            f"| {key} | {old['five_way_correct']}/{old['total']} | {second['five_way_correct']}/{second['total']} | "
            f"{third['five_way_correct']}/{third['total']} | {third['direct_vs_tool_correct']}/{third['total']} | "
            f"{third['tool_family_correct']}/{third['tool_family_total']} |"
        )
    lines += [
        "",
        f"Strict full+INT4 three-set confirmation: **{report['strict_confirmation']}**",
        f"Shared full→INT4 three-set confirmation: **{report['shared_confirmation']}**",
        f"Interpretation: {report['interpretation']}",
        "",
        "No Ember checkpoint or production pointer was changed.",
    ]
    return "\n".join(lines) + "\n"


def main():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN required")
    torch.set_num_threads(2)
    torch.manual_seed(20260910)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    third_counts = validate_third()
    train_cases = training_cases()

    with tempfile.TemporaryDirectory(prefix="ember-router-calibration-") as td:
        work = Path(td)
        archive = ev.download_verified(ev.PACKAGE_URL, work / "ember.zip", ev.PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        import sys
        sys.path.insert(0, str(work / "src" / "ember"))

        api = HfApi(token=token)
        owner = api.whoami()["name"]
        repo = f"{owner}/{held.MODEL_NAME}"

        full_model, full_tok, _ = held.load_full(repo, work / "full", token)
        full_model.eval()
        full_train = extract(full_model, full_tok, train_cases)
        full_old = extract(full_model, full_tok, held.CASES)
        full_second = extract(full_model, full_tok, confirm.CONFIRM)
        full_third = extract(full_model, full_tok, THIRD_CONFIRM)
        full_heads = prepare_heads(full_train)
        full_eval = evaluate_all(full_heads, full_old, full_second, full_third)

        int4_model, int4_tok = held.load_int4(repo, work / "int4", token)
        int4_model.eval()
        int4_train = extract(int4_model, int4_tok, train_cases)
        int4_old = extract(int4_model, int4_tok, held.CASES)
        int4_second = extract(int4_model, int4_tok, confirm.CONFIRM)
        int4_third = extract(int4_model, int4_tok, THIRD_CONFIRM)
        int4_heads = prepare_heads(int4_train)
        int4_eval = evaluate_all(int4_heads, int4_old, int4_second, int4_third)

        cross_eval = evaluate_all(full_heads, int4_old, int4_second, int4_third)
        strict = exact_all(full_eval) and exact_all(int4_eval)
        shared = exact_all(cross_eval)

        if strict:
            save_heads(OUT / "router-full-calibrated.pt", full_heads, "full")
            save_heads(OUT / "router-int4-calibrated.pt", int4_heads, "int4")

        if strict and shared:
            interpretation = (
                "The calibrated hierarchical block-3 router achieved exact routing on all three evaluation sets in full and INT4, and one full router transfers unchanged to INT4. "
                "Next: run router-controlled generation with a single shared router while keeping Ember language weights frozen."
            )
        elif strict:
            interpretation = (
                "Precision-specific calibrated routers achieved exact routing on all three sets, but one shared full router did not. "
                "Next: run router-controlled generation with precision-specific router heads."
            )
        else:
            interpretation = (
                "The calibrated router still has misses on at least one evaluation set. Inspect only those remaining direct/tool boundary cases; do not modify Ember base weights."
            )

        routers = {
            "full": {
                "binary_cv": full_heads["binary_cv"],
                "family_cv": full_heads["family_cv"],
                "evaluation": full_eval,
            },
            "int4": {
                "binary_cv": int4_heads["binary_cv"],
                "family_cv": int4_heads["family_cv"],
                "evaluation": int4_eval,
            },
            "full_heads_on_int4": {
                "binary_cv": None,
                "family_cv": None,
                "evaluation": cross_eval,
            },
        }
        report = {
            "schema_version": 1,
            "diagnostic": "ember-v054-calibrated-hierarchical-router-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_repo": repo,
            "full_checkpoint_sha256": held.BEST_SHA256,
            "int4_checkpoint_sha256": held.INT4_SHA256,
            "representation": REPRESENTATION,
            "training_count": len(train_cases),
            "training_counts": {label: sum(probe.label_of(c) == label for c in train_cases) for label in probe.LABELS},
            "old_heldout_count": len(held.CASES),
            "second_confirmation_count": len(confirm.CONFIRM),
            "third_confirmation_count": len(THIRD_CONFIRM),
            "third_confirmation_counts": third_counts,
            "routers": routers,
            "strict_confirmation": strict,
            "shared_confirmation": shared,
            "router_artifacts_written": strict,
            "ember_weights_changed": False,
            "production_changed": False,
            "interpretation": interpretation,
            "elapsed_seconds": time.monotonic() - started,
        }
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (OUT / "summary.md").write_text(summary_markdown(report), encoding="utf-8")

        print(json.dumps({
            "event": "router_calibration_complete",
            "full_binary_cv": full_heads["binary_cv"],
            "full_family_cv": full_heads["family_cv"],
            "int4_binary_cv": int4_heads["binary_cv"],
            "int4_family_cv": int4_heads["family_cv"],
            "full_old": compact(full_eval["old_heldout"]),
            "full_second": compact(full_eval["second_confirmation"]),
            "full_third": compact(full_eval["third_confirmation"]),
            "int4_old": compact(int4_eval["old_heldout"]),
            "int4_second": compact(int4_eval["second_confirmation"]),
            "int4_third": compact(int4_eval["third_confirmation"]),
            "full_on_int4_old": compact(cross_eval["old_heldout"]),
            "full_on_int4_second": compact(cross_eval["second_confirmation"]),
            "full_on_int4_third": compact(cross_eval["third_confirmation"]),
            "strict_confirmation": strict,
            "shared_confirmation": shared,
            "interpretation": interpretation,
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
