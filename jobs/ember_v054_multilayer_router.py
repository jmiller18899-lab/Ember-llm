"""Training-selected multi-layer frozen router for Ember v0.0.54.

The calibrated block-3 hierarchical router is near-perfect but left one distinct
fresh miss in full precision and one in INT4. Rather than tune to those prompts,
this diagnostic lets TRAINING-ONLY cross-validation select the hidden
representation independently for:
  stage 1: direct vs tool;
  stage 2: weather vs calculator vs web_search vs get_time.

Candidate representations are fixed before evaluation and use middle-layer
states: block_02, block_03, block_04, and concatenations thereof. Hyperparameter
and representation selection never sees any held-out set.

The router is trained on the existing calibrated 160 prompts. It is then tested
on all three previous evaluation sets plus a FOURTH fresh balanced 50-case set.
Full and INT4 get precision-specific heads; the selected full heads are also
applied unchanged to INT4 features as a portability check.

Ember weights remain frozen. No Ember checkpoint or production pointer changes.
Router tensors are artifact-only if the precision-specific strict gate passes.
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
from jobs import ember_v054_router_calibration as cal
from jobs import ember_v054_hierarchical_router as hier

OUT = Path("v054-multilayer-router")
REPRESENTATIONS = {
    "block_02": ("block_02",),
    "block_03": ("block_03",),
    "block_04": ("block_04",),
    "block_02+03": ("block_02", "block_03"),
    "block_03+04": ("block_03", "block_04"),
    "block_02+03+04": ("block_02", "block_03", "block_04"),
}

SYSTEM_G = (
    "You are Ember. Route each request to a direct reply or exactly one tool: weather, calculator, web_search, get_time. "
    "Use a tool only when live weather, explicit arithmetic, current web information, or current local time is needed. "
    "Discussion, writing, classification, and explanation about those topics are direct replies."
)
SYSTEM_H = (
    "You are Ember. Available tools are weather, calculator, web_search, and get_time. "
    "Choose based on the task itself, not keywords. If the user does not require live information or arithmetic, answer directly."
)


def make_prompt(system: str, user: str) -> str:
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"


def direct(cid: str, user: str, system: str = SYSTEM_G) -> dict:
    return {"id": cid, "kind": "direct_response", "user": user, "prompt": make_prompt(system, user)}


def tool(cid: str, user: str, expected_tool: str, system: str = SYSTEM_G) -> dict:
    return {"id": cid, "kind": "tool_call", "user": user, "prompt": make_prompt(system, user), "expected_tool": expected_tool}


FOURTH_CONFIRM = [
    # Direct cases: tool-adjacent wording without a live-data/arithmetic need.
    direct("fourth_d01", "Explain why a painting can depict stormy weather without checking any forecast.", SYSTEM_H),
    direct("fourth_d02", "Label the sentiment: The newest design is clean and pleasant to use.", SYSTEM_G),
    direct("fourth_d03", "Rewrite this heading: current clock settings help text.", SYSTEM_H),
    direct("fourth_d04", "Explain what a square root is without calculating a specific value.", SYSTEM_G),
    direct("fourth_d05", "Summarize: the search box moved to the top and no network request changed.", SYSTEM_H),
    direct("fourth_d06", "Classify this as a request or statement: The weather card is visible.", SYSTEM_G),
    direct("fourth_d07", "Compare a calendar and a clock in two short sentences.", SYSTEM_H),
    direct("fourth_d08", "Write a friendly sentence saying the latest draft is ready for review.", SYSTEM_G),
    direct("fourth_d09", "Define a search query without performing one.", SYSTEM_H),
    direct("fourth_d10", "Give two steps for testing a checkbox on a settings form.", SYSTEM_G),

    # Weather.
    tool("fourth_w01", "What is the weather in Fargo right now?", "weather", SYSTEM_H),
    tool("fourth_w02", "Is it raining in Cork at the moment?", "weather", SYSTEM_G),
    tool("fourth_w03", "Give me the live temperature in Sendai.", "weather", SYSTEM_H),
    tool("fourth_w04", "How is the weather in Seville right now?", "weather", SYSTEM_G),
    tool("fourth_w05", "Do I need an umbrella in Christchurch at the moment?", "weather", SYSTEM_H),
    tool("fourth_w06", "Check the current weather in Cleveland.", "weather", SYSTEM_G),
    tool("fourth_w07", "Is it sunny in Marseille right now?", "weather", SYSTEM_H),
    tool("fourth_w08", "What are the live weather conditions in Edmonton?", "weather", SYSTEM_G),
    tool("fourth_w09", "Tell me the current temperature in Naples.", "weather", SYSTEM_H),
    tool("fourth_w10", "What is the weather doing in St. John's right now?", "weather", SYSTEM_G),

    # Calculator.
    tool("fourth_c01", "Calculate 692 plus 1448.", "calculator", SYSTEM_H),
    tool("fourth_c02", "What is 87 multiplied by 62?", "calculator", SYSTEM_G),
    tool("fourth_c03", "Compute 7560 divided by 24.", "calculator", SYSTEM_H),
    tool("fourth_c04", "Calculate 16 percent of 875.", "calculator", SYSTEM_G),
    tool("fourth_c05", "What is 58 squared?", "calculator", SYSTEM_H),
    tool("fourth_c06", "Compute 48 plus 76 plus 125.", "calculator", SYSTEM_G),
    tool("fourth_c07", "Calculate 3100 minus 1276.", "calculator", SYSTEM_H),
    tool("fourth_c08", "What is 9.25 multiplied by 28?", "calculator", SYSTEM_G),
    tool("fourth_c09", "Compute 10752 divided by 84.", "calculator", SYSTEM_H),
    tool("fourth_c10", "Calculate 36 percent of 825.", "calculator", SYSTEM_G),

    # Current web facts.
    tool("fourth_s01", "Find the newest stable Scala release.", "web_search", SYSTEM_H),
    tool("fourth_s02", "What is the current stable openSUSE release?", "web_search", SYSTEM_G),
    tool("fourth_s03", "Find the latest official Helm release.", "web_search", SYSTEM_H),
    tool("fourth_s04", "Look up the newest stable NATS release.", "web_search", SYSTEM_G),
    tool("fourth_s05", "Find a recent official USGS announcement.", "web_search", SYSTEM_H),
    tool("fourth_s06", "What is the latest stable GCC release?", "web_search", SYSTEM_G),
    tool("fourth_s07", "Find the current stable CouchDB release.", "web_search", SYSTEM_H),
    tool("fourth_s08", "Look up the latest official Inkscape release.", "web_search", SYSTEM_G),
    tool("fourth_s09", "Find the newest stable Vivaldi release.", "web_search", SYSTEM_H),
    tool("fourth_s10", "What is the current stable AlmaLinux release?", "web_search", SYSTEM_G),

    # Current local time.
    tool("fourth_t01", "What time is it in Oslo right now?", "get_time", SYSTEM_H),
    tool("fourth_t02", "Give me the current local time in Ho Chi Minh City.", "get_time", SYSTEM_G),
    tool("fourth_t03", "What is the time in Asunción at this moment?", "get_time", SYSTEM_H),
    tool("fourth_t04", "Tell me the current time in Bucharest.", "get_time", SYSTEM_G),
    tool("fourth_t05", "What time is it in Doha right now?", "get_time", SYSTEM_H),
    tool("fourth_t06", "Give me the local time in Fukuoka at the moment.", "get_time", SYSTEM_G),
    tool("fourth_t07", "What is the current time in Dakar?", "get_time", SYSTEM_H),
    tool("fourth_t08", "Tell me what time it is in Luxembourg right now.", "get_time", SYSTEM_G),
    tool("fourth_t09", "Give me the current local time in La Paz.", "get_time", SYSTEM_H),
    tool("fourth_t10", "What time is it in Chennai at this moment?", "get_time", SYSTEM_G),
]


def validate_fourth(train_cases):
    if len(FOURTH_CONFIRM) != 50:
        raise RuntimeError(f"fourth confirmation must be 50 cases, got {len(FOURTH_CONFIRM)}")
    counts = {label: sum(probe.label_of(c) == label for c in FOURTH_CONFIRM) for label in probe.LABELS}
    if any(counts[label] != 10 for label in probe.LABELS):
        raise RuntimeError(f"fourth confirmation not balanced 10/class: {counts}")
    ids = [c["id"] for c in FOURTH_CONFIRM]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate fourth confirmation ids")
    forbidden = (
        {c["id"] for c in train_cases}
        | {c["id"] for c in held.CASES}
        | {c["id"] for c in confirm.CONFIRM}
        | {c["id"] for c in cal.THIRD_CONFIRM}
    )
    overlap = set(ids) & forbidden
    if overlap:
        raise RuntimeError(f"fourth confirmation overlaps earlier sets: {sorted(overlap)}")
    return counts


def extract(model, tokenizer, cases):
    rows, reps, _blocks, _head = probe.extract_representations(model, tokenizer, cases)
    required = {part for parts in REPRESENTATIONS.values() for part in parts}
    missing = required.difference(reps)
    if missing:
        raise RuntimeError(f"required representations missing: {sorted(missing)} available={reps}")
    return rows


def matrix(rows, rep_name: str):
    parts = REPRESENTATIONS[rep_name]
    X = torch.stack([
        torch.cat([row["features"][part] for part in parts], dim=0)
        for row in rows
    ]).double()
    labels = [row["label"] for row in rows]
    return X, labels


def select_stage(rows, stage: str):
    candidates = []
    for rep_name, parts in REPRESENTATIONS.items():
        X, labels = matrix(rows, rep_name)
        if stage == "binary":
            y = torch.tensor([0 if label == "direct" else 1 for label in labels], dtype=torch.long)
            num_classes = 2
            X_stage = X
        elif stage == "family":
            idx = [i for i, label in enumerate(labels) if label != "direct"]
            X_stage = X[idx]
            y = torch.tensor([hier.FAMILY_LABELS.index(labels[i]) for i in idx], dtype=torch.long)
            num_classes = 4
        else:
            raise ValueError(stage)
        cv_best, cv_records = hier.cv_ridge(X_stage, y, num_classes)
        candidates.append({
            "representation": rep_name,
            "parts": list(parts),
            "dimension": int(X_stage.shape[1]),
            "cv_best": cv_best,
            "cv_records": cv_records,
        })
    # Selection uses TRAINING CV only. Prefer higher CV accuracy, then lower dimension, then stronger regularization.
    candidates.sort(
        key=lambda r: (r["cv_best"]["accuracy"], -r["dimension"], r["cv_best"]["ridge"]),
        reverse=True,
    )
    selected = candidates[0]
    X, labels = matrix(rows, selected["representation"])
    if stage == "binary":
        y = torch.tensor([0 if label == "direct" else 1 for label in labels], dtype=torch.long)
        X_stage = X
        num_classes = 2
    else:
        idx = [i for i, label in enumerate(labels) if label != "direct"]
        X_stage = X[idx]
        y = torch.tensor([hier.FAMILY_LABELS.index(labels[i]) for i in idx], dtype=torch.long)
        num_classes = 4
    state = hier.fit_ridge(X_stage, y, num_classes, selected["cv_best"]["ridge"])
    state["representation"] = selected["representation"]
    state["parts"] = selected["parts"]
    return state, selected, candidates


def prepare_heads(rows):
    binary, binary_selected, binary_candidates = select_stage(rows, "binary")
    family, family_selected, family_candidates = select_stage(rows, "family")
    return {
        "binary": binary,
        "family": family,
        "binary_selected": binary_selected,
        "family_selected": family_selected,
        "binary_candidates": binary_candidates,
        "family_candidates": family_candidates,
    }


def predict_state_on_rows(state, rows):
    X, _labels = matrix(rows, state["representation"])
    return hier.predict(state, X)


def hierarchical_predict(heads, rows):
    binary_pred, _blogits, binary_margin = predict_state_on_rows(heads["binary"], rows)
    family_pred, _flogits, family_margin = predict_state_on_rows(heads["family"], rows)
    predicted = []
    margins = []
    for i in range(len(rows)):
        if int(binary_pred[i]) == 0:
            predicted.append("direct")
            margins.append(float(binary_margin[i]))
        else:
            predicted.append(hier.FAMILY_LABELS[int(family_pred[i])])
            margins.append(float(min(binary_margin[i], family_margin[i])))
    return predicted, margins


def evaluate_set(heads, rows, cases):
    pred, margins = hierarchical_predict(heads, rows)
    return hier.score(pred, rows, cases, margins)


def evaluate_all(heads, old_rows, second_rows, third_rows, fourth_rows):
    return {
        "old_heldout": evaluate_set(heads, old_rows, held.CASES),
        "second_confirmation": evaluate_set(heads, second_rows, confirm.CONFIRM),
        "third_confirmation": evaluate_set(heads, third_rows, cal.THIRD_CONFIRM),
        "fourth_confirmation": evaluate_set(heads, fourth_rows, FOURTH_CONFIRM),
    }


def exact_all(evaluation):
    for metrics in evaluation.values():
        if metrics["five_way_correct"] != metrics["total"]:
            return False
        if metrics["direct_vs_tool_correct"] != metrics["total"]:
            return False
        if metrics["tool_family_correct"] != metrics["tool_family_total"]:
            return False
    return True


def save_heads(path: Path, heads, precision: str):
    torch.save({
        "schema_version": 1,
        "kind": "ember-multilayer-hierarchical-routing-head",
        "source_model": held.MODEL_NAME,
        "source_checkpoint_sha256": held.BEST_SHA256 if precision == "full" else held.INT4_SHA256,
        "binary_labels": ["direct", "tool"],
        "family_labels": hier.FAMILY_LABELS,
        "precision": precision,
        "selection_rule": "training-only-stratified-CV",
        "binary": {
            "representation": heads["binary"]["representation"],
            "parts": heads["binary"]["parts"],
            "mean": heads["binary"]["mean"].float(),
            "std": heads["binary"]["std"].float(),
            "weight": heads["binary"]["weight"].float(),
            "ridge": heads["binary"]["ridge"],
        },
        "family": {
            "representation": heads["family"]["representation"],
            "parts": heads["family"]["parts"],
            "mean": heads["family"]["mean"].float(),
            "std": heads["family"]["std"].float(),
            "weight": heads["family"]["weight"].float(),
            "ridge": heads["family"]["ridge"],
        },
        "strict_four_set_confirmation": True,
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
        "# Ember v0.0.54 training-selected multi-layer router",
        "",
        "Ember weights are frozen. Binary/family representations are selected using training CV only.",
        f"Candidates: {', '.join(REPRESENTATIONS)}.",
        "The fourth 50-case set is fresh and was not used for fitting or representation selection.",
        "",
        "| Router | Binary rep | Family rep | Old20 | Second50 | Third50 | Fourth50 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for key in ("full", "int4", "full_heads_on_int4"):
        row = report["routers"][key]
        e = row["evaluation"]
        lines.append(
            f"| {key} | {row['binary_representation']} | {row['family_representation']} | "
            f"{e['old_heldout']['five_way_correct']}/{e['old_heldout']['total']} | "
            f"{e['second_confirmation']['five_way_correct']}/{e['second_confirmation']['total']} | "
            f"{e['third_confirmation']['five_way_correct']}/{e['third_confirmation']['total']} | "
            f"{e['fourth_confirmation']['five_way_correct']}/{e['fourth_confirmation']['total']} |"
        )
    lines += [
        "",
        f"Strict precision-specific four-set confirmation: **{report['strict_confirmation']}**",
        f"Shared full→INT4 four-set confirmation: **{report['shared_confirmation']}**",
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

    train_cases = cal.training_cases()
    fourth_counts = validate_fourth(train_cases)

    with tempfile.TemporaryDirectory(prefix="ember-multilayer-router-") as td:
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
        full_third = extract(full_model, full_tok, cal.THIRD_CONFIRM)
        full_fourth = extract(full_model, full_tok, FOURTH_CONFIRM)
        full_heads = prepare_heads(full_train)
        full_eval = evaluate_all(full_heads, full_old, full_second, full_third, full_fourth)

        int4_model, int4_tok = held.load_int4(repo, work / "int4", token)
        int4_model.eval()
        int4_train = extract(int4_model, int4_tok, train_cases)
        int4_old = extract(int4_model, int4_tok, held.CASES)
        int4_second = extract(int4_model, int4_tok, confirm.CONFIRM)
        int4_third = extract(int4_model, int4_tok, cal.THIRD_CONFIRM)
        int4_fourth = extract(int4_model, int4_tok, FOURTH_CONFIRM)
        int4_heads = prepare_heads(int4_train)
        int4_eval = evaluate_all(int4_heads, int4_old, int4_second, int4_third, int4_fourth)

        cross_eval = evaluate_all(full_heads, int4_old, int4_second, int4_third, int4_fourth)
        strict = exact_all(full_eval) and exact_all(int4_eval)
        shared = exact_all(cross_eval)

        if strict:
            save_heads(OUT / "router-full-multilayer.pt", full_heads, "full")
            save_heads(OUT / "router-int4-multilayer.pt", int4_heads, "int4")

        if strict and shared:
            interpretation = (
                "Training-selected multi-layer hierarchical routing is exact on all four evaluation sets in full and INT4, and the full router transfers unchanged to INT4. "
                "Next: run router-controlled generation with a single shared router while keeping Ember frozen."
            )
        elif strict:
            interpretation = (
                "Precision-specific multi-layer hierarchical routers are exact on all four evaluation sets; cross-precision sharing is imperfect. "
                "Next: run router-controlled generation with precision-specific router heads while keeping Ember frozen."
            )
        else:
            interpretation = (
                "Even after training-only representation selection, at least one precision has a routing miss. Inspect the fresh fourth-set misses before changing the router corpus; do not modify Ember weights."
            )

        def router_record(heads, evaluation):
            return {
                "binary_representation": heads["binary"]["representation"],
                "family_representation": heads["family"]["representation"],
                "binary_selected": heads["binary_selected"],
                "family_selected": heads["family_selected"],
                "binary_candidates": heads["binary_candidates"],
                "family_candidates": heads["family_candidates"],
                "evaluation": evaluation,
            }

        routers = {
            "full": router_record(full_heads, full_eval),
            "int4": router_record(int4_heads, int4_eval),
            "full_heads_on_int4": {
                "binary_representation": full_heads["binary"]["representation"],
                "family_representation": full_heads["family"]["representation"],
                "binary_selected": full_heads["binary_selected"],
                "family_selected": full_heads["family_selected"],
                "evaluation": cross_eval,
            },
        }
        report = {
            "schema_version": 1,
            "diagnostic": "ember-v054-training-selected-multilayer-router-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_repo": repo,
            "full_checkpoint_sha256": held.BEST_SHA256,
            "int4_checkpoint_sha256": held.INT4_SHA256,
            "representation_candidates": {k: list(v) for k, v in REPRESENTATIONS.items()},
            "training_count": len(train_cases),
            "training_counts": {label: sum(probe.label_of(c) == label for c in train_cases) for label in probe.LABELS},
            "fourth_confirmation_count": len(FOURTH_CONFIRM),
            "fourth_confirmation_counts": fourth_counts,
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
            "event": "multilayer_router_complete",
            "full_binary_rep": full_heads["binary"]["representation"],
            "full_family_rep": full_heads["family"]["representation"],
            "int4_binary_rep": int4_heads["binary"]["representation"],
            "int4_family_rep": int4_heads["family"]["representation"],
            "full_binary_cv": full_heads["binary_selected"]["cv_best"],
            "full_family_cv": full_heads["family_selected"]["cv_best"],
            "int4_binary_cv": int4_heads["binary_selected"]["cv_best"],
            "int4_family_cv": int4_heads["family_selected"]["cv_best"],
            "full_old": compact(full_eval["old_heldout"]),
            "full_second": compact(full_eval["second_confirmation"]),
            "full_third": compact(full_eval["third_confirmation"]),
            "full_fourth": compact(full_eval["fourth_confirmation"]),
            "int4_old": compact(int4_eval["old_heldout"]),
            "int4_second": compact(int4_eval["second_confirmation"]),
            "int4_third": compact(int4_eval["third_confirmation"]),
            "int4_fourth": compact(int4_eval["fourth_confirmation"]),
            "full_on_int4_fourth": compact(cross_eval["fourth_confirmation"]),
            "strict_confirmation": strict,
            "shared_confirmation": shared,
            "interpretation": interpretation,
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
