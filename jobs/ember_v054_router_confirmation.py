"""Confirm a tiny frozen block-3 Ember router on a second fresh test set.

The prior frozen probe showed that a regularized linear decoder on block_03 of
the exact saved v0.0.53 step-9 checkpoint classified the unchanged 20-case
routing challenge 20/20. This diagnostic keeps that 20-case set untouched and
adds a SECOND fresh 50-prompt confirmation set (10/class):
  direct, weather, calculator, web_search, get_time.

The router is trained only on the same 80 balanced development prompts used by
the prior probe. Ridge strength is selected only by stratified CV on those 80
training prompts. The held-out sets never choose hyperparameters.

Both full precision and INT4 hidden states are tested with independently fit
router heads. The full-precision router is also applied unchanged to INT4 hidden
states to measure whether one shared head survives quantization.

Ember weights remain frozen. No model checkpoint is saved/promoted. If strict
confirmation passes, small router-head tensors are written only to the workflow
artifact for reproducibility; they are not integrated into Ember or production.
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
import torch.nn.functional as F

from jobs import ember_hf_eval as ev
from jobs import ember_v053_heldout_routing as held
from jobs import ember_v054_frozen_router_probe as probe

OUT = Path("v054-router-confirmation")
REPRESENTATION = "block_03"

SYSTEM_C = (
    "You are Ember. Choose either a normal reply or one tool: weather, calculator, web_search, get_time. "
    "Use weather only for live weather, calculator for requested arithmetic, web_search for up-to-date web facts, "
    "and get_time for the current local time. Writing or explaining about those topics does not by itself require a tool."
)
SYSTEM_D = (
    "You are Ember. Tools available are weather, calculator, web_search, and get_time. "
    "Route by the user's actual task. If no live data or arithmetic is needed, answer directly."
)


def prompt(system: str, user: str) -> str:
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"


def direct(case_id: str, user: str, system: str = SYSTEM_C) -> dict:
    return {"id": case_id, "kind": "direct_response", "user": user, "prompt": prompt(system, user)}


def tool(case_id: str, user: str, expected_tool: str, system: str = SYSTEM_C) -> dict:
    return {
        "id": case_id,
        "kind": "tool_call",
        "user": user,
        "prompt": prompt(system, user),
        "expected_tool": expected_tool,
    }


CONFIRM = [
    # Direct: intentionally tool-adjacent but no tool is actually required.
    direct("confirm_d01", "Rewrite this heading clearly: weather panel layout needs polish."),
    direct("confirm_d02", "Explain what a percentage means without calculating a specific example.", SYSTEM_D),
    direct("confirm_d03", "Summarize this: the current time widget moved to the header and the layout is cleaner."),
    direct("confirm_d04", "Define web search in one plain-English sentence without performing one.", SYSTEM_D),
    direct("confirm_d05", "Rewrite this as a professional title: latest build notes need cleanup."),
    direct("confirm_d06", "Give me two short steps for testing a password visibility button.", SYSTEM_D),
    direct("confirm_d07", "Label the sentiment: The update finally fixed the problem."),
    direct("confirm_d08", "Compare JSON and CSV in two short sentences.", SYSTEM_D),
    direct("confirm_d09", "Write a friendly sentence thanking someone for reviewing a pull request."),
    direct("confirm_d10", "Explain why time zones exist without telling me the current time anywhere.", SYSTEM_D),

    # Live weather, all cities absent from prior routing sets.
    tool("confirm_w01", "What is the weather in Boise right now?", "weather"),
    tool("confirm_w02", "Is it raining in New Orleans at the moment?", "weather", SYSTEM_D),
    tool("confirm_w03", "Give me the live temperature in Edinburgh.", "weather"),
    tool("confirm_w04", "How is the weather in Zurich right now?", "weather", SYSTEM_D),
    tool("confirm_w05", "Do I need an umbrella in Jakarta at the moment?", "weather"),
    tool("confirm_w06", "Check the current weather conditions in Sacramento.", "weather", SYSTEM_D),
    tool("confirm_w07", "Is it sunny in Quebec City right now?", "weather"),
    tool("confirm_w08", "What is the live weather in Melbourne?", "weather", SYSTEM_D),
    tool("confirm_w09", "Tell me the current temperature in Warsaw.", "weather"),
    tool("confirm_w10", "What is the weather doing in Anchorage right now?", "weather", SYSTEM_D),

    # Arithmetic.
    tool("confirm_c01", "Calculate 763 plus 1289.", "calculator"),
    tool("confirm_c02", "What is 84 multiplied by 57?", "calculator", SYSTEM_D),
    tool("confirm_c03", "Compute 9450 divided by 27.", "calculator"),
    tool("confirm_c04", "Calculate 12.5 percent of 736.", "calculator", SYSTEM_D),
    tool("confirm_c05", "What is 46 squared?", "calculator"),
    tool("confirm_c06", "Compute 17 plus 29 plus 63 plus 91.", "calculator", SYSTEM_D),
    tool("confirm_c07", "Calculate 1500 minus 687.", "calculator"),
    tool("confirm_c08", "What is 6.25 multiplied by 48?", "calculator", SYSTEM_D),
    tool("confirm_c09", "Compute 8192 divided by 64.", "calculator"),
    tool("confirm_c10", "Calculate 31 percent of 920.", "calculator", SYSTEM_D),

    # Current web facts.
    tool("confirm_s01", "Find the newest stable PHP release.", "web_search"),
    tool("confirm_s02", "What is the current stable Alpine Linux release?", "web_search", SYSTEM_D),
    tool("confirm_s03", "Find the latest official Kubernetes release.", "web_search"),
    tool("confirm_s04", "Look up the newest stable Redis version.", "web_search", SYSTEM_D),
    tool("confirm_s05", "Find a recent official ESA announcement.", "web_search"),
    tool("confirm_s06", "What is the latest stable OpenSSL release?", "web_search", SYSTEM_D),
    tool("confirm_s07", "Find the current stable MariaDB release.", "web_search"),
    tool("confirm_s08", "Look up the latest official Godot release.", "web_search", SYSTEM_D),
    tool("confirm_s09", "Find the newest stable Chromium release.", "web_search"),
    tool("confirm_s10", "What is the current stable Ubuntu release?", "web_search", SYSTEM_D),

    # Current local time.
    tool("confirm_t01", "What time is it in Stockholm right now?", "get_time"),
    tool("confirm_t02", "Give me the current local time in Kuala Lumpur.", "get_time", SYSTEM_D),
    tool("confirm_t03", "What is the time in Bogotá at this moment?", "get_time"),
    tool("confirm_t04", "Tell me the current time in Vienna.", "get_time", SYSTEM_D),
    tool("confirm_t05", "What time is it in Istanbul right now?", "get_time"),
    tool("confirm_t06", "Give me the local time in Osaka at the moment.", "get_time", SYSTEM_D),
    tool("confirm_t07", "What is the current time in Casablanca?", "get_time"),
    tool("confirm_t08", "Tell me what time it is in Brussels right now.", "get_time", SYSTEM_D),
    tool("confirm_t09", "Give me the current local time in Santiago.", "get_time"),
    tool("confirm_t10", "What time is it in Mumbai at this moment?", "get_time", SYSTEM_D),
]


def validate_confirmation(train_cases: list[dict]):
    if len(CONFIRM) != 50:
        raise RuntimeError(f"confirmation set must contain 50 cases, got {len(CONFIRM)}")
    counts = {label: 0 for label in probe.LABELS}
    for c in CONFIRM:
        counts[probe.label_of(c)] += 1
    if any(counts[label] != 10 for label in probe.LABELS):
        raise RuntimeError(f"confirmation set is not balanced 10/class: {counts}")
    train_ids = {c["id"] for c in train_cases}
    old_ids = {c["id"] for c in held.CASES}
    confirm_ids = {c["id"] for c in CONFIRM}
    if len(confirm_ids) != len(CONFIRM):
        raise RuntimeError("duplicate confirmation ids")
    if confirm_ids & train_ids or confirm_ids & old_ids:
        raise RuntimeError("confirmation ids overlap training or prior held-out ids")
    return counts


def stack_rep(rows, representation: str):
    X = torch.stack([r["features"][representation] for r in rows]).double()
    y = torch.tensor([probe.LABEL_TO_ID[r["label"]] for r in rows], dtype=torch.long)
    return X, y


def fit_state(X_train, y_train, ridge: float):
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    X = (X_train - mean) / std
    X = torch.cat([X, torch.ones((X.shape[0], 1), dtype=X.dtype)], dim=1)
    Y = F.one_hot(y_train, num_classes=len(probe.LABELS)).double()
    K = X @ X.T + float(ridge) * torch.eye(X.shape[0], dtype=X.dtype)
    alpha = torch.linalg.solve(K, Y)
    W = X.T @ alpha
    return {"mean": mean, "std": std, "weight": W, "ridge": float(ridge)}


def predict_state(state, X):
    Z = (X - state["mean"]) / state["std"]
    Z = torch.cat([Z, torch.ones((Z.shape[0], 1), dtype=Z.dtype)], dim=1)
    logits = Z @ state["weight"]
    pred = torch.argmax(logits, dim=1)
    top2 = torch.topk(logits, k=2, dim=1).values
    margins = top2[:, 0] - top2[:, 1]
    return pred, logits, margins


def metrics(pred, truth, cases, margins):
    base = probe.classify_metrics(pred, truth)
    for row, case, margin in zip(base["rows"], cases, margins.tolist()):
        row["id"] = case["id"]
        row["margin"] = float(margin)
    correct_margins = [r["margin"] for r in base["rows"] if r["correct"]]
    wrong_margins = [r["margin"] for r in base["rows"] if not r["correct"]]
    base["min_correct_margin"] = min(correct_margins) if correct_margins else None
    base["mean_correct_margin"] = sum(correct_margins) / len(correct_margins) if correct_margins else None
    base["max_wrong_margin"] = max(wrong_margins) if wrong_margins else None
    return base


def extract_for(model, tokenizer, cases):
    rows, reps, block_modules, head = probe.extract_representations(model, tokenizer, cases)
    if REPRESENTATION not in reps:
        raise RuntimeError(f"{REPRESENTATION} not captured; available={reps}")
    return rows


def evaluate_head(state, train_rows, old_rows, confirm_rows, name: str):
    X_train, y_train = stack_rep(train_rows, REPRESENTATION)
    X_old, y_old = stack_rep(old_rows, REPRESENTATION)
    X_confirm, y_confirm = stack_rep(confirm_rows, REPRESENTATION)
    old_pred, _, old_margin = predict_state(state, X_old)
    confirm_pred, _, confirm_margin = predict_state(state, X_confirm)
    return {
        "name": name,
        "old_heldout": metrics(old_pred, y_old, held.CASES, old_margin),
        "confirmation": metrics(confirm_pred, y_confirm, CONFIRM, confirm_margin),
    }


def compact(m):
    return {
        "five_way": f"{m['five_way_correct']}/{m['total']}",
        "direct_tool": f"{m['direct_vs_tool_correct']}/{m['total']}",
        "tool_family": f"{m['tool_family_correct']}/{m['tool_family_total']}",
        "min_correct_margin": m["min_correct_margin"],
    }


def summary_markdown(report):
    lines = [
        "# Ember v0.0.54 block-3 router confirmation",
        "",
        "Ember language weights are frozen.",
        "Router training: 80 balanced development prompts. Hyperparameters chosen by training-only CV.",
        "Tests: unchanged 20-case held-out plus a second fresh balanced 50-case confirmation set.",
        "",
        "| Router | CV | Old held-out | Fresh confirmation | Direct/tool fresh | Tool-family fresh |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("full", "int4", "full_head_on_int4"):
        row = report["routers"][key]
        cv = row.get("cv_best")
        cv_text = "—" if cv is None else f"{cv['accuracy']:.1%}"
        old = row["evaluation"]["old_heldout"]
        new = row["evaluation"]["confirmation"]
        lines.append(
            f"| {key} | {cv_text} | {old['five_way_correct']}/{old['total']} | "
            f"{new['five_way_correct']}/{new['total']} | {new['direct_vs_tool_correct']}/{new['total']} | "
            f"{new['tool_family_correct']}/{new['tool_family_total']} |"
        )
    lines += [
        "",
        f"Strict full+INT4 router confirmation: **{report['strict_router_confirmation']}**",
        f"One shared full→INT4 head: **{report['shared_head_confirmation']}**",
        f"Interpretation: {report['interpretation']}",
        "",
        "Router-head tensors are artifact-only. No Ember checkpoint or production pointer was changed.",
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

    train_cases = probe.balanced_training()
    confirmation_counts = validate_confirmation(train_cases)

    with tempfile.TemporaryDirectory(prefix="ember-router-confirm-") as td:
        work = Path(td)
        archive = ev.download_verified(ev.PACKAGE_URL, work / "ember.zip", ev.PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        import sys
        sys.path.insert(0, str(work / "src" / "ember"))

        api = HfApi(token=token)
        owner = api.whoami()["name"]
        repo = f"{owner}/{held.MODEL_NAME}"

        full_model, full_tok, full_checkpoint = held.load_full(repo, work / "full-load", token)
        full_model.eval()
        full_train = extract_for(full_model, full_tok, train_cases)
        full_old = extract_for(full_model, full_tok, held.CASES)
        full_confirm = extract_for(full_model, full_tok, CONFIRM)
        X_full, y_full = stack_rep(full_train, REPRESENTATION)
        full_cv, full_cv_records = probe.choose_ridge(X_full, y_full)
        full_state = fit_state(X_full, y_full, full_cv["ridge"])
        full_eval = evaluate_head(full_state, full_train, full_old, full_confirm, "full")

        int4_model, int4_tok = held.load_int4(repo, work / "int4-load", token)
        int4_model.eval()
        int4_train = extract_for(int4_model, int4_tok, train_cases)
        int4_old = extract_for(int4_model, int4_tok, held.CASES)
        int4_confirm = extract_for(int4_model, int4_tok, CONFIRM)
        X_int4, y_int4 = stack_rep(int4_train, REPRESENTATION)
        int4_cv, int4_cv_records = probe.choose_ridge(X_int4, y_int4)
        int4_state = fit_state(X_int4, y_int4, int4_cv["ridge"])
        int4_eval = evaluate_head(int4_state, int4_train, int4_old, int4_confirm, "int4")

        # Cross-quantization portability: use full router state directly on INT4 block-3 features.
        X_i_old, y_i_old = stack_rep(int4_old, REPRESENTATION)
        X_i_confirm, y_i_confirm = stack_rep(int4_confirm, REPRESENTATION)
        p_old, _, m_old = predict_state(full_state, X_i_old)
        p_new, _, m_new = predict_state(full_state, X_i_confirm)
        cross_eval = {
            "name": "full_head_on_int4",
            "old_heldout": metrics(p_old, y_i_old, held.CASES, m_old),
            "confirmation": metrics(p_new, y_i_confirm, CONFIRM, m_new),
        }

        def exact(eval_row):
            a = eval_row["old_heldout"]
            b = eval_row["confirmation"]
            return (
                a["five_way_correct"] == a["total"]
                and b["five_way_correct"] == b["total"]
                and b["direct_vs_tool_correct"] == b["total"]
                and b["tool_family_correct"] == b["tool_family_total"]
            )

        strict = exact(full_eval) and exact(int4_eval)
        shared = exact(cross_eval)

        routers = {
            "full": {
                "representation": REPRESENTATION,
                "cv_best": full_cv,
                "cv_records": full_cv_records,
                "evaluation": full_eval,
            },
            "int4": {
                "representation": REPRESENTATION,
                "cv_best": int4_cv,
                "cv_records": int4_cv_records,
                "evaluation": int4_eval,
            },
            "full_head_on_int4": {
                "representation": REPRESENTATION,
                "cv_best": None,
                "evaluation": cross_eval,
            },
        }

        if strict:
            torch.save({
                "schema_version": 1,
                "kind": "ember-routing-head",
                "source_model": held.MODEL_NAME,
                "source_checkpoint_sha256": held.BEST_SHA256,
                "representation": REPRESENTATION,
                "labels": probe.LABELS,
                "mean": full_state["mean"].float(),
                "std": full_state["std"].float(),
                "weight": full_state["weight"].float(),
                "ridge": full_state["ridge"],
                "strict_confirmation": True,
                "intended_precision": "full",
            }, OUT / "router-full.pt")
            torch.save({
                "schema_version": 1,
                "kind": "ember-routing-head",
                "source_model": held.MODEL_NAME,
                "source_checkpoint_sha256": held.INT4_SHA256,
                "representation": REPRESENTATION,
                "labels": probe.LABELS,
                "mean": int4_state["mean"].float(),
                "std": int4_state["std"].float(),
                "weight": int4_state["weight"].float(),
                "ridge": int4_state["ridge"],
                "strict_confirmation": True,
                "intended_precision": "int4",
            }, OUT / "router-int4.pt")

        if strict and shared:
            interpretation = (
                "The block-3 routing signal generalizes perfectly on both held-out sets in full and INT4, and one full-precision router head also transfers perfectly to INT4. "
                "Next: test router-controlled generation while keeping Ember weights frozen; a single shared routing head is sufficient."
            )
        elif strict:
            interpretation = (
                "Separate full and INT4 block-3 router heads generalize perfectly, but the full head is not perfectly portable to INT4. "
                "Next: test router-controlled generation with precision-specific heads while keeping Ember weights frozen."
            )
        else:
            interpretation = (
                "The frozen block-3 router did not achieve exact confirmation on both precisions. Inspect the few misses and broaden/calibrate the router dataset before any integration."
            )

        report = {
            "schema_version": 1,
            "diagnostic": "ember-v054-router-second-confirmation-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_repo": repo,
            "full_checkpoint": held.BEST_PATH,
            "full_checkpoint_sha256": held.BEST_SHA256,
            "int4_checkpoint": held.INT4_PATH,
            "int4_checkpoint_sha256": held.INT4_SHA256,
            "representation": REPRESENTATION,
            "labels": probe.LABELS,
            "training_count": len(train_cases),
            "training_counts": {label: sum(probe.label_of(c) == label for c in train_cases) for label in probe.LABELS},
            "old_heldout_count": len(held.CASES),
            "confirmation_count": len(CONFIRM),
            "confirmation_counts": confirmation_counts,
            "routers": routers,
            "strict_router_confirmation": strict,
            "shared_head_confirmation": shared,
            "router_artifacts_written": strict,
            "ember_weights_changed": False,
            "production_changed": False,
            "interpretation": interpretation,
            "elapsed_seconds": time.monotonic() - started,
        }
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (OUT / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
        print(json.dumps({
            "event": "router_confirmation_complete",
            "full_cv": full_cv,
            "int4_cv": int4_cv,
            "full_old": compact(full_eval["old_heldout"]),
            "full_confirm": compact(full_eval["confirmation"]),
            "int4_old": compact(int4_eval["old_heldout"]),
            "int4_confirm": compact(int4_eval["confirmation"]),
            "full_head_on_int4_old": compact(cross_eval["old_heldout"]),
            "full_head_on_int4_confirm": compact(cross_eval["confirmation"]),
            "strict_router_confirmation": strict,
            "shared_head_confirmation": shared,
            "interpretation": interpretation,
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
