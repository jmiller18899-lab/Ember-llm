"""Hierarchical frozen block-3 router for Ember v0.0.54.

The one-shot 5-way block-3 router was perfect on the original 20-case held-out
set and 48/50 on a second fresh set. Every miss was a DIRECT prompt classified
as a tool; tool-family selection itself was 40/40. This diagnostic therefore
splits routing into two frozen-head decisions:

  stage 1: DIRECT vs TOOL, trained on 64 direct + 64 tool prompts;
  stage 2: weather vs calculator vs web_search vs get_time, trained only on the
           64 tool prompts (16 per family).

The 48 added direct training prompts are distinct from both prior held-out sets.
Hyperparameters are selected only by stratified CV on training data. The exact
saved Ember v0.0.53 step-9 weights remain frozen. Both full precision and INT4
are evaluated, plus one full-precision router applied unchanged to INT4 hidden
states.

No Ember checkpoint, model weights, or production pointer is changed. Router
head tensors are written only to the workflow artifact if strict confirmation
passes.
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
from jobs import ember_v054_router_confirmation as confirm
from jobs import ember_v054_routing_repair as v54

OUT = Path("v054-hierarchical-router")
REPRESENTATION = "block_03"
RIDGES = (0.01, 0.1, 1.0, 10.0, 100.0)
FOLDS = 4
FAMILY_LABELS = ["weather", "calculator", "web_search", "get_time"]


def direct(case_id: str, user: str, system: str) -> dict:
    return {
        "id": case_id,
        "kind": "direct_response",
        "user": user,
        "prompt": v54.make_prompt(system, user),
    }


def extra_directs() -> list[dict]:
    """48 fresh direct prompts used only for router training."""
    s1, s2, s3 = v54.SYSTEM_1, v54.SYSTEM_2, v54.SYSTEM_3
    rows = [
        # Tool-adjacent rewrites / editing tasks.
        ("hd01", "Rewrite this title clearly: weather dashboard spacing needs work.", s2),
        ("hd02", "Shorten this heading: current time display configuration options.", s3),
        ("hd03", "Rewrite this sentence professionally: calculator page looks kind of messy.", s1),
        ("hd04", "Turn this into a clear section title: latest search results layout ideas.", s2),
        ("hd05", "Make this friendlier: weather data is unavailable in the mockup.", s3),
        ("hd06", "Rewrite this bug title: time zone selector opens the wrong panel.", s1),
        ("hd07", "Polish this sentence: web search button needs a better label.", s2),
        ("hd08", "Shorten this title: calculator keyboard accessibility improvements.", s3),
        ("hd09", "Rewrite this heading: current release notes formatting cleanup.", s1),
        ("hd10", "Make this message clearer: latest update text is too long.", s2),
        ("hd11", "Rewrite as a professional title: weather icon alignment bug.", s3),
        ("hd12", "Polish this copy: search page empty-state wording needs improvement.", s1),

        # Concepts containing tool keywords but requiring no live call.
        ("hd13", "Explain what an umbrella forecast icon usually means without checking weather.", s2),
        ("hd14", "Explain what multiplication means without solving a particular problem.", s3),
        ("hd15", "Define a time zone offset without giving the current time.", s1),
        ("hd16", "Explain what a search engine index is without searching the web.", s2),
        ("hd17", "In one sentence, explain why weather forecasts have uncertainty.", s3),
        ("hd18", "Describe what the percent key on a calculator represents.", s1),
        ("hd19", "Explain daylight saving time conceptually without checking any clock.", s2),
        ("hd20", "Define a web query in plain English without running one.", s3),
        ("hd21", "Explain what a software release note is without finding the latest release.", s1),
        ("hd22", "Describe what current status means as a UI label.", s2),
        ("hd23", "Explain the difference between climate and weather without live data.", s3),
        ("hd24", "Explain what division means without calculating a number.", s1),

        # Summaries, comparisons, sentiment, classification.
        ("hd25", "Summarize: the weather card moved left, the font got smaller, and tests passed.", s2),
        ("hd26", "Compare a stopwatch and a clock in two short sentences.", s3),
        ("hd27", "Label the sentiment: The search redesign is much easier to use now.", s1),
        ("hd28", "Compare a calculator app and a spreadsheet conceptually.", s2),
        ("hd29", "Summarize: the latest-build badge was renamed and no code behavior changed.", s3),
        ("hd30", "Label the sentiment: The time zone menu is confusing and slow.", s1),
        ("hd31", "Compare JSON and YAML in two concise sentences.", s2),
        ("hd32", "Summarize: all checks passed, the branch is clean, and review can begin.", s3),
        ("hd33", "Classify this as a question or command: Check the weather icon color.", s1),
        ("hd34", "Label the sentiment: The calculator shortcut works exactly as expected.", s2),
        ("hd35", "Compare a browser tab and a browser window.", s3),
        ("hd36", "Summarize: search indexing is disabled in this test environment.", s1),

        # Planning / writing / ordinary direct tasks.
        ("hd37", "Give two steps for testing a profile-photo upload button.", s2),
        ("hd38", "Write one friendly sentence welcoming a user back to an app.", s3),
        ("hd39", "Give three short steps for checking a broken sidebar link.", s1),
        ("hd40", "Write a concise thank-you message for a code review.", s2),
        ("hd41", "Give two steps for testing whether a modal closes correctly.", s3),
        ("hd42", "Write a one-sentence description of a settings page.", s1),
        ("hd43", "Give two ideas for naming a backup feature.", s2),
        ("hd44", "Write a short tooltip explaining a refresh button.", s3),
        ("hd45", "Give two steps for checking keyboard navigation in a menu.", s1),
        ("hd46", "Write a friendly sentence asking someone to retry a failed upload.", s2),
        ("hd47", "Give two short reasons documentation is useful.", s3),
        ("hd48", "Write one sentence describing the difference between save and submit.", s1),
    ]
    return [direct(cid, user, system) for cid, user, system in rows]


def training_cases() -> list[dict]:
    base = probe.balanced_training()  # 16 direct + 64 tools.
    extra = extra_directs()
    train = base + extra
    counts = {label: 0 for label in probe.LABELS}
    for c in train:
        counts[probe.label_of(c)] += 1
    if counts["direct"] != 64:
        raise RuntimeError(f"direct training count must be 64: {counts}")
    for family in FAMILY_LABELS:
        if counts[family] != 16:
            raise RuntimeError(f"tool family {family} must have 16 cases: {counts}")
    ids = [c["id"] for c in train]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate hierarchical training ids")
    forbidden = {c["id"] for c in held.CASES} | {c["id"] for c in confirm.CONFIRM}
    overlap = forbidden.intersection(ids)
    if overlap:
        raise RuntimeError(f"hierarchical training overlaps held-out ids: {sorted(overlap)}")
    return train


def stack(rows, representation: str):
    X = torch.stack([r["features"][representation] for r in rows]).double()
    labels = [r["label"] for r in rows]
    return X, labels


def normalize_train_test(X_train, X_test):
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    a = (X_train - mean) / std
    b = (X_test - mean) / std
    a = torch.cat([a, torch.ones((a.shape[0], 1), dtype=a.dtype)], dim=1)
    b = torch.cat([b, torch.ones((b.shape[0], 1), dtype=b.dtype)], dim=1)
    return a, b, mean, std


def fit_ridge(X, y: torch.Tensor, num_classes: int, ridge: float):
    Xn, _, mean, std = normalize_train_test(X, X)
    Y = F.one_hot(y, num_classes=num_classes).double()
    K = Xn @ Xn.T + float(ridge) * torch.eye(Xn.shape[0], dtype=Xn.dtype)
    alpha = torch.linalg.solve(K, Y)
    W = Xn.T @ alpha
    return {"mean": mean, "std": std, "weight": W, "ridge": float(ridge), "num_classes": num_classes}


def predict(state, X):
    Z = (X - state["mean"]) / state["std"]
    Z = torch.cat([Z, torch.ones((Z.shape[0], 1), dtype=Z.dtype)], dim=1)
    logits = Z @ state["weight"]
    pred = torch.argmax(logits, dim=1)
    top2 = torch.topk(logits, k=min(2, logits.shape[1]), dim=1).values
    margins = top2[:, 0] - top2[:, 1] if logits.shape[1] > 1 else top2[:, 0]
    return pred, logits, margins


def stratified_folds(y: torch.Tensor, num_classes: int):
    folds = [[] for _ in range(FOLDS)]
    for class_id in range(num_classes):
        indices = torch.nonzero(y == class_id, as_tuple=False).flatten().tolist()
        if len(indices) < FOLDS:
            raise RuntimeError(f"class {class_id} has too few examples for {FOLDS}-fold CV")
        for offset, index in enumerate(indices):
            folds[offset % FOLDS].append(index)
    return [sorted(fold) for fold in folds]


def cv_ridge(X, y: torch.Tensor, num_classes: int):
    folds = stratified_folds(y, num_classes)
    all_indices = set(range(len(y)))
    records = []
    for ridge in RIDGES:
        correct = total = 0
        for test_idx in folds:
            train_idx = sorted(all_indices.difference(test_idx))
            state = fit_ridge(X[train_idx], y[train_idx], num_classes, ridge)
            pred, _, _ = predict(state, X[test_idx])
            truth = y[test_idx]
            correct += int((pred == truth).sum().item())
            total += len(test_idx)
        records.append({"ridge": float(ridge), "correct": correct, "total": total, "accuracy": correct / total})
    best = sorted(records, key=lambda r: (r["accuracy"], -r["ridge"]), reverse=True)[0]
    return best, records


def extract(model, tokenizer, cases):
    rows, reps, _blocks, _head = probe.extract_representations(model, tokenizer, cases)
    if REPRESENTATION not in reps:
        raise RuntimeError(f"{REPRESENTATION} unavailable: {reps}")
    return rows


def prepare_heads(rows):
    X, labels = stack(rows, REPRESENTATION)

    # Stage 1 is deliberately balanced: 64 direct vs 64 aggregate tool.
    binary_y = torch.tensor([0 if label == "direct" else 1 for label in labels], dtype=torch.long)
    if int((binary_y == 0).sum()) != 64 or int((binary_y == 1).sum()) != 64:
        raise RuntimeError("binary stage is not 64/64 balanced")
    binary_cv, binary_records = cv_ridge(X, binary_y, 2)
    binary_state = fit_ridge(X, binary_y, 2, binary_cv["ridge"])

    # Stage 2 uses only the 64 tool examples, 16 per family.
    tool_indices = [i for i, label in enumerate(labels) if label != "direct"]
    X_tool = X[tool_indices]
    family_y = torch.tensor([FAMILY_LABELS.index(labels[i]) for i in tool_indices], dtype=torch.long)
    for class_id in range(4):
        if int((family_y == class_id).sum()) != 16:
            raise RuntimeError("family stage is not 16/class balanced")
    family_cv, family_records = cv_ridge(X_tool, family_y, 4)
    family_state = fit_ridge(X_tool, family_y, 4, family_cv["ridge"])

    return {
        "binary": binary_state,
        "family": family_state,
        "binary_cv": binary_cv,
        "binary_cv_records": binary_records,
        "family_cv": family_cv,
        "family_cv_records": family_records,
    }


def hierarchical_predict(heads, X):
    binary_pred, binary_logits, binary_margin = predict(heads["binary"], X)
    family_pred, family_logits, family_margin = predict(heads["family"], X)
    labels = []
    margins = []
    for i in range(X.shape[0]):
        if int(binary_pred[i]) == 0:
            labels.append("direct")
            margins.append(float(binary_margin[i]))
        else:
            labels.append(FAMILY_LABELS[int(family_pred[i])])
            margins.append(float(min(binary_margin[i], family_margin[i])))
    return labels, margins


def score(predicted: list[str], rows, cases, margins: list[float]):
    truth = [r["label"] for r in rows]
    if len(predicted) != len(truth) or len(cases) != len(truth):
        raise RuntimeError("prediction/row/case length mismatch")
    correct = binary_correct = family_correct = family_total = 0
    detail = []
    confusion = {label: {p: 0 for p in probe.LABELS} for label in probe.LABELS}
    for pred, actual, case, margin in zip(predicted, truth, cases, margins):
        ok = pred == actual
        correct += int(ok)
        pred_tool = pred != "direct"
        actual_tool = actual != "direct"
        binary_correct += int(pred_tool == actual_tool)
        if actual_tool:
            family_total += 1
            family_correct += int(pred == actual)
        confusion[actual][pred] += 1
        detail.append({
            "id": case["id"],
            "truth": actual,
            "predicted": pred,
            "correct": ok,
            "margin": float(margin),
        })
    correct_margins = [r["margin"] for r in detail if r["correct"]]
    wrong_margins = [r["margin"] for r in detail if not r["correct"]]
    return {
        "five_way_correct": correct,
        "total": len(truth),
        "five_way_accuracy": correct / len(truth),
        "direct_vs_tool_correct": binary_correct,
        "direct_vs_tool_accuracy": binary_correct / len(truth),
        "tool_family_correct": family_correct,
        "tool_family_total": family_total,
        "tool_family_accuracy": family_correct / family_total if family_total else None,
        "min_correct_margin": min(correct_margins) if correct_margins else None,
        "max_wrong_margin": max(wrong_margins) if wrong_margins else None,
        "confusion": confusion,
        "rows": detail,
    }


def evaluate(heads, old_rows, confirm_rows):
    X_old, _ = stack(old_rows, REPRESENTATION)
    X_confirm, _ = stack(confirm_rows, REPRESENTATION)
    old_pred, old_margin = hierarchical_predict(heads, X_old)
    new_pred, new_margin = hierarchical_predict(heads, X_confirm)
    return {
        "old_heldout": score(old_pred, old_rows, held.CASES, old_margin),
        "confirmation": score(new_pred, confirm_rows, confirm.CONFIRM, new_margin),
    }


def exact(evaluation):
    old = evaluation["old_heldout"]
    new = evaluation["confirmation"]
    return (
        old["five_way_correct"] == old["total"]
        and new["five_way_correct"] == new["total"]
        and new["direct_vs_tool_correct"] == new["total"]
        and new["tool_family_correct"] == new["tool_family_total"]
    )


def save_heads(path: Path, heads, precision: str):
    torch.save({
        "schema_version": 1,
        "kind": "ember-hierarchical-routing-head",
        "source_model": held.MODEL_NAME,
        "source_checkpoint_sha256": held.BEST_SHA256 if precision == "full" else held.INT4_SHA256,
        "representation": REPRESENTATION,
        "binary_labels": ["direct", "tool"],
        "family_labels": FAMILY_LABELS,
        "precision": precision,
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
        "strict_confirmation": True,
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
        "# Ember v0.0.54 hierarchical block-3 router",
        "",
        "Ember language weights are frozen.",
        "Stage 1: balanced 64 direct vs 64 tool. Stage 2: 16 examples per tool family.",
        "Tests remain the unchanged prior 20 cases plus the fresh 50-case confirmation set.",
        "",
        "| Router | Binary CV | Family CV | Old held-out | Fresh | Fresh direct/tool | Fresh tool family |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("full", "int4", "full_heads_on_int4"):
        row = report["routers"][key]
        bcv = row.get("binary_cv")
        fcv = row.get("family_cv")
        old = row["evaluation"]["old_heldout"]
        new = row["evaluation"]["confirmation"]
        lines.append(
            f"| {key} | {'—' if bcv is None else f'{bcv['accuracy']:.1%}'} | "
            f"{'—' if fcv is None else f'{fcv['accuracy']:.1%}'} | "
            f"{old['five_way_correct']}/{old['total']} | {new['five_way_correct']}/{new['total']} | "
            f"{new['direct_vs_tool_correct']}/{new['total']} | {new['tool_family_correct']}/{new['tool_family_total']} |"
        )
    lines += [
        "",
        f"Strict full+INT4 confirmation: **{report['strict_confirmation']}**",
        f"Shared full→INT4 router confirmation: **{report['shared_confirmation']}**",
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

    train_cases = training_cases()

    with tempfile.TemporaryDirectory(prefix="ember-hier-router-") as td:
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
        full_confirm = extract(full_model, full_tok, confirm.CONFIRM)
        full_heads = prepare_heads(full_train)
        full_eval = evaluate(full_heads, full_old, full_confirm)

        int4_model, int4_tok = held.load_int4(repo, work / "int4", token)
        int4_model.eval()
        int4_train = extract(int4_model, int4_tok, train_cases)
        int4_old = extract(int4_model, int4_tok, held.CASES)
        int4_confirm = extract(int4_model, int4_tok, confirm.CONFIRM)
        int4_heads = prepare_heads(int4_train)
        int4_eval = evaluate(int4_heads, int4_old, int4_confirm)

        # Cross-precision: apply the full-precision hierarchical heads unchanged to INT4 block-3 states.
        cross_eval = evaluate(full_heads, int4_old, int4_confirm)

        strict = exact(full_eval) and exact(int4_eval)
        shared = exact(cross_eval)

        if strict:
            save_heads(OUT / "router-full-hierarchical.pt", full_heads, "full")
            save_heads(OUT / "router-int4-hierarchical.pt", int4_heads, "int4")

        routers = {
            "full": {
                "binary_cv": full_heads["binary_cv"],
                "binary_cv_records": full_heads["binary_cv_records"],
                "family_cv": full_heads["family_cv"],
                "family_cv_records": full_heads["family_cv_records"],
                "evaluation": full_eval,
            },
            "int4": {
                "binary_cv": int4_heads["binary_cv"],
                "binary_cv_records": int4_heads["binary_cv_records"],
                "family_cv": int4_heads["family_cv"],
                "family_cv_records": int4_heads["family_cv_records"],
                "evaluation": int4_eval,
            },
            "full_heads_on_int4": {
                "binary_cv": None,
                "family_cv": None,
                "evaluation": cross_eval,
            },
        }

        if strict and shared:
            interpretation = (
                "The hierarchical block-3 router achieved exact confirmation in full and INT4, and the full router transfers unchanged to INT4. "
                "Next: run router-controlled generation with one shared router while keeping Ember weights frozen."
            )
        elif strict:
            interpretation = (
                "Precision-specific hierarchical routers achieved exact confirmation, but one shared full router did not. "
                "Next: run router-controlled generation with precision-specific heads while keeping Ember weights frozen."
            )
        else:
            interpretation = (
                "The hierarchical router still has held-out misses. Inspect only those direct/tool boundary cases; tool-family semantics should remain a separate stable second stage."
            )

        report = {
            "schema_version": 1,
            "diagnostic": "ember-v054-hierarchical-block3-router-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_repo": repo,
            "full_checkpoint_sha256": held.BEST_SHA256,
            "int4_checkpoint_sha256": held.INT4_SHA256,
            "representation": REPRESENTATION,
            "training_count": len(train_cases),
            "training_counts": {label: sum(probe.label_of(c) == label for c in train_cases) for label in probe.LABELS},
            "old_heldout_count": len(held.CASES),
            "confirmation_count": len(confirm.CONFIRM),
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
            "event": "hierarchical_router_complete",
            "full_binary_cv": full_heads["binary_cv"],
            "full_family_cv": full_heads["family_cv"],
            "int4_binary_cv": int4_heads["binary_cv"],
            "int4_family_cv": int4_heads["family_cv"],
            "full_old": compact(full_eval["old_heldout"]),
            "full_confirm": compact(full_eval["confirmation"]),
            "int4_old": compact(int4_eval["old_heldout"]),
            "int4_confirm": compact(int4_eval["confirmation"]),
            "full_on_int4_old": compact(cross_eval["old_heldout"]),
            "full_on_int4_confirm": compact(cross_eval["confirmation"]),
            "strict_confirmation": strict,
            "shared_confirmation": shared,
            "interpretation": interpretation,
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
