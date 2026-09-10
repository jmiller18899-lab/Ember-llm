"""Frozen hidden-state routing probe for Ember v0.0.53 step 9.

Question: does Ember already encode direct/weather/calculator/web_search/get_time
semantics in its hidden states, while the language-model output head fails to
turn that representation into stable tool routing?

This diagnostic changes ZERO Ember weights. It extracts the last-token hidden
state after every transformer block for 80 balanced development prompts (16 per
class), fits tiny L2-regularized linear probes with stratified cross-validation,
and evaluates only on the unchanged 20-case v0.0.53 held-out routing challenge.
A nearest-centroid decoder is reported as a non-parametric sanity check.

If a frozen hidden layer linearly separates the held-out classes much better
than Ember's native output behavior, a small dedicated router head/adapter is a
better next move than continuing to distort the LM weights.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import zipfile

from huggingface_hub import HfApi
import torch
import torch.nn.functional as F

from jobs import ember_hf_eval as ev
from jobs import ember_v053_heldout_routing as held
from jobs import ember_v054_routing_repair as v54

OUT = Path("v054-frozen-router-probe")
LABELS = ["direct", "weather", "calculator", "web_search", "get_time"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0)
FOLDS = 4


def label_of(case: dict) -> str:
    return "direct" if case["kind"] == "direct_response" else case["expected_tool"]


def extra_tools() -> list[dict]:
    # Distinct from held.CASES and from the fixed promotion cases.
    W, C, S, T = "weather", "calculator", "web_search", "get_time"
    p = []
    weather = [
        ("w1", "What is the weather in San Diego right now?", "San Diego"),
        ("w2", "Is it raining in Toronto at the moment?", "Toronto"),
        ("w3", "Give me the live temperature in Seattle.", "Seattle"),
        ("w4", "How is the weather in Nashville today?", "Nashville"),
        ("w5", "Do I need an umbrella in Glasgow right now?", "Glasgow"),
        ("w6", "Check the current weather in Madrid.", "Madrid"),
        ("w7", "Is it sunny in San Antonio at the moment?", "San Antonio"),
        ("w8", "What are the live weather conditions in Montreal?", "Montreal"),
        ("w9", "Tell me the current temperature in Oslo.", "Oslo"),
        ("w10", "What is the weather doing in Auckland right now?", "Auckland"),
        ("w11", "Check whether it is raining in Atlanta now.", "Atlanta"),
        ("w12", "Give me today's live weather for Prague.", "Prague"),
    ]
    for cid, user, city in weather:
        p.append(v54.tool(f"probe_{cid}", user, W, {"location": city}, v54.SYSTEM_1 if len(p) % 2 == 0 else v54.SYSTEM_2))

    calc = [
        ("c1", "Calculate 625 plus 879.", "625+879"),
        ("c2", "What is 91 multiplied by 44?", "91*44"),
        ("c3", "Compute 7200 divided by 16.", "7200/16"),
        ("c4", "Calculate 15 percent of 860.", "0.15*860"),
        ("c5", "What is 37 squared?", "37**2"),
        ("c6", "Compute 144 plus 288 plus 432.", "144+288+432"),
        ("c7", "Calculate 999 minus 347.", "999-347"),
        ("c8", "What is 7.5 multiplied by 24?", "7.5*24"),
        ("c9", "Compute 4096 divided by 32.", "4096/32"),
        ("c10", "Calculate 22 percent of 450.", "0.22*450"),
        ("c11", "What is 3 to the power of 7?", "3**7"),
        ("c12", "Calculate 1280 minus 615.", "1280-615"),
    ]
    for i, (cid, user, expr) in enumerate(calc):
        p.append(v54.tool(f"probe_{cid}", user, C, {"expression": expr}, v54.SYSTEM_2 if i % 2 == 0 else v54.SYSTEM_3))

    search = [
        ("s1", "Find the newest stable Ruby release.", "newest stable Ruby release"),
        ("s2", "What is the current stable Debian release?", "current stable Debian release"),
        ("s3", "Find the latest official Docker Engine release.", "latest official Docker Engine release"),
        ("s4", "Look up the newest stable SQLite version.", "newest stable SQLite version"),
        ("s5", "Find a recent official SpaceX announcement.", "recent official SpaceX announcement"),
        ("s6", "What is the latest stable Git release?", "latest stable Git release"),
        ("s7", "Find the current stable nginx release.", "current stable nginx release"),
        ("s8", "Look up the latest official Blender release.", "latest official Blender release"),
        ("s9", "Find the newest stable Firefox release.", "newest stable Firefox release"),
        ("s10", "What is the current stable Fedora release?", "current stable Fedora release"),
        ("s11", "Find a recent official CERN announcement.", "recent official CERN announcement"),
        ("s12", "Look up the newest stable CMake release.", "newest stable CMake release"),
    ]
    for i, (cid, user, query) in enumerate(search):
        p.append(v54.tool(f"probe_{cid}", user, S, {"query": query}, v54.SYSTEM_3 if i % 2 == 0 else v54.SYSTEM_1))

    times = [
        ("t1", "What time is it in Reykjavik right now?", "Reykjavik"),
        ("t2", "Give me the current local time in Bangkok.", "Bangkok"),
        ("t3", "What is the time in Vancouver at this moment?", "Vancouver"),
        ("t4", "Tell me the current time in Cairo.", "Cairo"),
        ("t5", "What time is it in Rome right now?", "Rome"),
        ("t6", "Give me the local time in Manila at the moment.", "Manila"),
        ("t7", "What is the current time in Nairobi?", "Nairobi"),
        ("t8", "Tell me what time it is in Helsinki right now.", "Helsinki"),
        ("t9", "Give me the current local time in Lima.", "Lima"),
        ("t10", "What time is it in Taipei at this moment?", "Taipei"),
        ("t11", "Tell me the current time in Amsterdam.", "Amsterdam"),
        ("t12", "What is the local time in Cape Town right now?", "Cape Town"),
    ]
    for i, (cid, user, zone) in enumerate(times):
        p.append(v54.tool(f"probe_{cid}", user, T, {"timezone": zone}, v54.SYSTEM_1 if i % 2 == 0 else v54.SYSTEM_3))
    return p


def balanced_training() -> list[dict]:
    direct = [c for c in v54.TRAIN if c["kind"] == "direct_response"]
    existing_tools = [c for c in v54.TRAIN if c["kind"] == "tool_call"]
    extra = extra_tools()
    cases = direct + existing_tools + extra
    counts = {label: 0 for label in LABELS}
    for c in cases:
        counts[label_of(c)] += 1
    if any(counts[label] != 16 for label in LABELS):
        raise RuntimeError(f"probe training is not balanced 16/class: {counts}")
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate probe-training case ids")
    held_ids = {c["id"] for c in held.CASES}
    if held_ids.intersection(ids):
        raise RuntimeError("probe-training ids overlap held-out challenge")
    return cases


def tensor_from_output(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for value in output:
            if torch.is_tensor(value) and value.ndim >= 3:
                return value
    if isinstance(output, dict):
        for value in output.values():
            if torch.is_tensor(value) and value.ndim >= 3:
                return value
    return None


def detect_block_modules(model):
    found = {}
    pattern = re.compile(r"(?:^|\.)(?:h|blocks|layers)\.(\d+)$")
    for name, module in model.named_modules():
        match = pattern.search(name)
        if match:
            index = int(match.group(1))
            found[f"block_{index:02d}"] = (name, module)
    if not found:
        raise RuntimeError("could not discover transformer block modules")
    return dict(sorted(found.items()))


def detect_lm_head(model):
    candidates = []
    vocab = int(model.cfg.vocab_size)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and int(module.out_features) == vocab:
            candidates.append((name, module))
    if not candidates:
        return None
    candidates.sort(key=lambda item: ("lm_head" not in item[0].lower(), len(item[0])))
    return candidates[0]


def extract_representations(model, tokenizer, cases: list[dict]):
    blocks = detect_block_modules(model)
    head = detect_lm_head(model)
    capture = {}
    handles = []

    def block_hook(label):
        def hook(_module, _inputs, output):
            tensor = tensor_from_output(output)
            if tensor is not None:
                capture[label] = tensor[:, -1, :].detach().float().cpu()[0]
        return hook

    for label, (_name, module) in blocks.items():
        handles.append(module.register_forward_hook(block_hook(label)))

    if head is not None:
        def pre_hook(_module, inputs):
            if inputs and torch.is_tensor(inputs[0]):
                capture["final_hidden"] = inputs[0][:, -1, :].detach().float().cpu()[0]
        handles.append(head[1].register_forward_pre_hook(pre_hook))

    rows = []
    with torch.inference_mode():
        for case in cases:
            capture.clear()
            ids = tokenizer.encode(case["prompt"])
            x = torch.tensor([ids], dtype=torch.long)
            model(x, None)
            if not capture:
                raise RuntimeError("forward hooks captured no hidden representations")
            rows.append({
                "id": case["id"],
                "label": label_of(case),
                "features": {name: value.clone() for name, value in capture.items()},
            })

    for handle in handles:
        handle.remove()
    representations = sorted(set.intersection(*(set(r["features"]) for r in rows)))
    if not representations:
        raise RuntimeError("no representation was captured for every case")
    return rows, representations, {name: module_name for name, (module_name, _module) in blocks.items()}, None if head is None else head[0]


def stack(rows, representation: str):
    X = torch.stack([r["features"][representation] for r in rows]).double()
    y = torch.tensor([LABEL_TO_ID[r["label"]] for r in rows], dtype=torch.long)
    return X, y


def normalize_train_test(X_train, X_test):
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    a = (X_train - mean) / std
    b = (X_test - mean) / std
    a = torch.cat([a, torch.ones((a.shape[0], 1), dtype=a.dtype)], dim=1)
    b = torch.cat([b, torch.ones((b.shape[0], 1), dtype=b.dtype)], dim=1)
    return a, b


def ridge_predict(X_train, y_train, X_test, ridge: float):
    Xtr, Xte = normalize_train_test(X_train, X_test)
    Y = F.one_hot(y_train, num_classes=len(LABELS)).double()
    K = Xtr @ Xtr.T
    K = K + float(ridge) * torch.eye(K.shape[0], dtype=K.dtype)
    alpha = torch.linalg.solve(K, Y)
    W = Xtr.T @ alpha
    logits = Xte @ W
    return torch.argmax(logits, dim=1), logits


def stratified_folds(y):
    folds = [[] for _ in range(FOLDS)]
    for class_id in range(len(LABELS)):
        indices = torch.nonzero(y == class_id, as_tuple=False).flatten().tolist()
        for offset, index in enumerate(indices):
            folds[offset % FOLDS].append(index)
    return [sorted(fold) for fold in folds]


def choose_ridge(X, y):
    folds = stratified_folds(y)
    records = []
    all_indices = set(range(len(y)))
    for ridge in LAMBDAS:
        correct = total = 0
        for test_indices in folds:
            train_indices = sorted(all_indices.difference(test_indices))
            pred, _ = ridge_predict(X[train_indices], y[train_indices], X[test_indices], ridge)
            truth = y[test_indices]
            correct += int((pred == truth).sum().item())
            total += len(test_indices)
        records.append({"ridge": float(ridge), "correct": correct, "total": total, "accuracy": correct / total})
    best = sorted(records, key=lambda r: (r["accuracy"], r["ridge"]), reverse=True)[0]
    return best, records


def classify_metrics(pred, truth):
    pred = pred.tolist()
    truth = truth.tolist()
    confusion = {label: {p: 0 for p in LABELS} for label in LABELS}
    correct = 0
    binary_correct = 0
    tool_correct = 0
    tool_total = 0
    rows = []
    for p, t in zip(pred, truth):
        p_label, t_label = LABELS[p], LABELS[t]
        confusion[t_label][p_label] += 1
        correct += int(p == t)
        p_tool = p_label != "direct"
        t_tool = t_label != "direct"
        binary_correct += int(p_tool == t_tool)
        if t_tool:
            tool_total += 1
            tool_correct += int(p == t)
        rows.append({"truth": t_label, "predicted": p_label, "correct": p == t})
    return {
        "five_way_correct": correct,
        "total": len(truth),
        "five_way_accuracy": correct / len(truth),
        "direct_vs_tool_correct": binary_correct,
        "direct_vs_tool_accuracy": binary_correct / len(truth),
        "tool_family_correct": tool_correct,
        "tool_family_total": tool_total,
        "tool_family_accuracy": tool_correct / tool_total if tool_total else None,
        "confusion": confusion,
        "rows": rows,
    }


def centroid_predict(X_train, y_train, X_test):
    Xtr = F.normalize(X_train.float(), dim=1)
    Xte = F.normalize(X_test.float(), dim=1)
    centroids = []
    for class_id in range(len(LABELS)):
        centroid = Xtr[y_train == class_id].mean(dim=0)
        centroids.append(F.normalize(centroid, dim=0))
    C = torch.stack(centroids)
    logits = Xte @ C.T
    return torch.argmax(logits, dim=1), logits


def summary_markdown(report):
    lines = [
        "# Ember v0.0.54 frozen hidden-state routing probe",
        "",
        "Exact saved v0.0.53 step-9 checkpoint; Ember weights are frozen.",
        "Training: 80 balanced prompts (16/class). Test: unchanged 20-case routing challenge.",
        "",
        "| Representation | CV | Held-out 5-way | Direct/tool | Tool family | Centroid 5-way |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["representations"]:
        r = row["ridge_test"]
        c = row["centroid_test"]
        lines.append(
            f"| {row['name']} | {row['cv_best']['accuracy']:.1%} | {r['five_way_correct']}/{r['total']} ({r['five_way_accuracy']:.1%}) | "
            f"{r['direct_vs_tool_correct']}/{r['total']} ({r['direct_vs_tool_accuracy']:.1%}) | "
            f"{r['tool_family_correct']}/{r['tool_family_total']} ({r['tool_family_accuracy']:.1%}) | "
            f"{c['five_way_correct']}/{c['total']} ({c['five_way_accuracy']:.1%}) |"
        )
    best = report["best_representation"]
    lines += [
        "",
        f"Best frozen representation: **{best['name']}** — held-out 5-way {best['ridge_test']['five_way_correct']}/{best['ridge_test']['total']}, "
        f"direct/tool {best['ridge_test']['direct_vs_tool_correct']}/{best['ridge_test']['total']}, "
        f"tool family {best['ridge_test']['tool_family_correct']}/{best['ridge_test']['tool_family_total']}.",
        "",
        f"Interpretation: {report['interpretation']}",
        "",
        "No Ember optimizer step, checkpoint save, export, promotion, or integration occurred.",
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

    with tempfile.TemporaryDirectory(prefix="ember-frozen-router-") as td:
        work = Path(td)
        archive = ev.download_verified(ev.PACKAGE_URL, work / "ember.zip", ev.PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        import sys
        sys.path.insert(0, str(work / "src" / "ember"))

        api = HfApi(token=token)
        owner = api.whoami()["name"]
        repo = f"{owner}/{held.MODEL_NAME}"
        model, tokenizer, checkpoint = held.load_full(repo, work, token)
        if checkpoint.get("train_config", {}).get("version") != "0.0.53":
            raise RuntimeError("expected exact saved v0.0.53 checkpoint")
        model.eval()

        train_cases = balanced_training()
        test_cases = list(held.CASES)
        train_rows, representations, block_modules, head_module = extract_representations(model, tokenizer, train_cases)
        test_rows, test_representations, _, _ = extract_representations(model, tokenizer, test_cases)
        representations = [r for r in representations if r in test_representations]
        if not representations:
            raise RuntimeError("training/test representation sets do not intersect")

        results = []
        for name in representations:
            X_train, y_train = stack(train_rows, name)
            X_test, y_test = stack(test_rows, name)
            cv_best, cv_records = choose_ridge(X_train, y_train)
            pred, logits = ridge_predict(X_train, y_train, X_test, cv_best["ridge"])
            ridge_metrics = classify_metrics(pred, y_test)
            for row, case in zip(ridge_metrics["rows"], test_cases):
                row["id"] = case["id"]
            cpred, _ = centroid_predict(X_train, y_train, X_test)
            centroid_metrics = classify_metrics(cpred, y_test)
            for row, case in zip(centroid_metrics["rows"], test_cases):
                row["id"] = case["id"]
            record = {
                "name": name,
                "dimension": int(X_train.shape[1]),
                "cv_best": cv_best,
                "cv_records": cv_records,
                "ridge_test": ridge_metrics,
                "centroid_test": centroid_metrics,
            }
            results.append(record)
            print(json.dumps({
                "event": "frozen_router_representation",
                "name": name,
                "dimension": record["dimension"],
                "cv": cv_best,
                "ridge": {k: ridge_metrics[k] for k in (
                    "five_way_correct", "total", "direct_vs_tool_correct", "tool_family_correct", "tool_family_total"
                )},
                "centroid": {k: centroid_metrics[k] for k in ("five_way_correct", "total")},
            }), flush=True)

        results.sort(
            key=lambda r: (
                r["ridge_test"]["five_way_accuracy"],
                r["ridge_test"]["direct_vs_tool_accuracy"],
                r["ridge_test"]["tool_family_accuracy"],
                r["cv_best"]["accuracy"],
            ),
            reverse=True,
        )
        best = results[0]
        bm = best["ridge_test"]
        if bm["five_way_correct"] >= 16 and bm["direct_vs_tool_correct"] >= 18 and bm["tool_family_correct"] >= 7:
            interpretation = (
                "Routing semantics are already strongly linearly decodable from frozen Ember hidden states. "
                "The native LM routing/output mapping is the bottleneck; the next best experiment is a tiny dedicated router head/adapter while leaving base language weights frozen."
            )
        elif bm["five_way_correct"] >= 13 and bm["tool_family_correct"] >= 6:
            interpretation = (
                "Frozen Ember hidden states contain substantial routing signal but not a fully robust five-way boundary. "
                "A small router head with a broader routing corpus is still promising and safer than continued whole-LM micro-updates."
            )
        else:
            interpretation = (
                "The frozen hidden representations do not cleanly generalize the five-way routing task. "
                "The problem is deeper than LM-head calibration; Ember needs broader representation-level routing training before a dedicated router head would be reliable."
            )

        report = {
            "schema_version": 1,
            "diagnostic": "ember-v054-frozen-hidden-router-probe-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_repo": repo,
            "checkpoint": held.BEST_PATH,
            "checkpoint_sha256": held.BEST_SHA256,
            "labels": LABELS,
            "training_count": len(train_cases),
            "training_counts": {label: sum(label_of(c) == label for c in train_cases) for label in LABELS},
            "heldout_count": len(test_cases),
            "heldout_ids": [c["id"] for c in test_cases],
            "block_modules": block_modules,
            "lm_head_module": head_module,
            "representations": results,
            "best_representation": best,
            "interpretation": interpretation,
            "ember_weights_changed": False,
            "probe_only": True,
            "elapsed_seconds": time.monotonic() - started,
        }
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (OUT / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
        print(json.dumps({
            "event": "frozen_router_complete",
            "best": {
                "name": best["name"],
                "dimension": best["dimension"],
                "cv_accuracy": best["cv_best"]["accuracy"],
                "five_way_correct": bm["five_way_correct"],
                "direct_vs_tool_correct": bm["direct_vs_tool_correct"],
                "tool_family_correct": bm["tool_family_correct"],
                "tool_family_total": bm["tool_family_total"],
            },
            "interpretation": interpretation,
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
