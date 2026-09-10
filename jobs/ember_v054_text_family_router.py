"""Text-intent family router for Ember v0.0.54.

Ember language weights remain frozen. The proven block_04 ridge head still
makes only the direct-vs-tool decision. Once tool mode is selected, tool family
is classified from the user's request text with a tiny TF-IDF linear-kernel
ridge classifier. This deliberately uses a signal independent of Ember hidden
states for the weak weather/search/time boundary.

Family classifier choices (word n-grams, character n-grams, or hybrid) and ridge
are selected only by stratified training CV on the existing 64 examples per
family. The sealed failed final suites are not imported or scored. The 20+50+100
known suites are development evidence only. No router artifact is saved or
integrated here.
"""
from __future__ import annotations

from collections import Counter
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
from jobs import ember_v054_frozen_router_probe as probe
from jobs import ember_v054_router_confirmation as confirm
from jobs import ember_v054_router_third_confirmation as third
from jobs import ember_v054_hierarchical_router as hier
from jobs import ember_v054_expanded_router as expanded
from jobs import ember_v054_representation_sweep as sweep
from jobs import ember_v054_split_representation_router as split

OUT = Path("v054-text-family-router")
FAMILIES = tuple(hier.FAMILY_LABELS)
MODES = ("word", "char", "hybrid")
RIDGES = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
FOLDS = 4
WORD_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)?")


def unique_cases(*groups):
    out, seen = [], set()
    for group in groups:
        for case in group:
            if case["id"] not in seen:
                seen.add(case["id"])
                out.append(case)
    return out


def normalized_text(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def word_features(text: str):
    words = WORD_RE.findall(normalized_text(text))
    feats = [f"w1:{w}" for w in words]
    feats += [f"w2:{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
    return feats


def char_features(text: str):
    text = f" {normalized_text(text)} "
    feats = []
    for n in (3, 4, 5):
        feats.extend(f"c{n}:{text[i:i+n]}" for i in range(max(0, len(text) - n + 1)))
    return feats


def feature_list(text: str, mode: str):
    if mode == "word":
        return word_features(text)
    if mode == "char":
        return char_features(text)
    if mode == "hybrid":
        return word_features(text) + char_features(text)
    raise ValueError(mode)


def fit_vectorizer(texts, mode: str):
    df = Counter()
    for text in texts:
        df.update(set(feature_list(text, mode)))
    # Remove one-off memorization features. Keep deterministic lexical cues.
    vocab = {feat: i for i, feat in enumerate(sorted(k for k, count in df.items() if count >= 2))}
    n = len(texts)
    idf = torch.ones(len(vocab), dtype=torch.double)
    for feat, i in vocab.items():
        idf[i] = math.log((1.0 + n) / (1.0 + df[feat])) + 1.0
    return {"mode": mode, "vocab": vocab, "idf": idf}


def transform(texts, state):
    vocab, idf, mode = state["vocab"], state["idf"], state["mode"]
    X = torch.zeros((len(texts), len(vocab)), dtype=torch.double)
    for r, text in enumerate(texts):
        counts = Counter(feature_list(text, mode))
        for feat, count in counts.items():
            i = vocab.get(feat)
            if i is not None:
                X[r, i] = (1.0 + math.log(float(count))) * idf[i]
    norms = X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return X / norms


def fit_ridge(X, y, ridge: float):
    K = X @ X.T
    Y = F.one_hot(y, num_classes=len(FAMILIES)).double()
    alpha = torch.linalg.solve(K + float(ridge) * torch.eye(K.shape[0], dtype=torch.double), Y)
    return {"X": X, "alpha": alpha, "ridge": float(ridge)}


def predict_ridge(model, X):
    logits = (X @ model["X"].T) @ model["alpha"]
    pred = logits.argmax(dim=1)
    top2 = torch.topk(logits, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    return pred, logits, margin


def stratified_folds(y):
    folds = [[] for _ in range(FOLDS)]
    for cls in range(len(FAMILIES)):
        idx = torch.where(y == cls)[0].tolist()
        for j, item in enumerate(idx):
            folds[j % FOLDS].append(item)
    return [sorted(x) for x in folds]


def cv_config(cases, mode: str, ridge: float):
    y = torch.tensor([FAMILIES.index(probe.label_of(c)) for c in cases], dtype=torch.long)
    folds = stratified_folds(y)
    all_idx = set(range(len(cases)))
    correct = 0
    fold_rows = []
    for fold_i, val_idx in enumerate(folds):
        train_idx = sorted(all_idx - set(val_idx))
        train_texts = [cases[i]["user"] for i in train_idx]
        val_texts = [cases[i]["user"] for i in val_idx]
        vec = fit_vectorizer(train_texts, mode)
        Xtr = transform(train_texts, vec)
        Xv = transform(val_texts, vec)
        model = fit_ridge(Xtr, y[train_idx], ridge)
        pred, _logits, _margin = predict_ridge(model, Xv)
        fold_correct = int((pred == y[val_idx]).sum().item())
        correct += fold_correct
        fold_rows.append({"fold": fold_i, "correct": fold_correct, "total": len(val_idx), "vocab": len(vec["vocab"])})
    return {
        "mode": mode,
        "ridge": float(ridge),
        "correct": correct,
        "total": len(cases),
        "accuracy": correct / len(cases),
        "folds": fold_rows,
    }


def select_family(cases):
    records = [cv_config(cases, mode, ridge) for mode in MODES for ridge in RIDGES]
    best = max(records, key=lambda r: (r["accuracy"], r["mode"] == "hybrid", -r["ridge"]))
    texts = [c["user"] for c in cases]
    y = torch.tensor([FAMILIES.index(probe.label_of(c)) for c in cases], dtype=torch.long)
    vec = fit_vectorizer(texts, best["mode"])
    X = transform(texts, vec)
    model = fit_ridge(X, y, best["ridge"])
    return {"vectorizer": vec, "model": model, "cv": best, "cv_records": records}


def family_predict(head, cases):
    X = transform([c["user"] for c in cases], head["vectorizer"])
    return predict_ridge(head["model"], X)


def prepare_binary(rows_by_id, binary_cases, family_cases):
    # Reuse the exact split-router binary fitting contract; ignore its family heads.
    return split.fit_heads(rows_by_id, binary_cases, family_cases)


def predict(rows_by_id, cases, binary_head, text_head):
    xb = sweep.feature_matrix(rows_by_id, cases, split.BINARY_REP)
    bp, _bl, bm = hier.predict(binary_head["binary"], xb)
    fp, _fl, fm = family_predict(text_head, cases)
    labels, margins = [], []
    for i in range(len(cases)):
        if int(bp[i]) == 0:
            labels.append("direct")
            margins.append(float(bm[i]))
        else:
            labels.append(FAMILIES[int(fp[i])])
            margins.append(float(min(bm[i], fm[i])))
    return labels, margins


def eval_suite(rows_by_id, cases, binary_head, text_head):
    labels, margins = predict(rows_by_id, cases, binary_head, text_head)
    pseudo = [{"id": c["id"], "label": probe.label_of(c)} for c in cases]
    return hier.score(labels, pseudo, cases, margins)


def exact(m):
    return (
        int(m["five_way_correct"]) == int(m["total"])
        and int(m["direct_vs_tool_correct"]) == int(m["total"])
        and int(m["tool_family_correct"]) == int(m["tool_family_total"])
    )


def evaluate_precision(model, tokenizer, all_train, binary_cases, family_cases, text_head):
    dev = unique_cases(held.CASES, confirm.CONFIRM, third.THIRD)
    combined = unique_cases(all_train, dev)
    rows, reps, _blocks, _head = probe.extract_representations(model, tokenizer, combined)
    if "block_04" not in reps:
        raise RuntimeError(f"block_04 unavailable: {reps}")
    by_id = {r["id"]: r for r in rows}
    binary = prepare_binary(by_id, binary_cases, family_cases)
    suites = {
        "old20": eval_suite(by_id, held.CASES, binary, text_head),
        "confirm50": eval_suite(by_id, confirm.CONFIRM, binary, text_head),
        "third100": eval_suite(by_id, third.THIRD, binary, text_head),
    }
    return binary, suites


def summary_md(report):
    cv = report["text_family_cv"]
    lines = [
        "# Ember v0.0.54 text-intent family router", "",
        "Frozen Ember. Direct/tool uses block_04; tool family uses TF-IDF request-text ridge.",
        "Sealed finals excluded; text feature mode and ridge selected by training-only CV.", "",
        f"Selected text family head: **{cv['mode']}**, ridge={cv['ridge']}, CV={cv['correct']}/{cv['total']} ({cv['accuracy']:.1%}).", "",
        "| Mode | Old20 | Confirm50 | Third100 | Exact170 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for key in ("full", "int4"):
        r = report["routers"][key]
        s = r["suites"]
        lines.append(f"| {key} | {s['old20']['five_way_correct']}/20 | {s['confirm50']['five_way_correct']}/50 | {s['third100']['five_way_correct']}/100 | {r['exact_all_dev']} |")
    lines += ["", f"Strict development pass: **{report['strict_dev_pass']}**", "", f"Interpretation: {report['interpretation']}", "", "No router artifact, checkpoint, integration, or production state changed.", ""]
    return "\n".join(lines)


def main():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN required")
    torch.set_num_threads(2)
    torch.manual_seed(20260910)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    all_train, binary_cases, family_cases = expanded.build_training()
    text_head = select_family(family_cases)

    with tempfile.TemporaryDirectory(prefix="ember-text-family-") as td:
        work = Path(td)
        archive = ev.download_verified(ev.PACKAGE_URL, work / "ember.zip", ev.PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        import sys
        sys.path.insert(0, str(work / "src" / "ember"))
        api = HfApi(token=token)
        repo = f"{api.whoami()['name']}/{held.MODEL_NAME}"

        fm, ftok, _ = held.load_full(repo, work / "full", token)
        fm.eval()
        fb, fs = evaluate_precision(fm, ftok, all_train, binary_cases, family_cases, text_head)

        im, itok = held.load_int4(repo, work / "int4", token)
        im.eval()
        ib, ins = evaluate_precision(im, itok, all_train, binary_cases, family_cases, text_head)

        fexact = all(exact(x) for x in fs.values())
        iexact = all(exact(x) for x in ins.values())
        strict = bool(fexact and iexact)
        interpretation = (
            "The split frozen router with a shared text-intent family head clears all 170 established development prompts in full and INT4. Freeze this simpler family router and use a brand-new untouched confirmation."
            if strict else
            "The text-intent family head still has established development misses. Do not consume another final set; preserve the result and inspect only development errors."
        )
        report = {
            "schema_version": 1,
            "diagnostic": "ember-v054-text-intent-family-router-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_repo": repo,
            "binary_representation": list(split.BINARY_REP),
            "family_signal": "user_request_text_tfidf",
            "text_family_cv": text_head["cv"],
            "text_family_cv_records": text_head["cv_records"],
            "text_vocab_size": len(text_head["vectorizer"]["vocab"]),
            "sealed_finals_imported": False,
            "routers": {
                "full": {"binary_cv": fb["binary_cv"], "suites": fs, "exact_all_dev": fexact},
                "int4": {"binary_cv": ib["binary_cv"], "suites": ins, "exact_all_dev": iexact},
            },
            "strict_dev_pass": strict,
            "shared_family_head_across_precisions": True,
            "ember_weights_changed": False,
            "router_integrated": False,
            "production_changed": False,
            "interpretation": interpretation,
            "elapsed_seconds": time.monotonic() - started,
        }
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (OUT / "summary.md").write_text(summary_md(report), encoding="utf-8")
        print(json.dumps({
            "event": "text_family_router_complete",
            "text_family_cv": text_head["cv"],
            "text_vocab_size": len(text_head["vectorizer"]["vocab"]),
            "full": {k: f"{v['five_way_correct']}/{v['total']}" for k, v in fs.items()},
            "int4": {k: f"{v['five_way_correct']}/{v['total']}" for k, v in ins.items()},
            "strict_dev_pass": strict,
            "interpretation": interpretation,
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
