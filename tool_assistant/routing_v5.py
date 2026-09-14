"""A text-only follow-up to the strongest measured v4 routing control.

This trains the routing helper, not Ember. The model-feature decision is explicit
and fixed before the next evaluation; only ridge is selected on grouped training.
"""
from __future__ import annotations

import torch
from . import family
from .routing_data_v4 import LABELS
from .routing_v4 import fit_head, text_input, validate_request

REVISION = "routing-head-v5"
RIDGES = (0.01, 0.1, 1.0)
FOLDS = 4


def predict(state, texts):
    for text in texts:
        validate_request(text)
    x = family.transform([text_input(t) for t in texts], state["vectorizer"])
    logits = x @ state["weight"]
    top = logits.topk(2, dim=1).values
    return logits.argmax(1), top[:, 0] - top[:, 1]


def select(cases, historical_count):
    if not 0 < historical_count < len(cases):
        raise ValueError("Missing historical/development boundary")
    texts = [text_input(c["user"]) for c in cases]
    y = torch.tensor([LABELS.index(c["route"]) for c in cases])
    groups = sorted({c["group"] for c in cases[historical_count:]})
    if len(groups) < FOLDS:
        raise ValueError("Too few development groups")
    scores = {ridge: [] for ridge in RIDGES}
    predictions = {ridge: [] for ridge in RIDGES}
    for fold in range(FOLDS):
        held = set(groups[fold::FOLDS])
        vi = [i for i in range(historical_count, len(cases)) if cases[i]["group"] in held]
        held_indices = set(vi)
        ti = [i for i in range(len(cases)) if i not in held_indices]
        if set(y[vi].tolist()) != set(range(len(LABELS))):
            raise ValueError("Each development fold must contain all five routes")
        vectorizer = family.fit_vectorizer([texts[i] for i in ti], "hybrid")
        x = family.transform([texts[i] for i in ti], vectorizer)
        xv = family.transform([texts[i] for i in vi], vectorizer)
        for ridge in RIDGES:
            predicted = (xv @ fit_head(x, y[ti], ridge)).argmax(1)
            scores[ridge].append({"fold": fold, "groups": sorted(held),
                                  "correct": int((predicted == y[vi]).sum()), "total": len(vi)})
            predictions[ridge].extend({"id": cases[i]["id"], "expected": cases[i]["route"],
                                       "predicted": LABELS[int(p)]} for i, p in zip(vi, predicted))
    records = [{"ridge": ridge, "folds": folds, "correct": sum(f["correct"] for f in folds),
                "total": sum(f["total"] for f in folds)} for ridge, folds in scores.items()]
    chosen = max(records, key=lambda r: (r["correct"], -r["ridge"]))
    vectorizer = family.fit_vectorizer(texts, "hybrid")
    x = family.transform(texts, vectorizer)
    state = {"revision": REVISION, "labels": list(LABELS), "vectorizer": vectorizer,
             "weight": fit_head(x, y, chosen["ridge"]), "ridge": chosen["ridge"],
             "uses_ember_features": False}
    report = {"revision": REVISION, "training_count": len(cases), "historical_count": historical_count,
              "development_count": len(cases)-historical_count, "selection": chosen,
              "configurations": records, "held_frame_predictions": predictions[chosen["ridge"]],
              "uses_ember_features": False, "base_model_trained": False,
              "interpretation": "Grouped development selection for a text-only helper; not fresh confirmation."}
    return state, report
