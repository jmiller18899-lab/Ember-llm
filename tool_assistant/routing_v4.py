"""Five-way routing heads over frozen Ember and development-only text features.

Model weights never change. A text-only arm measures whether the frozen Ember
features add value; its selection must be reported rather than called LLM learning.
"""
from __future__ import annotations

import re
import torch
import torch.nn.functional as F

from . import family
from .routing_data_v4 import LABELS

REVISION = "routing-head-v4"
RIDGES = (0.01, 0.1, 1.0)
EMBEDDING_WEIGHTS = (0.0, 0.25, 1.0, 4.0)
FOLDS = 4


def text_input(text):
    # Numeric identity is irrelevant to tool choice. Keep operators and words,
    # including current-information cues, while sharing arithmetic patterns.
    return re.sub(r"(?<!\w)\d+(?:\.\d+)?", "#", text.casefold())


def validate_request(user):
    if not isinstance(user, str) or not user.strip():
        raise ValueError("Request must be a nonempty string")
    if "<|" in user:
        raise ValueError("Requests cannot contain Ember conversation markers")


def fit_features(texts, hidden):
    vectorizer = family.fit_vectorizer([text_input(s) for s in texts], "hybrid")
    return {"vectorizer": vectorizer, "mean": hidden.mean(0, keepdim=True),
            "std": hidden.std(0, unbiased=False, keepdim=True).clamp_min(1e-5)}


def transform(texts, hidden, features, embedding_weight):
    text = family.transform([text_input(s) for s in texts], features["vectorizer"])
    if not embedding_weight:
        return text
    z = (hidden.double() - features["mean"]) / features["std"]
    z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return torch.cat((text, z * float(embedding_weight)), dim=1)


def fit_head(x, labels, ridge):
    y = F.one_hot(labels, num_classes=len(LABELS)).double()
    alpha = torch.linalg.solve(x @ x.T + ridge * torch.eye(len(x), dtype=x.dtype), y)
    # Collapse the dual fit to a compact linear inference matrix; no training
    # requests, labels or nearest-neighbour lookup are needed at runtime.
    return x.T @ alpha


def predict(state, texts, hidden):
    x = transform(texts, hidden, state["features"], state["embedding_weight"])
    logits = x @ state["weight"]
    labels = logits.argmax(1)
    top = logits.topk(2, dim=1).values
    return labels, logits, top[:, 0] - top[:, 1]


def select(cases, hidden, historical_count):
    """Select using held-out development frames; historical training stays train.

    All four city/number variants and all labels in a frame are held out
    together. Text vocabulary and feature normalization are fitted per fold.
    Confirmation data must not enter this function.
    """
    if hidden.ndim != 2 or len(hidden) != len(cases) or not torch.isfinite(hidden).all():
        raise ValueError("Expected finite features for every training request")
    if not 0 < historical_count < len(cases):
        raise ValueError("Historical/development boundary is missing")
    groups = sorted({c["group"] for c in cases[historical_count:]})
    if len(groups) < FOLDS:
        raise ValueError("Too few independent development frame groups")
    texts = [c["user"] for c in cases]
    y = torch.tensor([LABELS.index(c["route"]) for c in cases])
    scores = {(w, r): [] for w in EMBEDDING_WEIGHTS for r in RIDGES}
    oof = {(w, r): [] for w in EMBEDDING_WEIGHTS for r in RIDGES}
    for fold in range(FOLDS):
        held = set(groups[fold::FOLDS])
        vi = [i for i in range(historical_count, len(cases)) if cases[i]["group"] in held]
        held_indices = set(vi)
        ti = [i for i in range(len(cases)) if i not in held_indices]
        if set(y[vi].tolist()) != set(range(len(LABELS))):
            raise ValueError("Every development fold must contain all routes")
        feats = fit_features([texts[i] for i in ti], hidden[ti])
        for w in EMBEDDING_WEIGHTS:
            x = transform([texts[i] for i in ti], hidden[ti], feats, w)
            xv = transform([texts[i] for i in vi], hidden[vi], feats, w)
            for ridge in RIDGES:
                weight = fit_head(x, y[ti], ridge)
                pred = (xv @ weight).argmax(1)
                scores[w, ridge].append({"fold": fold, "groups": sorted(held),
                    "correct": int((pred == y[vi]).sum()), "total": len(vi)})
                oof[w, ridge].extend({"id": cases[i]["id"], "expected": cases[i]["route"],
                                     "predicted": LABELS[int(p)]} for i, p in zip(vi, pred))
    records = [{"embedding_weight": w, "ridge": r,
                "correct": sum(f["correct"] for f in rows),
                "total": sum(f["total"] for f in rows), "folds": rows}
               for (w, r), rows in scores.items()]
    # Accuracy first; ties prefer the smaller feature contribution and ridge.
    # This avoids claiming a benefit from Ember when a text-only control ties.
    best = max(records, key=lambda r: (r["correct"], -r["embedding_weight"], -r["ridge"]))
    features = fit_features(texts, hidden)
    x = transform(texts, hidden, features, best["embedding_weight"])
    state = {"revision": REVISION, "labels": list(LABELS), "features": features,
             "embedding_weight": best["embedding_weight"], "ridge": best["ridge"],
             "weight": fit_head(x, y, best["ridge"])}
    return state, {"selection": best, "configurations": records,
                   "held_frame_predictions": oof[best["embedding_weight"], best["ridge"]],
                   "training_count": len(cases), "development_count": len(cases)-historical_count,
                   "base_model_trained": False,
                   "interpretation": "Grouped development cross-validation; not fresh confirmation."}
