"""Exact text-family ridge implementation from Ember commit 13bcf0eb.
Fitting is used only by build.py; Runtime never fits or selects a model.
"""
from collections import Counter
import math
import re
import torch
import torch.nn.functional as F
FAMILIES = ("weather", "calculator", "web_search", "get_time")
MODES = ("word", "char", "hybrid")
RIDGES = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
FOLDS = 4
WORD_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)?")
def label_of(c):
    return "direct" if c["kind"] == "direct_response" else c["expected_tool"]

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


def predict_ridge(model, X):
    logits = (X @ model["X"].T) @ model["alpha"]
    pred = logits.argmax(dim=1)
    top2 = torch.topk(logits, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    return pred, logits, margin


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


def fit_ridge(X, y, ridge: float):
    K = X @ X.T
    Y = F.one_hot(y, num_classes=len(FAMILIES)).double()
    alpha = torch.linalg.solve(K + float(ridge) * torch.eye(K.shape[0], dtype=torch.double), Y)
    return {"X": X, "alpha": alpha, "ridge": float(ridge)}


def stratified_folds(y):
    folds = [[] for _ in range(FOLDS)]
    for cls in range(len(FAMILIES)):
        idx = torch.where(y == cls)[0].tolist()
        for j, item in enumerate(idx):
            folds[j % FOLDS].append(item)
    return [sorted(x) for x in folds]


def cv_config(cases, mode: str, ridge: float):
    y = torch.tensor([FAMILIES.index(label_of(c)) for c in cases], dtype=torch.long)
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
    y = torch.tensor([FAMILIES.index(label_of(c)) for c in cases], dtype=torch.long)
    vec = fit_vectorizer(texts, best["mode"])
    X = transform(texts, vec)
    model = fit_ridge(X, y, best["ridge"])
    return {"vectorizer": vec, "model": model, "cv": best, "cv_records": records}
