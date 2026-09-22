"""Original block_04 binary ridge/CV contract from Ember commit 13bcf0eb."""
import torch
import torch.nn.functional as F
RIDGES = (0.01, 0.1, 1.0, 10.0, 100.0)
FOLDS = 4

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
