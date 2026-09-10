"""Frozen PCA+MLP family-router diagnostic for Ember v0.0.54.

Training/development only. All sealed failed final suites are excluded. Ember
language-model weights remain frozen.

Architecture:
- direct/tool gate: proven ridge head on block_04;
- family representation: concat(block_02, block_03);
- family classifier: tiny one-hidden-layer MLP after a training-only PCA
  projection to 64 dimensions;
- weather/get_time specialist: existing train-CV ridge on block_02+block_03.

MLP hyperparameters are selected only by deterministic stratified four-fold
training CV. PCA is fit separately inside each CV training fold, then refit on
all family-training examples after selection. No development example influences
PCA, MLP fitting, or hyperparameter selection.

CPU-only. No router artifact is saved or integrated. A new untouched final may
be consumed only if full and INT4 both clear all 170 established development
prompts.
"""
from __future__ import annotations

from datetime import datetime, timezone
import copy
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
from jobs import ember_v054_router_third_confirmation as third
from jobs import ember_v054_hierarchical_router as hier
from jobs import ember_v054_expanded_router as expanded
from jobs import ember_v054_representation_sweep as sweep
from jobs import ember_v054_split_representation_router as split

OUT = Path("v054-mlp-family-router")
FAMILIES = tuple(hier.FAMILY_LABELS)
PCA_DIM = 64
FOLDS = 4
STEPS = 140
CANDIDATES = (
    {"width": 8, "lr": 1e-2, "weight_decay": 1e-4},
    {"width": 16, "lr": 1e-2, "weight_decay": 1e-4},
    {"width": 32, "lr": 1e-2, "weight_decay": 1e-4},
    {"width": 16, "lr": 3e-3, "weight_decay": 1e-3},
    {"width": 32, "lr": 3e-3, "weight_decay": 1e-3},
    {"width": 64, "lr": 3e-3, "weight_decay": 1e-3},
)


def unique_cases(*groups):
    out, seen = [], set()
    for group in groups:
        for case in group:
            if case["id"] not in seen:
                seen.add(case["id"])
                out.append(case)
    return out


def family_xy(rows_by_id, cases):
    X = sweep.feature_matrix(rows_by_id, cases, split.FAMILY_REP).float()
    y = torch.tensor([FAMILIES.index(probe.label_of(c)) for c in cases], dtype=torch.long)
    return X, y


def fit_pca(X, dim=PCA_DIM):
    mean = X.mean(dim=0, keepdim=True)
    centered = X - mean
    q = min(int(dim), int(centered.shape[0]) - 1, int(centered.shape[1]))
    # Deterministic exact SVD on the small training matrix.
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:q].T.contiguous()
    projected = centered @ components
    scale = projected.std(dim=0, unbiased=False).clamp_min(1e-6)
    return {"mean": mean, "components": components, "scale": scale}


def pca_transform(X, state):
    return ((X - state["mean"]) @ state["components"]) / state["scale"]


def init_mlp(input_dim, width, seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    w1 = torch.randn(input_dim, width, generator=g, dtype=torch.float32) * (2.0 / max(1, input_dim)) ** 0.5
    b1 = torch.zeros(width, dtype=torch.float32)
    w2 = torch.randn(width, len(FAMILIES), generator=g, dtype=torch.float32) * (2.0 / max(1, width)) ** 0.5
    b2 = torch.zeros(len(FAMILIES), dtype=torch.float32)
    for t in (w1, b1, w2, b2):
        t.requires_grad_(True)
    return [w1, b1, w2, b2]


def forward(params, X):
    w1, b1, w2, b2 = params
    hidden = F.gelu(X @ w1 + b1)
    return hidden @ w2 + b2


def train_mlp(X, y, cfg, seed):
    params = init_mlp(int(X.shape[1]), int(cfg["width"]), seed)
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        logits = forward(params, X)
        loss = F.cross_entropy(logits, y)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("non-finite MLP loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        opt.step()
    return [p.detach().clone() for p in params]


def mlp_predict(params, X):
    with torch.inference_mode():
        logits = forward(params, X)
        pred = logits.argmax(dim=1)
        top2 = torch.topk(logits, k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
    return pred, logits, margin


def stratified_folds(y):
    folds = [[] for _ in range(FOLDS)]
    for cls in range(len(FAMILIES)):
        indices = torch.where(y == cls)[0].tolist()
        for j, idx in enumerate(indices):
            folds[j % FOLDS].append(idx)
    return [sorted(x) for x in folds]


def cv_candidate(X, y, cfg, candidate_index):
    folds = stratified_folds(y)
    all_idx = set(range(int(X.shape[0])))
    correct = 0
    fold_rows = []
    for fold_i, val_idx in enumerate(folds):
        train_idx = sorted(all_idx - set(val_idx))
        train_X = X[train_idx]
        val_X = X[val_idx]
        state = fit_pca(train_X)
        z_train = pca_transform(train_X, state)
        z_val = pca_transform(val_X, state)
        params = train_mlp(z_train, y[train_idx], cfg, seed=20260910 + candidate_index * 100 + fold_i)
        pred, _logits, _margin = mlp_predict(params, z_val)
        fold_correct = int((pred == y[val_idx]).sum().item())
        correct += fold_correct
        fold_rows.append({"fold": fold_i, "correct": fold_correct, "total": len(val_idx)})
    return {
        **cfg,
        "correct": correct,
        "total": int(X.shape[0]),
        "accuracy": correct / int(X.shape[0]),
        "folds": fold_rows,
    }


def select_and_fit(X, y):
    records = []
    for idx, cfg in enumerate(CANDIDATES):
        records.append(cv_candidate(X, y, cfg, idx))
    # Training-only selection: accuracy first, then smaller width and stronger WD.
    best = max(records, key=lambda r: (r["accuracy"], -r["width"], r["weight_decay"], -r["lr"]))
    pca = fit_pca(X)
    z = pca_transform(X, pca)
    chosen_index = list(CANDIDATES).index({k: best[k] for k in ("width", "lr", "weight_decay")})
    params = train_mlp(z, y, best, seed=20260910 + chosen_index * 1000 + 77)
    return {"pca": pca, "params": params, "cv": best, "cv_records": records}


def prepare_precision(rows_by_id, binary_cases, family_cases):
    base = split.fit_heads(rows_by_id, binary_cases, family_cases)
    X, y = family_xy(rows_by_id, family_cases)
    mlp = select_and_fit(X, y)
    return base, mlp


def family_predict(mlp, X_raw):
    z = pca_transform(X_raw.float(), mlp["pca"])
    return mlp_predict(mlp["params"], z)


def predict(rows_by_id, cases, base, mlp):
    xb = sweep.feature_matrix(rows_by_id, cases, split.BINARY_REP)
    xf = sweep.feature_matrix(rows_by_id, cases, split.FAMILY_REP)
    bp, _bl, bm = hier.predict(base["binary"], xb)
    fp, _fl, fm = family_predict(mlp, xf)
    wp, _wl, wm = hier.predict(base["weather_time"], xf)
    labels, margins = [], []
    for i in range(len(cases)):
        if int(bp[i]) == 0:
            labels.append("direct")
            margins.append(float(bm[i]))
            continue
        family = FAMILIES[int(fp[i])]
        if family in {"weather", "get_time"}:
            family = "weather" if int(wp[i]) == 0 else "get_time"
            margins.append(float(min(bm[i], wm[i])))
        else:
            margins.append(float(min(bm[i], fm[i])))
        labels.append(family)
    return labels, margins


def eval_suite(rows_by_id, cases, base, mlp):
    labels, margins = predict(rows_by_id, cases, base, mlp)
    pseudo = [{"id": c["id"], "label": probe.label_of(c)} for c in cases]
    return hier.score(labels, pseudo, cases, margins)


def exact(m):
    return (
        int(m["five_way_correct"]) == int(m["total"])
        and int(m["direct_vs_tool_correct"]) == int(m["total"])
        and int(m["tool_family_correct"]) == int(m["tool_family_total"])
    )


def evaluate(model, tok, all_train, binary_cases, family_cases):
    dev = unique_cases(held.CASES, confirm.CONFIRM, third.THIRD)
    combined = unique_cases(all_train, dev)
    rows, reps, _blocks, _head = probe.extract_representations(model, tok, combined)
    needed = set(split.BINARY_REP + split.FAMILY_REP)
    if not needed.issubset(set(reps)):
        raise RuntimeError(f"missing representations {sorted(needed)} from {reps}")
    by_id = {r["id"]: r for r in rows}
    base, mlp = prepare_precision(by_id, binary_cases, family_cases)
    suites = {
        "old20": eval_suite(by_id, held.CASES, base, mlp),
        "confirm50": eval_suite(by_id, confirm.CONFIRM, base, mlp),
        "third100": eval_suite(by_id, third.THIRD, base, mlp),
    }
    return base, mlp, suites


def summary_md(report):
    lines = [
        "# Ember v0.0.54 frozen PCA+MLP family router", "",
        "Frozen Ember. Binary gate=`block_04`; family MLP=`block_02+block_03` via fold-local PCA.",
        "All sealed final suites excluded; MLP hyperparameters selected by training-only CV.", "",
        "| Mode | Family CV | width | lr | wd | Old20 | Confirm50 | Third100 | Exact170 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in ("full", "int4"):
        r = report["routers"][key]
        s = r["suites"]
        cv = r["mlp_cv"]
        lines.append(
            f"| {key} | {cv['accuracy']:.1%} | {cv['width']} | {cv['lr']} | {cv['weight_decay']} | "
            f"{s['old20']['five_way_correct']}/20 | {s['confirm50']['five_way_correct']}/50 | "
            f"{s['third100']['five_way_correct']}/100 | {r['exact_all_dev']} |"
        )
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

    with tempfile.TemporaryDirectory(prefix="ember-mlp-family-") as td:
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
        fb, fmlp, fs = evaluate(fm, ftok, all_train, binary_cases, family_cases)

        im, itok = held.load_int4(repo, work / "int4", token)
        im.eval()
        ib, imlp, ins = evaluate(im, itok, all_train, binary_cases, family_cases)

        fexact = all(exact(x) for x in fs.values())
        iexact = all(exact(x) for x in ins.values())
        strict = bool(fexact and iexact)
        interpretation = (
            "The frozen PCA+MLP family router clears all 170 established development prompts in full and INT4. Freeze this head class and use a brand-new untouched confirmation."
            if strict else
            "The frozen PCA+MLP family router still has established development misses. Do not consume another final set; preserve evidence and reassess the router model class."
        )
        report = {
            "schema_version": 1,
            "diagnostic": "ember-v054-pca-mlp-family-router-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_repo": repo,
            "binary_representation": list(split.BINARY_REP),
            "family_representation": list(split.FAMILY_REP),
            "pca_dim": PCA_DIM,
            "cv_folds": FOLDS,
            "mlp_steps": STEPS,
            "sealed_finals_imported": False,
            "routers": {
                "full": {
                    "binary_cv": fb["binary_cv"],
                    "weather_time_cv": fb["weather_time_cv"],
                    "mlp_cv": fmlp["cv"],
                    "mlp_cv_records": fmlp["cv_records"],
                    "suites": fs,
                    "exact_all_dev": fexact,
                },
                "int4": {
                    "binary_cv": ib["binary_cv"],
                    "weather_time_cv": ib["weather_time_cv"],
                    "mlp_cv": imlp["cv"],
                    "mlp_cv_records": imlp["cv_records"],
                    "suites": ins,
                    "exact_all_dev": iexact,
                },
            },
            "strict_dev_pass": strict,
            "precision_specific_heads_required": True,
            "ember_weights_changed": False,
            "router_integrated": False,
            "production_changed": False,
            "interpretation": interpretation,
            "elapsed_seconds": time.monotonic() - started,
        }
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (OUT / "summary.md").write_text(summary_md(report), encoding="utf-8")
        print(json.dumps({
            "event": "mlp_family_router_complete",
            "full_cv": fmlp["cv"],
            "int4_cv": imlp["cv"],
            "full": {k: f"{v['five_way_correct']}/{v['total']}" for k, v in fs.items()},
            "int4": {k: f"{v['five_way_correct']}/{v['total']}" for k, v in ins.items()},
            "strict_dev_pass": strict,
            "interpretation": interpretation,
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
