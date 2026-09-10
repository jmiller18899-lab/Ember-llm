"""Localize the exact Ember tool-family decision gradient versus protected behavior.

Runs on the saved v0.0.53 step-9 checkpoint. No optimizer steps are taken.
The family objective is a balanced 4-way cross entropy over the first token that
separates weather, calculator, web_search, and get_time after the opening quote
of the tool-name value, conditioned on correct arguments in the existing
arguments-first JSON order.

It compares that gradient against:
  - the protected eight-case placement/copy objective;
  - the broad direct-response routing margin on fresh direct prompts.

CPU-only; no checkpoint write, export, promotion, or integration.
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
from jobs import ember_v053_first_token_anchor as base
from jobs import ember_v054_routing_repair as repair
from jobs import ember_v054_tool_family_diagnostic as family
from jobs import ember_alternating_trust_region as trust

OUT = Path("v054-family-token-localize")
FAMILIES = family.FAMILIES


def group_name(name: str) -> str:
    m = re.search(r"(?:^|\.)(?:h|blocks|layers)\.(\d+)(?:\.|$)", name)
    if m:
        return f"block_{int(m.group(1)):02d}"
    low = name.lower()
    if "lm_head" in low or ("output" in low and "head" in low):
        return "lm_head"
    if any(x in low for x in ("wte", "token_embedding", "tok_emb", "embedding")):
        return "token_embedding"
    if any(x in low for x in ("ln_f", "final_norm", "norm_f")):
        return "final_norm"
    return name.split(".", 1)[0]


def snapshot(model):
    out = {}
    total_sq = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float().cpu().clone()
        out[name] = g
        total_sq += float((g * g).sum())
    return out, math.sqrt(total_sq)


def cosine(a: dict, b: dict) -> float:
    dot = aa = bb = 0.0
    keys = set(a) | set(b)
    for k in keys:
        ga = a.get(k)
        gb = b.get(k)
        if ga is not None:
            aa += float((ga * ga).sum())
        if gb is not None:
            bb += float((gb * gb).sum())
        if ga is not None and gb is not None:
            dot += float((ga * gb).sum())
    denom = math.sqrt(aa * bb)
    return dot / denom if denom else 0.0


def copy_objective(model, tokenizer, selected, template):
    model.zero_grad(set_to_none=True)
    tool_id, _ = trust.objectives.token_contract(tokenizer)
    examples = [trust.objectives.supervised_example(tokenizer, c, "placement", template, tool_id) for c in selected]
    loss = trust.objectives.batch_loss(model, torch, examples, list(range(len(examples))))
    loss.backward()
    return float(loss.detach())


def direct_objective(model, tokenizer, tool_id, special_ids):
    cases = [c for c in repair.TRAIN if c["kind"] == "direct_response"]
    model.zero_grad(set_to_none=True)
    values = []
    for c in cases:
        logits = base.next_logits(model, tokenizer, c["prompt"])
        loss = repair.route_margin_loss(logits, tool_id, special_ids, False)
        (loss / len(cases)).backward()
        values.append(float(loss.detach()))
    return sum(values) / len(values)


def family_token_contract(tokenizer, shared_prefix_id):
    ids = {name: family.continuation_ids(tokenizer, json.dumps(name), shared_prefix_id) for name in FAMILIES}
    shared = family.common_prefix(list(ids.values()))
    if not shared:
        raise RuntimeError("tool-name strings unexpectedly have no shared token prefix")
    if any(len(v) <= len(shared) for v in ids.values()):
        raise RuntimeError("tool-name candidate ended before first distinguishing token")
    next_ids = {name: int(v[len(shared)]) for name, v in ids.items()}
    if len(set(next_ids.values())) != len(FAMILIES):
        raise RuntimeError(f"first distinguishing tool-family tokens are not unique: {next_ids}")
    return ids, shared, next_ids


def family_objective(model, tokenizer, contract):
    cases = [c for c in repair.TRAIN if c["kind"] == "tool_call"]
    shared_prefix_id = contract.get("shared_prefix_id")
    name_ids, shared, next_ids = family_token_contract(tokenizer, shared_prefix_id)
    family_index = {name: i for i, name in enumerate(FAMILIES)}

    model.zero_grad(set_to_none=True)
    losses = []
    rows = []
    for c in cases:
        args_json = json.dumps(c["arguments"], separators=(",", ":"))
        prefix = '<|tool|>\n{"arguments":' + args_json + ',"name":'
        context = [int(x) for x in tokenizer.encode(c["prompt"] + prefix)] + shared
        x = torch.tensor([context], dtype=torch.long)
        logits, _ = model(x, None)
        next_logits = logits[0, -1]
        candidate = torch.stack([next_logits[next_ids[name]] for name in FAMILIES])
        target = torch.tensor([family_index[c["expected_tool"]]], dtype=torch.long)
        loss = F.cross_entropy(candidate.unsqueeze(0), target)
        (loss / len(cases)).backward()
        losses.append(float(loss.detach()))
        ranked = sorted(FAMILIES, key=lambda name: float(next_logits[next_ids[name]].detach()), reverse=True)
        rows.append({
            "id": c["id"],
            "expected": c["expected_tool"],
            "winner": ranked[0],
            "expected_rank": ranked.index(c["expected_tool"]) + 1,
            "candidate_logits": {name: float(next_logits[next_ids[name]].detach()) for name in FAMILIES},
        })
    return sum(losses) / len(losses), rows, {
        "name_token_ids": name_ids,
        "shared_prefix_ids": shared,
        "shared_prefix_text": tokenizer.decode(shared),
        "first_distinguishing_token_ids": next_ids,
        "first_distinguishing_token_text": {name: tokenizer.decode([tid]) for name, tid in next_ids.items()},
    }


def group_rows(model, family_g, copy_g, direct_g, family_total, copy_total, direct_total):
    buckets = {}
    params = dict(model.named_parameters())
    for name in params:
        fg = family_g.get(name)
        cg = copy_g.get(name)
        dg = direct_g.get(name)
        fn = float(fg.norm()) if fg is not None else 0.0
        cn = float(cg.norm()) if cg is not None else 0.0
        dn = float(dg.norm()) if dg is not None else 0.0
        g = buckets.setdefault(group_name(name), {"family_sq": 0.0, "copy_sq": 0.0, "direct_sq": 0.0, "parameters": 0})
        g["family_sq"] += fn * fn
        g["copy_sq"] += cn * cn
        g["direct_sq"] += dn * dn
        g["parameters"] += int(params[name].numel())
    rows = []
    for name, g in buckets.items():
        fn = math.sqrt(g.pop("family_sq"))
        cn = math.sqrt(g.pop("copy_sq"))
        dn = math.sqrt(g.pop("direct_sq"))
        fs = fn / family_total if family_total else 0.0
        cs = cn / copy_total if copy_total else 0.0
        ds = dn / direct_total if direct_total else 0.0
        protected = 0.5 * cs + 0.5 * ds
        ratio = fs / max(protected, 1e-8)
        score = fs * math.log1p(ratio)
        rows.append({
            "group": name,
            **g,
            "family_norm": fn,
            "copy_norm": cn,
            "direct_norm": dn,
            "family_share": fs,
            "copy_share": cs,
            "direct_share": ds,
            "protected_share": protected,
            "family_to_protected_ratio": ratio,
            "localization_score": score,
        })
    rows.sort(key=lambda x: x["localization_score"], reverse=True)
    return rows


def summary_markdown(report):
    lines = [
        "# Ember v0.0.54 exact family-token gradient localization", "",
        "Saved v0.0.53 step-9 candidate; no optimizer steps.",
        f"Family 4-way CE: {report['losses']['family_token']:.6f}",
        f"Copy loss: {report['losses']['copy']:.6f}",
        f"Direct-routing hinge: {report['losses']['direct']:.6f}",
        f"Cosine family↔copy: {report['global_cosines']['family_copy']:+.4f}",
        f"Cosine family↔direct: {report['global_cosines']['family_direct']:+.4f}",
        "", "| Group | Family share | Copy share | Direct share | Family/protected | Score |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["groups"][:12]:
        lines.append(
            f"| {row['group']} | {row['family_share']:.4f} | {row['copy_share']:.4f} | "
            f"{row['direct_share']:.4f} | {row['family_to_protected_ratio']:.2f} | {row['localization_score']:.4f} |"
        )
    lines += ["", "No training, checkpoint save, export, promotion, or integration occurred."]
    return "\n".join(lines) + "\n"


def main():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    torch.set_num_threads(2)
    torch.manual_seed(20260910)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="ember-v054-family-token-") as td:
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
        model.train()
        if float(model.cfg.dropout) != 0:
            raise RuntimeError("zero dropout required")
        contract, tool_id, special_ids = repair.routing_contract(tokenizer)
        cfg, template, template_report, selected, selected_ids, *_ = base.build_placement_fixture(work)

        family_loss, family_rows, family_contract = family_objective(model, tokenizer, contract)
        family_g, family_total = snapshot(model)

        copy_loss = copy_objective(model, tokenizer, selected, template)
        copy_g, copy_total = snapshot(model)

        direct_loss = direct_objective(model, tokenizer, tool_id, special_ids)
        direct_g, direct_total = snapshot(model)

        groups = group_rows(model, family_g, copy_g, direct_g, family_total, copy_total, direct_total)
        report = {
            "schema_version": 1,
            "diagnostic": "ember-v054-exact-family-token-gradient-localization-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_repo": repo,
            "checkpoint": held.BEST_PATH,
            "checkpoint_sha256": held.BEST_SHA256,
            "family_contract": family_contract,
            "family_training_rows": family_rows,
            "selected_copy_case_ids": selected_ids,
            "losses": {"family_token": family_loss, "copy": copy_loss, "direct": direct_loss},
            "global_norms": {"family_token": family_total, "copy": copy_total, "direct": direct_total},
            "global_cosines": {
                "family_copy": cosine(family_g, copy_g),
                "family_direct": cosine(family_g, direct_g),
                "copy_direct": cosine(copy_g, direct_g),
            },
            "groups": groups,
            "cpu_only": True,
            "optimizer_steps": 0,
            "elapsed_seconds": time.monotonic() - started,
        }
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (OUT / "summary.md").write_text(summary_markdown(report))
        print(json.dumps({
            "event": "family_token_localize_complete",
            "losses": report["losses"],
            "global_norms": report["global_norms"],
            "global_cosines": report["global_cosines"],
            "family_contract": family_contract,
            "top_groups": [
                {k: row[k] for k in ("group", "family_share", "copy_share", "direct_share", "family_to_protected_ratio", "localization_score")}
                for row in groups[:8]
            ],
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
