"""Localize Ember routing/tool-name gradients versus the protected copy objective.

Runs on the exact saved v0.0.53 step-9 checkpoint. It performs no optimizer
steps. Three gradients are measured independently:
  - direct-vs-tool routing margin on fresh v0.0.54 training prompts;
  - tool-name token CE on true tool prompts;
  - the protected eight-case placement/copy objective.

The report ranks parameter groups by routing/tool-name signal relative to copy
signal so a later repair can update the least-conflicting part of the model.
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

from jobs import ember_hf_eval as ev
from jobs import ember_v053_heldout_routing as held
from jobs import ember_v053_first_token_anchor as base
from jobs import ember_v054_routing_repair as v54
from jobs import ember_alternating_trust_region as trust

OUT = Path("v054-gradient-localize")


def group_name(name: str) -> str:
    m = re.search(r"(?:^|\.)(?:h|blocks|layers)\.(\d+)(?:\.|$)", name)
    if m:
        return f"block_{int(m.group(1)):02d}"
    low = name.lower()
    if "lm_head" in low or "output" in low and "head" in low:
        return "lm_head"
    if any(x in low for x in ("wte", "token_embedding", "tok_emb", "embedding")):
        return "token_embedding"
    if any(x in low for x in ("ln_f", "final_norm", "norm_f")):
        return "final_norm"
    return name.split(".", 1)[0]


def grad_snapshot(model):
    out = {}
    total_sq = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float().cpu().clone()
        out[name] = g
        total_sq += float((g * g).sum())
    return out, math.sqrt(total_sq)


def grad_norms(model):
    norms = {}
    total_sq = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        n2 = float((p.grad.detach().float() ** 2).sum())
        norms[name] = math.sqrt(n2)
        total_sq += n2
    return norms, math.sqrt(total_sq)


def objective_copy(model, tokenizer, selected, template):
    model.zero_grad(set_to_none=True)
    tool_id, _ = trust.objectives.token_contract(tokenizer)
    examples = [
        trust.objectives.supervised_example(tokenizer, c, "placement", template, tool_id)
        for c in selected
    ]
    loss = trust.objectives.batch_loss(model, torch, examples, list(range(len(examples))))
    loss.backward()
    return float(loss.detach())


def objective_direct(model, tokenizer, tool_id, special_ids):
    cases = [c for c in v54.TRAIN if c["kind"] == "direct_response"]
    model.zero_grad(set_to_none=True)
    values = []
    for c in cases:
        logits = base.next_logits(model, tokenizer, c["prompt"])
        loss = v54.route_margin_loss(logits, tool_id, special_ids, False)
        (loss / len(cases)).backward()
        values.append(float(loss.detach()))
    return sum(values) / len(values)


def objective_name(model, tokenizer, contract):
    cases = [c for c in v54.TRAIN if c["kind"] == "tool_call"]
    model.zero_grad(set_to_none=True)
    values = []
    counts = 0
    for c in cases:
        loss, n = v54.name_loss(model, tokenizer, c, contract.get("shared_prefix_id"))
        (loss / len(cases)).backward()
        values.append(float(loss.detach()))
        counts += int(n)
    return sum(values) / len(values), counts


def compare_to_copy(model, copy_grads):
    norms, total = grad_norms(model)
    dot = 0.0
    copy_sq = 0.0
    cur_sq = 0.0
    per_param = {}
    for name, cg in copy_grads.items():
        copy_n = float(cg.norm())
        g = dict(model.named_parameters())[name].grad
        if g is None:
            cur_n = 0.0
            d = 0.0
        else:
            gf = g.detach().float().cpu()
            cur_n = float(gf.norm())
            d = float((gf * cg).sum())
        dot += d
        copy_sq += copy_n * copy_n
        cur_sq += cur_n * cur_n
        per_param[name] = {"norm": cur_n, "copy_norm": copy_n, "dot_copy": d}
    denom = math.sqrt(copy_sq * cur_sq)
    return norms, total, (dot / denom if denom > 0 else 0.0), per_param


def aggregate_groups(copy_grads, direct_stats, name_stats, copy_total, direct_total, name_total):
    groups = {}
    for pname, cg in copy_grads.items():
        gname = group_name(pname)
        row = groups.setdefault(gname, {
            "parameters": 0,
            "copy_sq": 0.0,
            "direct_sq": 0.0,
            "name_sq": 0.0,
            "direct_dot_copy": 0.0,
            "name_dot_copy": 0.0,
        })
        row["parameters"] += int(cg.numel())
        cn = float(cg.norm())
        dn = float(direct_stats.get(pname, {}).get("norm", 0.0))
        nn = float(name_stats.get(pname, {}).get("norm", 0.0))
        row["copy_sq"] += cn * cn
        row["direct_sq"] += dn * dn
        row["name_sq"] += nn * nn
        row["direct_dot_copy"] += float(direct_stats.get(pname, {}).get("dot_copy", 0.0))
        row["name_dot_copy"] += float(name_stats.get(pname, {}).get("dot_copy", 0.0))

    result = []
    for name, row in groups.items():
        cn = math.sqrt(row.pop("copy_sq"))
        dn = math.sqrt(row.pop("direct_sq"))
        nn = math.sqrt(row.pop("name_sq"))
        cshare = cn / copy_total if copy_total else 0.0
        dshare = dn / direct_total if direct_total else 0.0
        nshare = nn / name_total if name_total else 0.0
        repair_share = 0.5 * dshare + 0.5 * nshare
        separation = repair_share / max(cshare, 1e-8)
        score = repair_share * math.log1p(separation)
        result.append({
            "group": name,
            **row,
            "copy_norm": cn,
            "direct_norm": dn,
            "name_norm": nn,
            "copy_share": cshare,
            "direct_share": dshare,
            "name_share": nshare,
            "combined_repair_share": repair_share,
            "repair_to_copy_ratio": separation,
            "localization_score": score,
        })
    result.sort(key=lambda x: x["localization_score"], reverse=True)
    return result


def summary_markdown(report):
    lines = [
        "# Ember v0.0.54 routing gradient localization", "",
        "Exact saved v0.0.53 step-9 checkpoint; gradient-only, no optimizer step.",
        f"Copy loss: {report['losses']['copy']:.6f}",
        f"Direct routing hinge: {report['losses']['direct']:.6f}",
        f"Tool-name CE: {report['losses']['tool_name']:.6f}",
        f"Global cosine direct↔copy: {report['global_cosines']['direct_copy']:+.4f}",
        f"Global cosine tool-name↔copy: {report['global_cosines']['tool_name_copy']:+.4f}",
        "", "| Group | Repair share | Copy share | Repair/copy | Score |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in report["groups"][:12]:
        lines.append(
            f"| {r['group']} | {r['combined_repair_share']:.4f} | {r['copy_share']:.4f} | "
            f"{r['repair_to_copy_ratio']:.2f} | {r['localization_score']:.4f} |"
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

    with tempfile.TemporaryDirectory(prefix="ember-v054-localize-") as td:
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
        contract, tool_id, special_ids = v54.routing_contract(tokenizer)
        cfg, template, template_report, selected, selected_ids, *_ = base.build_placement_fixture(work)

        copy_loss = objective_copy(model, tokenizer, selected, template)
        copy_grads, copy_total = grad_snapshot(model)

        direct_loss = objective_direct(model, tokenizer, tool_id, special_ids)
        direct_norms, direct_total, direct_cos, direct_stats = compare_to_copy(model, copy_grads)

        name_loss_value, name_tokens = objective_name(model, tokenizer, contract)
        name_norms, name_total, name_cos, name_stats = compare_to_copy(model, copy_grads)

        groups = aggregate_groups(copy_grads, direct_stats, name_stats, copy_total, direct_total, name_total)
        params = []
        for pname, cg in copy_grads.items():
            cn = float(cg.norm())
            dn = float(direct_stats.get(pname, {}).get("norm", 0.0))
            nn = float(name_stats.get(pname, {}).get("norm", 0.0))
            cshare = cn / copy_total if copy_total else 0.0
            rshare = 0.5 * (dn / direct_total if direct_total else 0.0) + 0.5 * (nn / name_total if name_total else 0.0)
            ratio = rshare / max(cshare, 1e-8)
            params.append({
                "parameter": pname,
                "group": group_name(pname),
                "copy_share": cshare,
                "combined_repair_share": rshare,
                "repair_to_copy_ratio": ratio,
                "score": rshare * math.log1p(ratio),
            })
        params.sort(key=lambda x: x["score"], reverse=True)

        report = {
            "schema_version": 1,
            "diagnostic": "ember-v054-routing-gradient-localization-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_repo": repo,
            "checkpoint": held.BEST_PATH,
            "checkpoint_sha256": held.BEST_SHA256,
            "selected_case_ids": selected_ids,
            "losses": {"copy": copy_loss, "direct": direct_loss, "tool_name": name_loss_value},
            "tool_name_target_tokens": name_tokens,
            "global_norms": {"copy": copy_total, "direct": direct_total, "tool_name": name_total},
            "global_cosines": {"direct_copy": direct_cos, "tool_name_copy": name_cos},
            "groups": groups,
            "parameters": params[:40],
            "cpu_only": True,
            "optimizer_steps": 0,
            "elapsed_seconds": time.monotonic() - started,
        }
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (OUT / "summary.md").write_text(summary_markdown(report))
        print(json.dumps({
            "event": "v054_gradient_localize_complete",
            "losses": report["losses"],
            "global_norms": report["global_norms"],
            "global_cosines": report["global_cosines"],
            "top_groups": [
                {k: r[k] for k in ("group", "combined_repair_share", "copy_share", "repair_to_copy_ratio", "localization_score")}
                for r in groups[:8]
            ],
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
