"""Objective probes and bounded CPU optimization for Ember v0.0.48."""
from __future__ import annotations

import json
import random
from statistics import mean

from jobs import ember_v048_data as data

base = data.base
TOOL = data.v045.TOOL


def token_contract(tokenizer) -> tuple[int, int]:
    contract = base.semantic_gate.token_contract(tokenizer)
    return int(contract["signatures"][TOOL][0]), int(contract["eos_id"])


def entry_probe(model, tokenizer, torch, cases: list[dict]) -> dict:
    import torch.nn.functional as F
    tool_id, _eos = token_contract(tokenizer)
    rows = []
    with torch.inference_mode():
        for case in cases:
            ids = tokenizer.encode(case["prompt"])
            x = torch.tensor([ids], dtype=torch.long, device="cpu")
            logits, _ = model(x)
            next_logits = logits[0, -1, :].float()
            logp = F.log_softmax(next_logits, dim=-1)
            rank = int((next_logits > next_logits[tool_id]).sum().item()) + 1
            rows.append({
                "id": case["id"], "subtype": case["subtype"],
                "top1": int(torch.argmax(next_logits).item()) == tool_id,
                "rank": rank,
                "loss": -float(logp[tool_id].item()),
            })
    return {
        "cases": len(rows),
        "top1": sum(row["top1"] for row in rows),
        "top1_rate": sum(row["top1"] for row in rows) / len(rows),
        "mean_loss": mean(row["loss"] for row in rows),
        "mean_rank": mean(row["rank"] for row in rows),
        "rows": rows,
    }


def placement_probe(model, tokenizer, torch, cases: list[dict], template: dict) -> dict:
    import torch.nn.functional as F
    rows = []
    with torch.inference_mode():
        for case in cases:
            prompt_ids = tokenizer.encode(case["prompt"])
            target_ids = data.value_continuation_ids(tokenizer, template, case["target"])
            forced = list(template["prefix_ids"])
            tokens = []
            for expected_id in target_ids:
                x = torch.tensor([prompt_ids + forced], dtype=torch.long, device="cpu")
                logits, _ = model(x)
                next_logits = logits[0, -1, :].float()
                logp = F.log_softmax(next_logits, dim=-1)
                greedy = int(torch.argmax(next_logits).item())
                tokens.append({
                    "expected_id": int(expected_id),
                    "top1": greedy == int(expected_id),
                    "loss": -float(logp[int(expected_id)].item()),
                    "rank": int((next_logits > next_logits[int(expected_id)]).sum().item()) + 1,
                })
                forced.append(int(expected_id))
            rows.append({
                "id": case["id"], "subtype": case["subtype"],
                "target_token_count": len(tokens),
                "exact_top1": all(token["top1"] for token in tokens),
                "token_top1": sum(token["top1"] for token in tokens),
                "mean_loss": mean(token["loss"] for token in tokens),
                "tokens": tokens,
            })
    total_tokens = sum(row["target_token_count"] for row in rows)
    return {
        "cases": len(rows),
        "exact_top1": sum(row["exact_top1"] for row in rows),
        "exact_top1_rate": sum(row["exact_top1"] for row in rows) / len(rows),
        "token_top1": sum(row["token_top1"] for row in rows),
        "tokens": total_tokens,
        "token_top1_rate": sum(row["token_top1"] for row in rows) / total_tokens,
        "mean_loss": mean(row["mean_loss"] for row in rows),
        "rows": rows,
    }


def supervised_example(tokenizer, case: dict, objective: str, template: dict, tool_id: int) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer.encode(case["prompt"])
    if objective == "entry":
        x = list(prompt_ids)
        y = [-100] * len(x)
        y[-1] = tool_id
        return x, y
    if objective != "placement":
        raise ValueError(objective)
    value_ids = data.value_continuation_ids(tokenizer, template, case["target"])
    prefix_ids = list(template["prefix_ids"])
    x = list(prompt_ids) + prefix_ids + value_ids[:-1]
    y = [-100] * len(x)
    start = len(prompt_ids) + len(prefix_ids) - 1
    for offset, token_id in enumerate(value_ids):
        y[start + offset] = int(token_id)
    return x, y


def batch_loss(model, torch, examples: list[tuple[list[int], list[int]]], indices: list[int]):
    import torch.nn.functional as F
    max_len = max(len(examples[i][0]) for i in indices)
    xs, ys = [], []
    for i in indices:
        x, y = examples[i]
        xs.append(x + [0] * (max_len - len(x)))
        ys.append(y + [-100] * (max_len - len(y)))
    x_tensor = torch.tensor(xs, dtype=torch.long, device="cpu")
    y_tensor = torch.tensor(ys, dtype=torch.long, device="cpu")
    logits, _ = model(x_tensor)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y_tensor.reshape(-1), ignore_index=-100)


def run_canary(model, tokenizer, torch, cfg: dict, entry_cases: list[dict], placement_cases: list[dict], template: dict) -> list[dict]:
    tool_id, _eos = token_contract(tokenizer)
    entry_examples = [supervised_example(tokenizer, case, "entry", template, tool_id) for case in entry_cases]
    placement_examples = [supervised_example(tokenizer, case, "placement", template, tool_id) for case in placement_cases]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    rng = random.Random(int(cfg["seed"]))
    history = []
    model.train()
    for step in range(1, int(cfg["optimizer_steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        entry_idx = [rng.randrange(len(entry_examples)) for _ in range(int(cfg["entry_batch_size"]))]
        placement_idx = [rng.randrange(len(placement_examples)) for _ in range(int(cfg["placement_batch_size"]))]
        entry_loss = batch_loss(model, torch, entry_examples, entry_idx)
        placement_loss = batch_loss(model, torch, placement_examples, placement_idx)
        total = float(cfg["entry_loss_weight"]) * entry_loss + float(cfg["placement_loss_weight"]) * placement_loss
        if not bool(torch.isfinite(total).item()):
            raise ValueError("non-finite v0.0.48 canary loss")
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip"]), error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 20 == 0 or step == int(cfg["optimizer_steps"]):
            event = {
                "step": step,
                "entry_loss": float(entry_loss.detach()),
                "placement_loss": float(placement_loss.detach()),
                "combined_loss": float(total.detach()),
                "gradient_norm": float(norm),
            }
            history.append(event)
            print(json.dumps({"event": "canary_step", **event}), flush=True)
    del optimizer
    model.eval()
    return history
