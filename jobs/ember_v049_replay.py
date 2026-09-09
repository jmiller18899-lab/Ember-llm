"""Fresh synthetic replay construction and replay objective helpers for v0.0.49."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
from statistics import mean

from jobs import ember_v048_value_compat  # patches context-aware placement tokenization
from jobs import ember_v048_data as v048d
from jobs import ember_v048_objectives as v048o

base = v048d.base
copy_data = v048d.copy_data
v044 = v048d.v044
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config/ember_protected_replay_v0.0.49.json"
ENTRY_SUBTYPES = set(v048d.ENTRY_SUBTYPES)
PLACEMENT_SUBTYPES = set(v048d.PLACEMENT_SUBTYPES)
TARGET_SUBTYPES = ENTRY_SUBTYPES | PLACEMENT_SUBTYPES


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if path.resolve() != DEFAULT_CONFIG.resolve() or cfg.get("version") != "0.0.49":
        raise ValueError("unsupported v0.0.49 configuration")
    if cfg.get("cpu_learning_authorized") is not True:
        raise ValueError("v0.0.49 CPU learning authorization missing")
    if any(cfg.get(k) is not False for k in ("gpu_training_authorized", "production_authorized", "promotion_authorized")):
        raise ValueError("v0.0.49 cannot authorize GPU, production, or promotion")
    if set(cfg.get("entry_subtypes", [])) != ENTRY_SUBTYPES or set(cfg.get("placement_subtypes", [])) != PLACEMENT_SUBTYPES:
        raise ValueError("v0.0.49 target subtype sets changed")
    weights = [float(cfg[k]) for k in (
        "entry_loss_weight", "placement_loss_weight", "tool_replay_loss_weight", "copy_replay_loss_weight"
    )]
    if any(w <= 0 for w in weights) or abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError("v0.0.49 loss weights must be positive and sum to one")
    if not 0 < float(cfg["learning_rate"]) <= 5e-7:
        raise ValueError("v0.0.49 learning rate exceeds protected bound")
    if not 1 <= int(cfg["optimizer_steps"]) <= 120:
        raise ValueError("v0.0.49 step count exceeds protected bound")
    for key in ("tool_replay_values_per_variant", "copy_replay_values_per_variant"):
        if int(cfg.get(key, 0)) <= 0:
            raise ValueError(f"v0.0.49 missing positive replay count: {key}")
    if cfg.get("historical_kind_floor") != v048d.v043.EXPECTED_KIND_FLOOR:
        raise ValueError("historical kind floors changed")
    if cfg.get("historical_subtype_floor") != v048d.v043.EXPECTED_SUBTYPE_FLOOR:
        raise ValueError("historical subtype floors changed")
    return cfg


def _used_values() -> set[str]:
    used = set(v048d.historical_used_values())
    old_cfg = v048d.load_config(v048d.DEFAULT_CONFIG)
    old = v048d.synthetic_values(old_cfg)
    used.update(v for phase in old.values() for values in phase.values() for v in values)
    return used


def target_values(cfg: dict) -> dict[str, dict[str, list[str]]]:
    used = _used_values()
    counts = {
        "template": int(cfg["template_values_per_subtype"]),
        "train": int(cfg["train_values_per_subtype"]),
        "development": int(cfg["development_values_per_subtype"]),
    }
    out = {phase: {} for phase in counts}
    for phase, count in counts.items():
        for subtype in sorted(TARGET_SUBTYPES):
            kind, variant = v048d.SUBTYPE_VARIANT[subtype]
            values = []
            i = 0
            while len(values) < count:
                seed = copy_data._digest("v049-target", phase, subtype, i)
                value = copy_data._render(kind, variant, seed)
                i += 1
                if value in used or v044.subtype_for(kind, value) != subtype:
                    continue
                used.add(value)
                values.append(value)
            out[phase][subtype] = values
    flat = [v for phase in out.values() for values in phase.values() for v in values]
    if len(flat) != len(set(flat)) or set(flat) & set(copy_data.HELD_OUT_VALUES):
        raise ValueError("v0.0.49 target values overlap protected values")
    return out


def replay_value_rows(cfg: dict, family: str) -> list[dict]:
    if family not in {"tool", "copy"}:
        raise ValueError(family)
    used = _used_values()
    rows = []
    count_key = "tool_replay_values_per_variant" if family == "tool" else "copy_replay_values_per_variant"
    count = int(cfg[count_key])
    for kind in copy_data.KINDS:
        for variant in range(int(copy_data.VARIANTS[kind])):
            made = 0
            i = 0
            while made < count:
                seed = copy_data._digest("v049-replay", family, kind, variant, i)
                value = copy_data._render(kind, variant, seed)
                i += 1
                if value in used:
                    continue
                used.add(value)
                rows.append({"kind": kind, "variant": variant, "target": value})
                made += 1
    values = {row["target"] for row in rows}
    if values & set(copy_data.HELD_OUT_VALUES):
        raise ValueError("replay values overlap familiar held-out battery")
    return rows


def tool_prompt(kind: str, value: str) -> dict:
    variant, user = v044.frozen_user(kind, value)
    prompt = v044._prompt(user, v044.baseline_system(kind))
    tool, field = v044.TOOL_BY_KIND[kind]
    return {
        "kind": kind, "target": value, "variant_name": variant,
        "expected_tool": tool, "argument_key": field, "prompt": prompt,
    }


def prepare_tool_replay(model, tokenizer, torch, cfg: dict) -> tuple[list[dict], dict]:
    kept = []
    attempted = Counter()
    for index, row in enumerate(replay_value_rows(cfg, "tool")):
        case = {"id": f"tool_replay_{row['kind']}_{index:03d}", **tool_prompt(row["kind"], row["target"])}
        attempted[row["kind"]] += 1
        generation = base.semantic_gate.generate_completion(model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"]))
        score = v048d.control.score_case(case, generation["completion"])
        if score["envelope_json_valid"] and score["tool_name_correct"] and generation["generated_ids"]:
            kept.append({**case, "source_ids": [int(x) for x in generation["generated_ids"]]})
    by_kind = Counter(row["kind"] for row in kept)
    if len(kept) < int(cfg["minimum_tool_replay_examples"]):
        raise ValueError(f"too few valid tool replay examples: {len(kept)}")
    for kind in copy_data.KINDS:
        if by_kind[kind] < int(cfg["minimum_tool_replay_per_kind"]):
            raise ValueError(f"too few tool replay examples for {kind}: {by_kind[kind]}")
    return kept, {"attempted": dict(attempted), "kept": len(kept), "kept_by_kind": dict(by_kind)}


def prepare_copy_replay(model, tokenizer, torch, cfg: dict) -> tuple[list[dict], dict]:
    kept = []
    for index, row in enumerate(replay_value_rows(cfg, "copy")):
        value = row["target"]
        prompt = copy_data.prompt(value, copy_data._corrupt(value, 3), copy_data._corrupt(value, 11))
        generation = base.semantic_gate.generate_completion(model, tokenizer, torch, prompt, int(cfg["generation_budget"]))
        ids = [int(x) for x in generation["generated_ids"]]
        if ids and not generation["completion"].lstrip().startswith(v048d.v045.TOOL):
            kept.append({"id": f"copy_replay_{row['kind']}_{index:03d}", "kind": row["kind"], "prompt": prompt, "source_ids": ids})
    if len(kept) < int(cfg["minimum_copy_replay_examples"]):
        raise ValueError(f"too few direct copy replay examples: {len(kept)}")
    return kept, {"kept": len(kept), "kept_by_kind": dict(Counter(row["kind"] for row in kept))}


def sequence_example(tokenizer, row: dict) -> tuple[list[int], list[int]]:
    prompt_ids = [int(x) for x in tokenizer.encode(row["prompt"])]
    target_ids = list(row["source_ids"])
    if not prompt_ids or not target_ids:
        raise ValueError("empty replay sequence")
    x = prompt_ids + target_ids[:-1]
    y = [-100] * len(x)
    start = len(prompt_ids) - 1
    for offset, token_id in enumerate(target_ids):
        y[start + offset] = int(token_id)
    return x, y


def replay_probe(model, tokenizer, torch, rows: list[dict], batch_size: int = 8) -> dict:
    examples = [sequence_example(tokenizer, row) for row in rows]
    total = correct = 0
    losses = []
    import torch.nn.functional as F
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            indices = list(range(start, min(start + batch_size, len(examples))))
            max_len = max(len(examples[i][0]) for i in indices)
            xs, ys = [], []
            for i in indices:
                x, y = examples[i]
                xs.append(x + [0] * (max_len - len(x)))
                ys.append(y + [-100] * (max_len - len(y)))
            xt = torch.tensor(xs, dtype=torch.long, device="cpu")
            yt = torch.tensor(ys, dtype=torch.long, device="cpu")
            logits, _ = model(xt)
            mask = yt != -100
            pred = torch.argmax(logits, dim=-1)
            total += int(mask.sum().item())
            correct += int(((pred == yt) & mask).sum().item())
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yt.reshape(-1), ignore_index=-100)
            losses.append(float(loss.item()))
    return {"examples": len(rows), "tokens": total, "token_top1": correct, "token_top1_rate": correct / total, "mean_batch_loss": mean(losses)}


def run_protected_canary(model, tokenizer, torch, cfg: dict, entry_cases, placement_cases, template, tool_rows, copy_rows):
    tool_id, _eos = v048o.token_contract(tokenizer)
    entry_examples = [v048o.supervised_example(tokenizer, case, "entry", template, tool_id) for case in entry_cases]
    placement_examples = [v048o.supervised_example(tokenizer, case, "placement", template, tool_id) for case in placement_cases]
    tool_examples = [sequence_example(tokenizer, row) for row in tool_rows]
    copy_examples = [sequence_example(tokenizer, row) for row in copy_rows]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    rng = random.Random(int(cfg["seed"]))
    history = []
    model.train()
    for step in range(1, int(cfg["optimizer_steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        def sample(n, size): return [rng.randrange(n) for _ in range(size)]
        entry_loss = v048o.batch_loss(model, torch, entry_examples, sample(len(entry_examples), int(cfg["entry_batch_size"])))
        placement_loss = v048o.batch_loss(model, torch, placement_examples, sample(len(placement_examples), int(cfg["placement_batch_size"])))
        tool_loss = v048o.batch_loss(model, torch, tool_examples, sample(len(tool_examples), int(cfg["tool_replay_batch_size"])))
        copy_loss = v048o.batch_loss(model, torch, copy_examples, sample(len(copy_examples), int(cfg["copy_replay_batch_size"])))
        total = (
            float(cfg["entry_loss_weight"]) * entry_loss
            + float(cfg["placement_loss_weight"]) * placement_loss
            + float(cfg["tool_replay_loss_weight"]) * tool_loss
            + float(cfg["copy_replay_loss_weight"]) * copy_loss
        )
        if not bool(torch.isfinite(total).item()):
            raise ValueError("non-finite v0.0.49 protected canary loss")
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip"]), error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 20 == 0 or step == int(cfg["optimizer_steps"]):
            event = {
                "step": step,
                "entry_loss": float(entry_loss.detach()), "placement_loss": float(placement_loss.detach()),
                "tool_replay_loss": float(tool_loss.detach()), "copy_replay_loss": float(copy_loss.detach()),
                "combined_loss": float(total.detach()), "gradient_norm": float(norm),
            }
            history.append(event)
            print(json.dumps({"event": "protected_canary_step", **event}), flush=True)
    del optimizer
    model.eval()
    return history
