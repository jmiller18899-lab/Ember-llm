"""Protected three-term CPU optimization for Ember v0.0.49.

Two mechanisms separate this from v0.0.48, which learned both objectives and
destroyed everything else:

1. a replay term, weighted to dominate, on the source model's own correct
   envelope and copy behaviour across every kind; and
2. a hard trust region, projected after every optimizer step, that bounds how far
   any parameter tensor may drift from the source. AdamW takes a roughly
   lr-sized step per parameter regardless of gradient magnitude, so a lower
   learning rate alone only slows the drift -- it does not bound it. The
   projection does, and the achieved drift is reported rather than assumed.
"""
from __future__ import annotations

from collections import defaultdict
import json
import random
from statistics import mean

from jobs import ember_v048_objectives as prior
from jobs import ember_v049_data as data

base = data.base
control = data.control
TOOL = prior.TOOL
DEFAULT_MAX_TOKENS = 256
# Rescaling a float32 tensor to an exact target norm lands a little over it: the
# division and multiply each round. Measured slack after a real clip is ~2e-5
# relative, so the containment check uses a relative tolerance rather than an
# absolute epsilon, which would fail the gate on arithmetic noise alone.
DRIFT_TOLERANCE = 1e-3

token_contract = prior.token_contract
entry_probe = prior.entry_probe
placement_probe = prior.placement_probe


def replay_example(tokenizer, row: dict, max_len: int = DEFAULT_MAX_TOKENS):
    """Supervise the source model's own completion, and nothing before it."""
    prompt_ids = [int(i) for i in tokenizer.encode(row["prompt"])]
    completion_ids = [int(i) for i in tokenizer.encode(row["completion"])]
    if not completion_ids:
        return None
    x = prompt_ids + completion_ids[:-1]
    if len(x) > max_len:
        return None
    y = [-100] * len(x)
    start = len(prompt_ids) - 1
    for offset, token_id in enumerate(completion_ids):
        y[start + offset] = token_id
    return x, y


def build_replay_examples(tokenizer, rows: list[dict], max_len: int = DEFAULT_MAX_TOKENS) -> tuple[list, dict]:
    examples, dropped = [], 0
    for row in rows:
        built = replay_example(tokenizer, row, max_len)
        if built is None:
            dropped += 1
            continue
        examples.append(built)
    return examples, {"rows": len(rows), "examples": len(examples), "dropped_too_long": dropped}


class TrustRegion:
    """Bound each parameter tensor's relative L2 drift from the source weights."""

    def __init__(self, model, torch, relative: float):
        self.relative = float(relative)
        self.torch = torch
        self.reference = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }
        self.norms = {
            name: float(tensor.norm().item()) for name, tensor in self.reference.items()
        }

    def _limit(self, name: str) -> float:
        # A zero-norm tensor (a zero-initialised bias) has no meaningful relative
        # scale, so the same number is applied as an absolute cap instead.
        norm = self.norms[name]
        return self.relative * norm if norm > 0 else self.relative

    def project(self, model) -> dict:
        torch = self.torch
        clipped = 0
        worst = 0.0
        total = 0.0
        with torch.no_grad():
            for name, param in model.named_parameters():
                reference = self.reference[name]
                delta = param.detach() - reference
                distance = float(delta.norm().item())
                limit = self._limit(name)
                if distance > limit > 0:
                    param.copy_(reference + delta * (limit / distance))
                    distance = limit
                    clipped += 1
                norm = self.norms[name]
                relative = distance / norm if norm > 0 else distance
                worst = max(worst, relative)
                total += relative
        count = len(self.reference)
        return {
            "clipped_tensors": clipped,
            "max_relative_drift": worst,
            "mean_relative_drift": total / count if count else 0.0,
        }

    def measure(self, model) -> dict:
        torch = self.torch
        worst = 0.0
        total = 0.0
        with torch.no_grad():
            for name, param in model.named_parameters():
                norm = self.norms[name]
                distance = float((param.detach() - self.reference[name]).norm().item())
                relative = distance / norm if norm > 0 else distance
                worst = max(worst, relative)
                total += relative
        count = len(self.reference)
        return {
            "max_relative_drift": worst,
            "mean_relative_drift": total / count if count else 0.0,
            "limit": self.relative,
            "tolerance": DRIFT_TOLERANCE,
            "within_trust_region": worst <= self.relative * (1.0 + DRIFT_TOLERANCE) + 1e-12,
        }


def interference_probe(model, tokenizer, torch, cfg: dict, cases: list[dict]) -> dict:
    """A cheap per-kind envelope check, used to trace interference during the run."""
    by_kind_total: dict[str, int] = defaultdict(int)
    by_kind_ok: dict[str, int] = defaultdict(int)
    correct = 0
    for case in cases:
        generation = base.semantic_gate.generate_completion(
            model, tokenizer, torch, case["prompt"], int(cfg["generation_budget"])
        )
        score = control.score_case(case, generation["completion"])
        ok = bool(score["envelope_json_valid"] and score["tool_name_correct"])
        by_kind_total[case["kind"]] += 1
        if ok:
            by_kind_ok[case["kind"]] += 1
            correct += 1
    return {
        "cases": len(cases),
        "correct_envelope_and_tool": correct,
        "rate": correct / len(cases) if cases else None,
        "by_kind": {kind: by_kind_ok.get(kind, 0) for kind in sorted(by_kind_total)},
        "by_kind_total": dict(sorted(by_kind_total.items())),
    }


def run_canary(model, tokenizer, torch, cfg: dict, entry_cases: list[dict],
               placement_cases: list[dict], replay_rows: list[dict], template: dict,
               probe=None) -> dict:
    """Three-term bounded update. `probe` is called at the configured steps."""
    tool_id, _eos = token_contract(tokenizer)
    max_len = int(getattr(getattr(model, "config", None), "block_size", DEFAULT_MAX_TOKENS) or DEFAULT_MAX_TOKENS)
    entry_examples = [prior.supervised_example(tokenizer, case, "entry", template, tool_id) for case in entry_cases]
    placement_examples = [prior.supervised_example(tokenizer, case, "placement", template, tool_id) for case in placement_cases]
    replay_examples, replay_build = build_replay_examples(tokenizer, replay_rows, max_len)
    if not replay_examples:
        raise ValueError("v0.0.49 requires a non-empty replay corpus")

    trust = TrustRegion(model, torch, float(cfg["trust_region_relative_drift"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    rng = random.Random(int(cfg["seed"]))
    probe_steps = {int(s) for s in cfg.get("trajectory_probe_steps", [])}
    history, trajectory = [], []
    clipped_steps = 0
    total_steps = int(cfg["optimizer_steps"])
    model.train()
    for step in range(1, total_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        entry_idx = [rng.randrange(len(entry_examples)) for _ in range(int(cfg["entry_batch_size"]))]
        placement_idx = [rng.randrange(len(placement_examples)) for _ in range(int(cfg["placement_batch_size"]))]
        replay_idx = [rng.randrange(len(replay_examples)) for _ in range(int(cfg["replay_batch_size"]))]
        entry_loss = prior.batch_loss(model, torch, entry_examples, entry_idx)
        placement_loss = prior.batch_loss(model, torch, placement_examples, placement_idx)
        replay_loss = prior.batch_loss(model, torch, replay_examples, replay_idx)
        total = (
            float(cfg["entry_loss_weight"]) * entry_loss
            + float(cfg["placement_loss_weight"]) * placement_loss
            + float(cfg["replay_loss_weight"]) * replay_loss
        )
        if not bool(torch.isfinite(total).item()):
            raise ValueError("non-finite v0.0.49 canary loss")
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip"]), error_if_nonfinite=True)
        optimizer.step()
        projection = trust.project(model)
        if projection["clipped_tensors"]:
            clipped_steps += 1
        if step == 1 or step % 20 == 0 or step == total_steps:
            event = {
                "step": step,
                "entry_loss": float(entry_loss.detach()),
                "placement_loss": float(placement_loss.detach()),
                "replay_loss": float(replay_loss.detach()),
                "combined_loss": float(total.detach()),
                "gradient_norm": float(norm),
                **projection,
            }
            history.append(event)
            print(json.dumps({"event": "canary_step", **event}), flush=True)
        if step in probe_steps and probe is not None:
            model.eval()
            snapshot = {"step": step, "drift": trust.measure(model), **probe(model, step)}
            trajectory.append(snapshot)
            print(json.dumps({"event": "trajectory", "step": step,
                              "interference": snapshot.get("interference", {}).get("rate")}), flush=True)
            model.train()
    del optimizer
    model.eval()
    final_drift = trust.measure(model)
    # A cap that never activates protected nothing: the learning rate and the
    # replay term did all the work. Say so rather than letting the mechanism take
    # credit it did not earn.
    final_drift["steps_clipped"] = clipped_steps
    final_drift["engaged"] = clipped_steps > 0
    final_drift["headroom"] = (
        final_drift["max_relative_drift"] / self_limit
        if (self_limit := float(cfg["trust_region_relative_drift"])) else None
    )
    return {
        "history": history,
        "trajectory": trajectory,
        "replay_build": replay_build,
        "final_drift": final_drift,
        "objective_rows": {
            "entry": len(entry_examples),
            "placement": len(placement_examples),
            "replay": len(replay_examples),
        },
    }
