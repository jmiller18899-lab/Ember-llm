from __future__ import annotations


def decide(before_entry: dict, after_entry: dict, before_place: dict, after_place: dict,
           reference: dict, familiar: dict, copy_guard: dict, changed: bool, cfg: dict) -> dict:
    entry_reduction = (before_entry["mean_loss"] - after_entry["mean_loss"]) / before_entry["mean_loss"]
    placement_reduction = (before_place["mean_loss"] - after_place["mean_loss"]) / before_place["mean_loss"]
    entry_gain = after_entry["top1"] - before_entry["top1"]
    placement_exact_gain = after_place["exact_top1"] - before_place["exact_top1"]
    placement_token_gain = after_place["token_top1_rate"] - before_place["token_top1_rate"]
    checks = {
        "state_changed": changed,
        "entry_loss": entry_reduction >= float(cfg["gate"]["minimum_entry_development_loss_reduction"]),
        "placement_loss": placement_reduction >= float(cfg["gate"]["minimum_placement_development_loss_reduction"]),
        "entry_top1": entry_gain >= int(cfg["gate"]["minimum_entry_top1_gain"]),
        "placement_exact": placement_exact_gain >= int(cfg["gate"]["minimum_placement_exact_gain"]),
        "placement_tokens": placement_token_gain >= float(cfg["gate"]["minimum_placement_token_top1_gain"]),
        "reference": reference["passed"] is True,
        "familiar": familiar["passed"] is True,
        "copy": copy_guard["passed"] is True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "entry_loss_reduction": entry_reduction,
            "entry_top1_gain": entry_gain,
            "placement_loss_reduction": placement_reduction,
            "placement_exact_gain": placement_exact_gain,
            "placement_token_top1_gain": placement_token_gain,
        },
    }
