"""Canary decision for Ember v0.0.49.

Every v0.0.48 check is kept at the same threshold -- lowering the learning bar
while lowering the update would make a PASS meaningless -- and two protection
checks are added.
"""
from __future__ import annotations

from jobs import ember_v048_gate as prior


def decide(before_entry: dict, after_entry: dict, before_place: dict, after_place: dict,
           reference: dict, familiar: dict, copy_guard: dict, changed: bool,
           drift: dict, cfg: dict) -> dict:
    decision = prior.decide(
        before_entry, after_entry, before_place, after_place,
        reference, familiar, copy_guard, changed, cfg,
    )
    checks = dict(decision["checks"])
    checks["trust_region"] = bool(drift.get("within_trust_region"))
    metrics = dict(decision["metrics"])
    metrics["max_relative_drift"] = drift.get("max_relative_drift")
    metrics["mean_relative_drift"] = drift.get("mean_relative_drift")
    metrics["trust_region_limit"] = drift.get("limit")
    return {"passed": all(checks.values()), "checks": checks, "metrics": metrics}


def interference_summary(before: dict, after: dict, trajectory: list[dict]) -> dict:
    """The frontier v0.0.48 could not show: what protection cost, step by step."""
    points = [{
        "step": 0,
        "interference_rate": before["rate"],
        "interference_correct": before["correct_envelope_and_tool"],
        "by_kind": before["by_kind"],
    }]
    for snapshot in trajectory:
        probe = snapshot.get("interference", {})
        points.append({
            "step": snapshot["step"],
            "interference_rate": probe.get("rate"),
            "interference_correct": probe.get("correct_envelope_and_tool"),
            "by_kind": probe.get("by_kind"),
            "max_relative_drift": snapshot.get("drift", {}).get("max_relative_drift"),
            "entry_top1": snapshot.get("entry", {}).get("top1"),
            "placement_exact_top1": snapshot.get("placement", {}).get("exact_top1"),
        })
    points.append({
        "step": "final",
        "interference_rate": after["rate"],
        "interference_correct": after["correct_envelope_and_tool"],
        "by_kind": after["by_kind"],
    })
    return {
        "points": points,
        "cases": before["cases"],
        "retained": after["correct_envelope_and_tool"] >= before["correct_envelope_and_tool"],
    }
