"""Recompute the WR5 automatic verdict from committed evidence. CPU only."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports/evidence/writing-repair5-20261003/evidence"
COPYING = ROOT / "reports/evidence/wr5-copying-20261003"


def load(name, folder=EVIDENCE):
    return json.loads((folder / name).read_text())


def flips(before, after):
    b = {(r["suite"], r["id"]): r for r in before["records"]}
    a = {(r["suite"], r["id"]): r for r in after["records"]}
    lost = [k for k in sorted(b) if b[k]["ok"] and not a[k]["ok"]]
    gained = [k for k in sorted(b) if (not b[k]["ok"]) and a[k]["ok"]]
    return lost, gained


def summary():
    report = load("final-report.json")
    lost, gained = flips(load("baseline-744.json"), load("candidate-744.json"))
    copying = load("final-report.json", COPYING)
    steps = {row["step"]: row["groups"]["benchmark_shortening"]
             for row in copying["checkpoint_summaries"]}
    return {
        "totals": {
            "before_pass": sum(s["pass"] for s in report["before"].values()),
            "after_pass": sum(s["pass"] for s in report["after"].values()),
            "before_fail": sum(s["fail"] for s in report["before"].values()),
            "after_fail": sum(s["fail"] for s in report["after"].values()),
        },
        "lost": lost,
        "gained": gained,
        "shortening": {
            "before": report["shortening_before"]["benchmark"],
            "after": report["shortening_after"]["benchmark"],
        },
        "copying_steps": {str(k): v for k, v in sorted(steps.items())},
        "mean_reference_length_ratio": report["data_audit"]["mean_reference_length_ratio"],
        "rejected": True,
        "reference": "Repair2",
    }


def main():
    result = summary()
    totals = result["totals"]
    if totals["before_pass"] != 714 or totals["after_pass"] != 711:
        raise SystemExit("WR5 totals drifted from the saved 714 → 711 verdict")
    if result["shortening"]["before"]["verbatim_copy"] != 5:
        raise SystemExit("Repair2 copy count drifted")
    if result["shortening"]["after"]["verbatim_copy"] != 8:
        raise SystemExit("WR5 copy count drifted")
    print("WR5_VERDICT", json.dumps(result, default=str))


if __name__ == "__main__":
    main()
