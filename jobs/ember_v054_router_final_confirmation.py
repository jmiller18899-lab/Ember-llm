"""Final untouched routing confirmation for Ember v0.0.54 frozen router.

Architecture and training are frozen before this test:
  * precision-specific expanded hierarchical block-3 router;
  * binary direct/tool gate;
  * four-way tool-family head;
  * weather-vs-get_time specialist used only inside that family pair.

This test contains 150 new prompts, exactly 30/class. None are used for fitting,
CV, architecture selection, or threshold selection. Full and INT4 routers are
fit only from the already-frozen training corpus. No tuning is allowed after
seeing this test. Ember weights remain unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

from huggingface_hub import HfApi
import torch

from jobs import ember_hf_eval as ev
from jobs import ember_v053_heldout_routing as held
from jobs import ember_v054_frozen_router_probe as probe
from jobs import ember_v054_router_confirmation as confirm
from jobs import ember_v054_router_third_confirmation as third
from jobs import ember_v054_hierarchical_router as hier
from jobs import ember_v054_expanded_router as expanded
from jobs import ember_v054_weather_time_specialist as specialist
from jobs import ember_v054_routing_repair as v54

OUT = Path("v054-router-final-confirmation")
SYSTEM_Y = (
    "You are Ember. Decide whether the request should be answered directly or routed to one of four tools: "
    "weather, calculator, web_search, get_time. Writing, editing, summarizing, explaining, comparing, planning, "
    "and classification are direct unless the request actually requires live weather, arithmetic, a current external fact, "
    "or the present local time. Route by intent rather than keywords."
)
SYSTEM_Z = (
    "You are Ember. Available tools: weather, calculator, web_search, get_time. Use exactly the capability the task needs. "
    "Mentions of weather, current, latest, search, time, or calculator do not by themselves require a tool."
)
SYSTEMS = (SYSTEM_Y, SYSTEM_Z, v54.SYSTEM_1, v54.SYSTEM_3)


def p(system: str, user: str) -> str:
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"


def d(cid: str, text: str, i: int) -> dict:
    return {"id": cid, "kind": "direct_response", "user": text, "prompt": p(SYSTEMS[i % 4], text)}


def t(cid: str, text: str, family: str, i: int) -> dict:
    return {"id": cid, "kind": "tool_call", "user": text, "prompt": p(SYSTEMS[i % 4], text), "expected_tool": family}


DIRECT_TEXTS = [
    "Label the sentiment: The rollout completed smoothly and users are happy with the result.",
    "Label the sentiment: The current weather card title is difficult to read.",
    "Rewrite this title clearly: latest search panel styling cleanup.",
    "Rewrite this message professionally: the calculator button is kind of confusing.",
    "Summarize: the time-zone menu was reorganized, tests passed, and behavior stayed the same.",
    "Summarize: the weather mockup uses sample data and all visual checks succeeded.",
    "Explain what a current-status badge means without checking any live status.",
    "Explain why weather measurements can vary between nearby stations without checking live weather.",
    "Explain what a calculator percentage key represents without computing a number.",
    "Explain what local time means conceptually without checking a clock.",
    "Explain what a search-results page is without performing a search.",
    "Compare a weather widget and a clock widget as interface components.",
    "Compare a calculator and a notepad in two concise sentences.",
    "Compare a search box and an address bar without browsing the web.",
    "Give three steps for testing a current-status label using mock data.",
    "Give two steps for checking a weather-card loading animation in a prototype.",
    "Give three steps for testing calculator keyboard shortcuts without doing arithmetic.",
    "Give two steps for testing a time-zone selector without checking the current time.",
    "Classify as command or statement: Review the latest release heading.",
    "Classify as positive, negative, or neutral: The search interface is much clearer now.",
    "Classify as question or statement: The current time label appears in the header.",
    "Write a short tooltip for a weather preferences button.",
    "Write a friendly empty-state message for a search-results screen.",
    "Write one sentence describing a calculator history panel.",
    "Write a concise tooltip for a time-zone preference.",
    "Make this friendlier: Current information is unavailable in this mockup.",
    "Shorten this heading: latest weather-related documentation updates.",
    "Polish this bug title: search current page button has too much padding.",
    "Describe the difference between a live value and a cached value without looking anything up.",
    "Write one sentence thanking someone for reviewing the calculator and weather documentation.",
]

WEATHER_LOCS = [
    "Boise","Charleston, South Carolina","Anchorage","Quebec City","Victoria, British Columbia",
    "Oslo","Stockholm","Gothenburg","Warsaw","Vienna","Salzburg","Nice","Bordeaux","Edinburgh","Liverpool",
    "Osaka","Kobe","Jeju","Taipei","Kuala Lumpur","Jakarta","Cebu","Christchurch","Hobart","Darwin",
    "Casablanca","Tunis","Dakar","Windhoek","Gaborone",
]
TIME_LOCS = [
    "Istanbul","Izmir","Warsaw","Vienna","Zurich","Brussels","Luxembourg","Stockholm","Oslo","Reykjavik",
    "Tokyo","Osaka","Seoul","Hong Kong","Kuala Lumpur","Jakarta","Manila","Auckland","Sydney","Melbourne",
    "Toronto","Montreal","Denver","Seattle","Atlanta","Bogota","Santiago","Sao Paulo","Cape Town","Casablanca",
]
SEARCH_ITEMS = [
    "Kotlin","Scala","GCC","Clang","Cython","Poetry","Pydantic","SQLAlchemy","Celery","Redis",
    "MariaDB","TimescaleDB","InfluxDB","Kubernetes","Helm","Argo CD","Flux CD","Podman","Buildah","containerd",
    "Electron","Tauri","Playwright","Selenium","JupyterLab","Home Assistant","Godot","Unreal Engine","GIMP","Inkscape",
]


def build_cases() -> list[dict]:
    rows = [d(f"final_d{i+1:02d}", text, i) for i, text in enumerate(DIRECT_TEXTS)]
    weather_templates = [
        "What is the current weather in {x}?",
        "What is the temperature in {x} right now?",
        "Are there live rain conditions in {x} at the moment?",
        "Tell me the weather conditions in {x} right now.",
        "Should I bring an umbrella in {x} today based on the current weather?",
    ]
    for i, loc in enumerate(WEATHER_LOCS):
        rows.append(t(f"final_w{i+1:02d}", weather_templates[i % 5].format(x=loc), "weather", i + 31))

    for i in range(30):
        mode = i % 5
        if mode == 0:
            text = f"Calculate {541 + i*17} plus {83 + i*5}."
        elif mode == 1:
            text = f"What is {27 + i} multiplied by {19 + i*2}?"
        elif mode == 2:
            b = 7 + (i % 11)
            a = b * (211 + i*3)
            text = f"Compute {a} divided by {b}."
        elif mode == 3:
            text = f"Calculate {9 + i} percent of {360 + i*22}."
        else:
            text = f"What is {5 + (i % 9)} to the power of {3 + (i % 4)}?"
        rows.append(t(f"final_c{i+1:02d}", text, "calculator", i + 61))

    search_templates = [
        "Find the newest stable {x} release.",
        "What is the current official {x} version?",
        "Look up the latest stable {x} release.",
        "Find the most recent official {x} release.",
        "What is the newest production release of {x}?",
    ]
    for i, item in enumerate(SEARCH_ITEMS):
        rows.append(t(f"final_s{i+1:02d}", search_templates[i % 5].format(x=item), "web_search", i + 91))

    time_templates = [
        "What time is it in {x} right now?",
        "Give me the current local time in {x}.",
        "Tell me the present time in {x}.",
        "What is the local time in {x} at this moment?",
        "Check the current clock time in {x}.",
    ]
    for i, loc in enumerate(TIME_LOCS):
        rows.append(t(f"final_t{i+1:02d}", time_templates[i % 5].format(x=loc), "get_time", i + 121))
    return rows


FINAL = build_cases()


def validate(train_cases):
    if len(FINAL) != 150:
        raise RuntimeError(f"expected 150 final cases, got {len(FINAL)}")
    counts = {label: 0 for label in probe.LABELS}
    for c in FINAL:
        counts[probe.label_of(c)] += 1
    if any(counts[label] != 30 for label in probe.LABELS):
        raise RuntimeError(f"final confirmation must be 30/class: {counts}")
    ids = [c["id"] for c in FINAL]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate final confirmation ids")
    forbidden = {c["id"] for c in train_cases} | {c["id"] for c in held.CASES} | {c["id"] for c in confirm.CONFIRM} | {c["id"] for c in third.THIRD}
    overlap = forbidden.intersection(ids)
    if overlap:
        raise RuntimeError(f"final confirmation id overlap: {sorted(overlap)}")
    return counts


def evaluate(model, tokenizer, all_train, binary_cases, family_cases):
    heads, wt = specialist.prepare_precision(model, tokenizer, all_train, binary_cases, family_cases)
    metrics = specialist.evaluate_suite(model, tokenizer, heads, wt, FINAL)
    return heads, wt, metrics


def exact(m):
    return (
        m["five_way_correct"] == 150
        and m["direct_vs_tool_correct"] == 150
        and m["tool_family_correct"] == 120
    )


def compact(m):
    return {
        "five_way": f"{m['five_way_correct']}/{m['total']}",
        "direct_tool": f"{m['direct_vs_tool_correct']}/{m['total']}",
        "tool_family": f"{m['tool_family_correct']}/{m['tool_family_total']}",
        "specialist_invocations": m["specialist_invocations"],
        "min_correct_margin": m["min_correct_margin"],
        "max_wrong_margin": m["max_wrong_margin"],
        "misses": [r for r in m["rows"] if not r["correct"]],
    }


def summary_md(report):
    lines = [
        "# Ember v0.0.54 frozen-router final confirmation",
        "",
        "150 untouched prompts, balanced 30/class. Architecture and training corpus were frozen before evaluation.",
        "Precision-specific full and INT4 router heads; Ember language weights unchanged.",
        "",
        "| Mode | 5-way | Direct/tool | Tool family | Specialist uses | Min correct margin | Exact |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in ("full", "int4"):
        m = report["routers"][key]["final"]
        lines.append(
            f"| {key} | {m['five_way_correct']}/150 | {m['direct_vs_tool_correct']}/150 | "
            f"{m['tool_family_correct']}/120 | {m['specialist_invocations']} | {m['min_correct_margin']:.6f} | {report['routers'][key]['exact']} |"
        )
    lines += ["", f"Strict final confirmation: **{report['strict_final_confirmation']}**", "", "## Misses", ""]
    misses = []
    for key in ("full", "int4"):
        for row in report["routers"][key]["final"]["rows"]:
            if not row["correct"]:
                misses.append((key, row))
    if not misses:
        lines.append("None.")
    else:
        by_id = {c["id"]: c for c in FINAL}
        for key, row in misses:
            c = by_id[row["id"]]
            lines.append(f"- **{key} / {row['id']}**: `{row['truth']}` → `{row['predicted']}`, margin={row['margin']:.6f}; {c['user']}")
    lines += ["", f"Interpretation: {report['interpretation']}", "", "No checkpoint, router integration, or production state changed.", ""]
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
    counts = validate(all_train)

    with tempfile.TemporaryDirectory(prefix="ember-router-final-") as td:
        work = Path(td)
        archive = ev.download_verified(ev.PACKAGE_URL, work / "ember.zip", ev.PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        import sys
        sys.path.insert(0, str(work / "src" / "ember"))
        api = HfApi(token=token)
        owner = api.whoami()["name"]
        repo = f"{owner}/{held.MODEL_NAME}"

        fm, ftok, _ = held.load_full(repo, work / "full", token)
        fm.eval()
        fh, fwt, fmetrics = evaluate(fm, ftok, all_train, binary_cases, family_cases)

        im, itok = held.load_int4(repo, work / "int4", token)
        im.eval()
        ih, iwt, imetrics = evaluate(im, itok, all_train, binary_cases, family_cases)

        fexact = exact(fmetrics)
        iexact = exact(imetrics)
        strict = fexact and iexact
        if strict:
            interpretation = (
                "The frozen precision-specific router architecture passed the untouched 150-case final confirmation exactly in full and INT4. "
                "Next: save router-head artifacts and test authoritative router-controlled generation without modifying Ember weights."
            )
        else:
            interpretation = (
                "The frozen router architecture missed at least one untouched final case. Do not tune against this final set or integrate the router; preserve the result and reassess architecture using training/development evidence only."
            )

        report = {
            "schema_version": 1,
            "diagnostic": "ember-v054-router-final-confirmation-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_repo": repo,
            "model_checkpoint_sha256": held.BEST_SHA256,
            "int4_checkpoint_sha256": held.INT4_SHA256,
            "final_count": len(FINAL),
            "final_counts": counts,
            "architecture_frozen_before_test": True,
            "training_frozen_before_test": True,
            "precision_specific_heads_required": True,
            "routers": {
                "full": {
                    "binary_cv": fh["binary_cv"],
                    "family_cv": fh["family_cv"],
                    "specialist_cv": fwt["cv"],
                    "final": fmetrics,
                    "exact": fexact,
                },
                "int4": {
                    "binary_cv": ih["binary_cv"],
                    "family_cv": ih["family_cv"],
                    "specialist_cv": iwt["cv"],
                    "final": imetrics,
                    "exact": iexact,
                },
            },
            "strict_final_confirmation": strict,
            "ember_weights_changed": False,
            "router_integrated": False,
            "production_changed": False,
            "interpretation": interpretation,
            "elapsed_seconds": time.monotonic() - started,
        }
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (OUT / "summary.md").write_text(summary_md(report), encoding="utf-8")
        print(json.dumps({
            "event": "router_final_confirmation_complete",
            "full": compact(fmetrics),
            "int4": compact(imetrics),
            "strict_final_confirmation": strict,
            "interpretation": interpretation,
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
