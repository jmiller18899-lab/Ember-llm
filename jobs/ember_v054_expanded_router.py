"""Expanded frozen hierarchical router for Ember v0.0.54.

Ember weights stay frozen. The binary block-3 gate is trained on 128 direct and
128 tool prompts. The family gate is trained on 64 prompts per tool family.
All added training prompts are distinct from the prior 20, 50, and 100 case
suites, which are now development evidence only.

No final confirmation is consumed here. If the expanded router clears all known
development suites in full and INT4, a fourth untouched confirmation is the next
step. No checkpoint, router integration, or production pointer is changed.
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
from jobs import ember_v054_routing_repair as v54

OUT = Path("v054-expanded-router")
REP = hier.REPRESENTATION
SYSTEM_X = (
    "You are Ember. Route the request by what must be done, not by words that appear in it. "
    "Answer writing, editing, explanation, summary, classification, comparison, and planning directly. "
    "Use weather only for live weather, calculator only for requested arithmetic, web_search only for current external facts, "
    "and get_time only for the present local time."
)
SYSTEMS = (v54.SYSTEM_1, v54.SYSTEM_2, v54.SYSTEM_3, SYSTEM_X)


def mk_prompt(system, user):
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"


def d(cid, text, i):
    return {"id": cid, "kind": "direct_response", "user": text, "prompt": mk_prompt(SYSTEMS[i % len(SYSTEMS)], text)}


def t(cid, text, family, i):
    return {"id": cid, "kind": "tool_call", "user": text, "prompt": mk_prompt(SYSTEMS[i % len(SYSTEMS)], text), "expected_tool": family}


def extra_directs():
    groups = [
        [
            "Label the sentiment: The patch fixed the issue and the team is pleased.",
            "Label the sentiment: The weather card is visible in the mockup.",
            "Label the sentiment: The calculator redesign is frustrating to use.",
            "Label the sentiment: The search page loaded exactly as expected.",
            "Label the sentiment: The time display moved to the footer.",
            "Label the sentiment: The latest build finally passed every test.",
            "Label the sentiment: The current status message is unclear.",
            "Label the sentiment: The release-note cleanup looks excellent.",
        ],
        [
            "Rewrite this title clearly: weather widget colors need cleanup.",
            "Rewrite this sentence professionally: calculator page feels cluttered.",
            "Shorten this heading: current local time display preferences.",
            "Polish this label: search latest docs button.",
            "Rewrite this bug title: live weather icon overlaps the title.",
            "Make this message friendlier: current status unavailable in the demo.",
            "Rewrite this heading: calculator shortcut documentation updates.",
            "Polish this sentence: web search layout needs more spacing.",
        ],
        [
            "Summarize: the weather mockup changed colors and all visual tests passed.",
            "Summarize: the calculator migration finished and no user behavior changed.",
            "Summarize: the time-zone selector moved and accessibility checks passed.",
            "Summarize: the search index is disabled only in this test environment.",
            "Summarize: the latest-build label changed but the code path stayed the same.",
            "Summarize: current-status wording was shortened and review is complete.",
            "Summarize: the weather page was renamed and snapshots were updated.",
            "Summarize: calculator tests passed, the branch is clean, and review can begin.",
        ],
        [
            "Explain why weather forecasts can disagree without checking any live forecast.",
            "Explain what a calculator memory key does without solving an equation.",
            "Explain why time zones use offsets without checking the current time.",
            "Explain what web-search ranking means without performing a search.",
            "Explain what the word current means in a status label.",
            "Explain what a software release means without finding the latest one.",
            "Explain the difference between weather and climate without live data.",
            "Explain what multiplication means without calculating a result.",
        ],
        [
            "Compare a weather app and a calendar app conceptually.",
            "Compare a calculator and a spreadsheet in two sentences.",
            "Compare a clock and a stopwatch without checking the time.",
            "Compare a web page and a search result without browsing.",
            "Compare JSON and TOML in two short sentences.",
            "Compare save and submit buttons in a form.",
            "Compare a status badge and a notification banner.",
            "Compare a release note and a commit message.",
        ],
        [
            "Give three steps for testing a weather-card layout using mock data.",
            "Give two steps for testing calculator-keyboard focus behavior.",
            "Give three steps for testing a time-zone dropdown without checking a clock.",
            "Give two steps for testing a search-box empty state without searching.",
            "Give three steps for checking a broken navigation link.",
            "Give two steps for testing a modal close button.",
            "Give three steps for checking a profile upload flow.",
            "Give two steps for reviewing a settings-page redesign.",
        ],
        [
            "Classify as command or statement: Check the weather icon alignment.",
            "Classify as question or statement: The calculator page has four tabs.",
            "Classify as positive or negative: The time picker is much easier now.",
            "Classify as command or statement: Search results appear under the header.",
            "Classify as a title or sentence: Current Status Overview.",
            "Classify as question or command: Review the latest build label.",
            "Classify the tone as formal or casual: Thanks for fixing that so quickly!",
            "Classify as warning or success: All checks passed.",
        ],
        [
            "Write a friendly tooltip for a weather-settings icon.",
            "Write one sentence describing a calculator settings page.",
            "Write a short tooltip for a time-zone selector.",
            "Write a friendly empty-state message for a search page.",
            "Write one sentence thanking someone for reviewing current documentation.",
            "Write a concise description of a release-notes panel.",
            "Write a friendly message asking someone to retry an upload.",
            "Write a one-sentence welcome for a new project member.",
        ],
    ]
    rows, n = [], 0
    for group in groups:
        for text in group:
            n += 1
            rows.append(d(f"xr_d{n:03d}", text, n))
    if len(rows) != 64:
        raise RuntimeError("extra direct count must be 64")
    return rows


WEATHER_LOCS = [
    "Fargo","Des Moines","Knoxville","Little Rock","El Paso","Fresno","Spokane","Wichita",
    "Toledo","Mobile","Norfolk","Sioux Falls","Bismarck","Cheyenne","Duluth","Madison",
    "Lexington","Providence","Hartford","Birmingham UK","Cardiff","Belfast","Lyon","Marseille",
    "Bern","Hamburg","Cologne","Florence","Turin","Porto","Seville","Bilbao",
    "Krakow","Gdansk","Brno","Bratislava","Ljubljana","Zagreb","Sarajevo","Sofia",
    "Bucharest","Belgrade","Tbilisi","Yerevan","Baku","Amman","Beirut","Alexandria Egypt",
]
TIME_LOCS = [
    "Doha","Riyadh","Muscat","Tehran","Karachi","Dhaka","Hanoi","Phnom Penh",
    "Bandar Seri Begawan","Ulaanbaatar","Busan","Fukuoka","Sapporo","Wellington","Suva","Honolulu",
    "Vancouver","Edmonton","Winnipeg","Kansas City","Memphis","Charlotte","Baltimore","Pittsburgh",
    "Boston","Detroit","Phoenix","Las Vegas","San Diego","San Jose","Guatemala City","Panama City",
    "Quito","La Paz","Asuncion","Recife","Salvador Brazil","Accra","Lagos","Addis Ababa",
    "Kampala","Dar es Salaam","Johannesburg","Cairo","Jerusalem","Kuwait City","Abu Dhabi","Kathmandu",
]
SEARCH_ITEMS = [
    "CockroachDB","HashiCorp Vault","Consul","Nomad","Bazel","Meson","Ninja build","Zig",
    "Elixir","Erlang OTP","Julia","R language","pandas","SciPy","Matplotlib","FastAPI",
    "Flask","Spring Boot",".NET","Java JDK","Apache Maven","Gradle","Yarn","pnpm",
    "Deno","Bun","Vite","React","Vue","Svelte","Next.js","Astro",
    "Hugo","Jekyll","RabbitMQ","Apache Kafka","Elasticsearch","OpenSearch","MongoDB","Apache Cassandra",
    "ClickHouse","DuckDB","MySQL","FreeBSD","Debian","Fedora","Rocky Linux","Arch Linux",
]


def extra_tools():
    rows = {k: [] for k in hier.FAMILY_LABELS}
    weather_templates = [
        "What is the weather in {x} right now?",
        "What are the current weather conditions in {x}?",
        "Give me the live temperature in {x}.",
        "Is it raining in {x} at the moment?",
    ]
    time_templates = [
        "What time is it in {x} right now?",
        "Give me the current local time in {x}.",
        "Tell me what time it is in {x} at the moment.",
        "What is the present time in {x}?",
    ]
    search_templates = [
        "Find the newest stable {x} release.",
        "What is the current stable {x} version?",
        "Look up the latest official {x} release.",
        "Find the most recent stable {x} release.",
    ]
    for i, loc in enumerate(WEATHER_LOCS):
        rows["weather"].append(t(f"xr_w{i+1:03d}", weather_templates[i % 4].format(x=loc), "weather", i))
    for i, loc in enumerate(TIME_LOCS):
        rows["get_time"].append(t(f"xr_t{i+1:03d}", time_templates[i % 4].format(x=loc), "get_time", i+1))
    for i, item in enumerate(SEARCH_ITEMS):
        rows["web_search"].append(t(f"xr_s{i+1:03d}", search_templates[i % 4].format(x=item), "web_search", i+2))

    calc = []
    for i in range(12):
        a, b = 317 + i * 29, 41 + i * 3
        calc.append(f"Calculate {a} plus {b}.")
    for i in range(12):
        a, b = 23 + i * 4, 31 + i * 2
        calc.append(f"What is {a} multiplied by {b}?")
    for i in range(12):
        b = 6 + i
        a = b * (137 + i * 5)
        calc.append(f"Compute {a} divided by {b}.")
    for i in range(12):
        pct, value = 6 + i * 2, 220 + i * 35
        calc.append(f"Calculate {pct} percent of {value}.")
    for i, text in enumerate(calc):
        rows["calculator"].append(t(f"xr_c{i+1:03d}", text, "calculator", i+3))

    for family, items in rows.items():
        if len(items) != 48:
            raise RuntimeError(f"{family} extra count must be 48, got {len(items)}")
    return rows


def build_training():
    base = hier.training_cases()  # 64 direct + 16/family tools.
    xd = extra_directs()
    xt = extra_tools()
    base_direct = [c for c in base if probe.label_of(c) == "direct"]
    base_tools = {f: [c for c in base if probe.label_of(c) == f] for f in hier.FAMILY_LABELS}
    binary = base_direct + xd
    for f in hier.FAMILY_LABELS:
        binary += base_tools[f] + xt[f][:16]
    family = []
    for f in hier.FAMILY_LABELS:
        family += base_tools[f] + xt[f]
    all_cases = base_direct + xd + family
    ids = [c["id"] for c in all_cases]
    if len(ids) != len(set(ids)):
        raise RuntimeError("expanded training ids are not unique")
    forbidden = {c["id"] for c in held.CASES} | {c["id"] for c in confirm.CONFIRM} | {c["id"] for c in third.THIRD}
    if forbidden.intersection(ids):
        raise RuntimeError("expanded training overlaps development ids")
    if sum(probe.label_of(c) == "direct" for c in binary) != 128 or sum(probe.label_of(c) != "direct" for c in binary) != 128:
        raise RuntimeError("binary training must be 128 direct / 128 tool")
    for f in hier.FAMILY_LABELS:
        if sum(probe.label_of(c) == f for c in family) != 64:
            raise RuntimeError(f"family training must be 64 for {f}")
    return all_cases, binary, family


def rows_for(rows_by_id, cases):
    return [rows_by_id[c["id"]] for c in cases]


def fit_expanded(all_rows, binary_cases, family_cases):
    by_id = {r["id"]: r for r in all_rows}
    rb = rows_for(by_id, binary_cases)
    rf = rows_for(by_id, family_cases)
    Xb, labels_b = hier.stack(rb, REP)
    yb = torch.tensor([0 if x == "direct" else 1 for x in labels_b], dtype=torch.long)
    bcv, brecs = hier.cv_ridge(Xb, yb, 2)
    bstate = hier.fit_ridge(Xb, yb, 2, bcv["ridge"])
    Xf, labels_f = hier.stack(rf, REP)
    yf = torch.tensor([hier.FAMILY_LABELS.index(x) for x in labels_f], dtype=torch.long)
    fcv, frecs = hier.cv_ridge(Xf, yf, 4)
    fstate = hier.fit_ridge(Xf, yf, 4, fcv["ridge"])
    return {"binary": bstate, "family": fstate, "binary_cv": bcv, "binary_cv_records": brecs, "family_cv": fcv, "family_cv_records": frecs}


def eval_suite(heads, rows, cases):
    X, _ = hier.stack(rows, REP)
    pred, margins = hier.hierarchical_predict(heads, X)
    return hier.score(pred, rows, cases, margins)


def compact(m):
    return {
        "five_way": f"{m['five_way_correct']}/{m['total']}",
        "direct_tool": f"{m['direct_vs_tool_correct']}/{m['total']}",
        "tool_family": f"{m['tool_family_correct']}/{m['tool_family_total']}",
        "min_correct_margin": m["min_correct_margin"],
        "max_wrong_margin": m["max_wrong_margin"],
        "misses": [r for r in m["rows"] if not r["correct"]],
    }


def all_exact(suites):
    return all(m["five_way_correct"] == m["total"] for m in suites.values())


def evaluate_precision(model, tok, all_train, binary_cases, family_cases):
    train_rows = hier.extract(model, tok, all_train)
    heads = fit_expanded(train_rows, binary_cases, family_cases)
    suites = {
        "old20": eval_suite(heads, hier.extract(model, tok, held.CASES), held.CASES),
        "confirm50": eval_suite(heads, hier.extract(model, tok, confirm.CONFIRM), confirm.CONFIRM),
        "third100": eval_suite(heads, hier.extract(model, tok, third.THIRD), third.THIRD),
    }
    return heads, suites


def summary_md(report):
    lines = [
        "# Ember v0.0.54 expanded hierarchical frozen router", "",
        "Ember weights are frozen. Binary gate: 128 direct / 128 tool. Family head: 64 per family.",
        "Prior 20 + 50 + 100 suites are development-only. No final confirmation is consumed here.", "",
        "| Mode | Binary CV | Family CV | Old20 | Confirm50 | Third100 | Exact all dev |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in ("full", "int4", "full_heads_on_int4"):
        row = report["routers"][key]
        bcv = row.get("binary_cv")
        fcv = row.get("family_cv")
        b = "—" if bcv is None else f"{bcv['accuracy']:.1%}"
        f = "—" if fcv is None else f"{fcv['accuracy']:.1%}"
        s = row["suites"]
        lines.append(f"| {key} | {b} | {f} | {s['old20']['five_way_correct']}/20 | {s['confirm50']['five_way_correct']}/50 | {s['third100']['five_way_correct']}/100 | {row['exact_all_dev']} |")
    lines += ["", f"Strict precision-specific development pass: **{report['strict_dev_pass']}**", f"Shared full→INT4 development pass: **{report['shared_dev_pass']}**", "", f"Interpretation: {report['interpretation']}", "", "No checkpoint, router integration, or production state changed.\n"]
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
    all_train, binary_cases, family_cases = build_training()

    with tempfile.TemporaryDirectory(prefix="ember-expanded-router-") as td:
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
        fh, fs = evaluate_precision(fm, ftok, all_train, binary_cases, family_cases)

        im, itok = held.load_int4(repo, work / "int4", token)
        im.eval()
        ih, ins = evaluate_precision(im, itok, all_train, binary_cases, family_cases)

        # Shared full heads on INT4 development hidden states.
        cross = {
            "old20": eval_suite(fh, hier.extract(im, itok, held.CASES), held.CASES),
            "confirm50": eval_suite(fh, hier.extract(im, itok, confirm.CONFIRM), confirm.CONFIRM),
            "third100": eval_suite(fh, hier.extract(im, itok, third.THIRD), third.THIRD),
        }

        full_exact = all_exact(fs)
        int4_exact = all_exact(ins)
        cross_exact = all_exact(cross)
        strict = full_exact and int4_exact
        if strict:
            interpretation = "Expanded precision-specific frozen routers clear all 170 known development prompts exactly. Next: run a fourth untouched confirmation before saving or integrating any router."
        else:
            interpretation = "Expanded frozen router still has known development misses. Inspect only those misses; do not consume a new final confirmation yet."

        report = {
            "schema_version": 1,
            "diagnostic": "ember-v054-expanded-hierarchical-router-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_repo": repo,
            "representation": REP,
            "binary_training_count": len(binary_cases),
            "family_training_count": len(family_cases),
            "all_training_count": len(all_train),
            "routers": {
                "full": {"binary_cv": fh["binary_cv"], "family_cv": fh["family_cv"], "suites": fs, "exact_all_dev": full_exact},
                "int4": {"binary_cv": ih["binary_cv"], "family_cv": ih["family_cv"], "suites": ins, "exact_all_dev": int4_exact},
                "full_heads_on_int4": {"binary_cv": None, "family_cv": None, "suites": cross, "exact_all_dev": cross_exact},
            },
            "strict_dev_pass": strict,
            "shared_dev_pass": cross_exact,
            "ember_weights_changed": False,
            "router_integrated": False,
            "production_changed": False,
            "fresh_final_confirmation_consumed": False,
            "interpretation": interpretation,
            "elapsed_seconds": time.monotonic() - started,
        }
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (OUT / "summary.md").write_text(summary_md(report))
        print(json.dumps({
            "event": "expanded_router_complete",
            "full": {k: compact(v) for k, v in fs.items()},
            "int4": {k: compact(v) for k, v in ins.items()},
            "full_heads_on_int4": {k: compact(v) for k, v in cross.items()},
            "full_binary_cv": fh["binary_cv"], "full_family_cv": fh["family_cv"],
            "int4_binary_cv": ih["binary_cv"], "int4_family_cv": ih["family_cv"],
            "strict_dev_pass": strict, "shared_dev_pass": cross_exact,
            "interpretation": interpretation,
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
