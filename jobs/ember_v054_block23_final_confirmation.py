"""Untouched final confirmation for the frozen Ember v0.0.54 block2+3 router.

Frozen before this test:
  * Ember v0.0.53 step-9 language weights;
  * precision-specific frozen router heads;
  * representation = concat(block_02, block_03);
  * expanded binary direct/tool head;
  * four-way family head;
  * weather-vs-get_time specialist;
  * training corpus and CV policy.

This file contains 200 new prompts, balanced 40/class. It does NOT import or
score the previously failed 150-case final confirmation. No tuning, threshold
selection, model-weight updates, router integration, or promotion occurs here.
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
from jobs import ember_v054_hierarchical_router as hier
from jobs import ember_v054_expanded_router as expanded
from jobs import ember_v054_representation_sweep as sweep

OUT = Path("v054-block23-final-confirmation")
REP = ("block_02", "block_03")

SYS_A = (
    "You are Ember. Answer ordinary writing, explanation, classification, comparison, planning, and summarization directly. "
    "Use weather only for live weather, calculator only for requested arithmetic, web_search only for current external facts, "
    "and get_time only for the current local time. Route by intent, not by keywords."
)
SYS_B = (
    "You are Ember. Available tools: weather, calculator, web_search, get_time. Mentions of words like weather, current, latest, "
    "search, time, or calculator do not by themselves require a tool. Use a tool only when its capability is actually required."
)
SYS_C = (
    "You are Ember. Choose between a direct response and exactly one of weather, calculator, web_search, or get_time. "
    "Direct tasks stay direct unless they require live data, arithmetic, current web information, or the present time."
)
SYSTEMS = (SYS_A, SYS_B, SYS_C)


def prompt(system: str, user: str) -> str:
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"


def direct(cid: str, user: str, i: int) -> dict:
    return {"id": cid, "kind": "direct_response", "user": user, "prompt": prompt(SYSTEMS[i % 3], user)}


def tool(cid: str, user: str, family: str, i: int) -> dict:
    return {"id": cid, "kind": "tool_call", "user": user, "expected_tool": family, "prompt": prompt(SYSTEMS[i % 3], user)}


DIRECT_TEXTS = [
    "Label the sentiment: The new navigation feels faster and easier to understand.",
    "Label the sentiment: The weather icon is too small in the prototype.",
    "Label the sentiment: The calculator panel appears beneath the form.",
    "Label the sentiment: The search layout still feels confusing.",
    "Label the sentiment: The current-time label looks much cleaner now.",
    "Rewrite this title professionally: latest weather card spacing ideas.",
    "Rewrite this bug title clearly: calculator history button opens wrong panel.",
    "Rewrite this heading: current local time preference settings.",
    "Shorten this label without searching: find latest documentation.",
    "Make this friendlier: Current status cannot be shown in this mockup.",
    "Summarize: the weather screen was restyled, snapshots passed, and behavior stayed unchanged.",
    "Summarize: calculator keyboard support was improved and all tests passed.",
    "Summarize: the time-zone menu moved into settings and accessibility checks passed.",
    "Summarize: the search page received new empty-state copy and review is complete.",
    "Summarize: the latest-release heading was renamed without changing the underlying data.",
    "Explain why weather forecasts become less certain farther into the future without checking live weather.",
    "Explain what a calculator memory register is without computing anything.",
    "Explain why time zones exist without telling me the current time anywhere.",
    "Explain what a search index does without performing a web search.",
    "Explain the difference between current and historical data without looking anything up.",
    "Compare a weather widget and a status badge as interface elements.",
    "Compare a calculator and a spreadsheet conceptually in two sentences.",
    "Compare a clock and a calendar without checking the current time.",
    "Compare a search result and a bookmark without browsing the web.",
    "Compare a release note and a changelog in two short sentences.",
    "Give three steps for testing a weather widget with fake data.",
    "Give two steps for testing calculator keyboard focus without solving a calculation.",
    "Give three steps for testing a time-zone dropdown without checking any clock.",
    "Give two steps for checking a search-box clear button without running a search.",
    "Give three steps for reviewing a settings-page layout change.",
    "Classify as command or statement: Check the weather icon alignment.",
    "Classify as question or statement: The calculator page has a history tab.",
    "Classify as positive or negative: The current-time display finally works well.",
    "Classify as command or statement: Search results appear below the filter bar.",
    "Classify the tone as formal or casual: Thanks for fixing the latest build so quickly!",
    "Write a one-sentence tooltip for weather display preferences.",
    "Write a friendly description of a calculator history screen.",
    "Write a concise tooltip for a time-zone selector.",
    "Write a friendly empty-state message for a search-results page.",
    "Write one sentence thanking a teammate for reviewing the current documentation.",
]

WEATHER_LOCS = [
    "Fort Wayne, Indiana", "Grand Rapids, Michigan", "Green Bay, Wisconsin", "Rochester, New York",
    "Buffalo, New York", "Syracuse, New York", "Harrisburg, Pennsylvania", "Richmond, Virginia",
    "Raleigh, North Carolina", "Columbia, South Carolina", "Jackson, Mississippi", "Baton Rouge, Louisiana",
    "Tulsa, Oklahoma", "Santa Fe, New Mexico", "Flagstaff, Arizona", "Bakersfield, California",
    "Sacramento, California", "Eugene, Oregon", "Tacoma, Washington", "Juneau, Alaska",
    "Whitehorse, Yukon", "Saskatoon, Saskatchewan", "Regina, Saskatchewan", "St. John's, Newfoundland",
    "Cork, Ireland", "Galway, Ireland", "Aberdeen, Scotland", "Inverness, Scotland",
    "Rotterdam, Netherlands", "Antwerp, Belgium", "Strasbourg, France", "Toulouse, France",
    "Nuremberg, Germany", "Dresden, Germany", "Wroclaw, Poland", "Poznan, Poland",
    "Skopje, North Macedonia", "Tirana, Albania", "Thessaloniki, Greece", "Chisinau, Moldova",
]

TIME_LOCS = [
    "Noumea", "Papeete", "Apia", "Port Moresby", "Perth", "Adelaide", "Brisbane", "Yangon",
    "Vientiane", "Colombo", "Male", "Astana", "Almaty", "Tashkent", "Bishkek", "Dushanbe",
    "Ashgabat", "Baghdad", "Damascus", "Ankara", "Pristina", "Podgorica", "Valletta", "Riga",
    "Vilnius", "Havana", "Kingston, Jamaica", "San Jose, Costa Rica", "Managua", "Tegucigalpa",
    "San Salvador", "Caracas", "Georgetown, Guyana", "Paramaribo", "Brasilia", "Maputo",
    "Harare", "Lusaka", "Kigali", "Antananarivo",
]

SEARCH_ITEMS = [
    "Apache Airflow", "Prefect", "Dagster", "Ruff", "mypy", "Black", "Hatch", "PDM",
    "Litestar", "Starlette", "Django Ninja", "SQLModel", "Alembic", "psycopg", "Uvicorn", "Gunicorn",
    "gRPC", "Protocol Buffers", "OpenSSL", "curl", "Wget", "Git LFS", "GNU Make", "Autoconf",
    "Clojure", "Haskell GHC", "OCaml", "Crystal", "Nim", "Lua", "LuaJIT", "Groovy",
    "Apache Spark", "Trino", "Apache Flink", "dbt Core", "Metabase", "Superset", "MinIO", "Ceph",
]


def build_cases() -> list[dict]:
    rows = [direct(f"b23final_d{i+1:02d}", text, i) for i, text in enumerate(DIRECT_TEXTS)]

    wt = [
        "What is the weather in {x} right now?",
        "Tell me the live temperature in {x}.",
        "Are there current rain conditions in {x}?",
        "How is the weather in {x} at the moment?",
        "Should I bring an umbrella in {x} based on the current weather?",
    ]
    for i, loc in enumerate(WEATHER_LOCS):
        rows.append(tool(f"b23final_w{i+1:02d}", wt[i % len(wt)].format(x=loc), "weather", i + 40))

    for i in range(40):
        mode = i % 5
        if mode == 0:
            text = f"Calculate {683 + i * 19} plus {127 + i * 7}."
        elif mode == 1:
            text = f"What is {31 + i} multiplied by {17 + i * 2}?"
        elif mode == 2:
            divisor = 8 + (i % 13)
            dividend = divisor * (173 + i * 4)
            text = f"Compute {dividend} divided by {divisor}."
        elif mode == 3:
            text = f"Calculate {11 + i} percent of {425 + i * 23}."
        else:
            text = f"What is {4 + (i % 8)} to the power of {3 + (i % 5)}?"
        rows.append(tool(f"b23final_c{i+1:02d}", text, "calculator", i + 80))

    st = [
        "Find the newest stable {x} release.",
        "What is the current official {x} version?",
        "Look up the latest stable {x} release.",
        "Find the most recent production release of {x}.",
        "What is the newest officially released {x} version?",
    ]
    for i, item in enumerate(SEARCH_ITEMS):
        rows.append(tool(f"b23final_s{i+1:02d}", st[i % len(st)].format(x=item), "web_search", i + 120))

    tt = [
        "What time is it in {x} right now?",
        "Give me the current local time in {x}.",
        "Tell me the present time in {x}.",
        "What is the local time in {x} at this moment?",
        "Check the current clock time in {x}.",
    ]
    for i, loc in enumerate(TIME_LOCS):
        rows.append(tool(f"b23final_t{i+1:02d}", tt[i % len(tt)].format(x=loc), "get_time", i + 160))
    return rows


FINAL = build_cases()


def validate(all_train: list[dict]) -> dict:
    if len(FINAL) != 200:
        raise RuntimeError(f"expected 200 final cases, got {len(FINAL)}")
    counts = {label: 0 for label in probe.LABELS}
    for c in FINAL:
        counts[probe.label_of(c)] += 1
    if any(counts[label] != 40 for label in probe.LABELS):
        raise RuntimeError(f"final set must be 40/class: {counts}")
    ids = [c["id"] for c in FINAL]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate final ids")
    train_ids = {c["id"] for c in all_train}
    overlap = train_ids.intersection(ids)
    if overlap:
        raise RuntimeError(f"final ids overlap training: {sorted(overlap)}")
    return counts


def fit_and_eval(model, tokenizer, all_train, binary_cases, family_cases):
    combined = all_train + FINAL
    rows, reps, _blocks, _head = probe.extract_representations(model, tokenizer, combined)
    if not all(rep in reps for rep in REP):
        raise RuntimeError(f"selected representation unavailable: {REP}, got {reps}")
    by_id = {r["id"]: r for r in rows}
    heads = sweep.fit_candidate(by_id, REP, binary_cases, family_cases)
    metrics = sweep.eval_suite(by_id, REP, heads, FINAL)
    return heads, metrics


def exact(m: dict) -> bool:
    return (
        int(m["five_way_correct"]) == 200
        and int(m["direct_vs_tool_correct"]) == 200
        and int(m["tool_family_correct"]) == 160
    )


def compact(m: dict) -> dict:
    return {
        "five_way": f"{m['five_way_correct']}/{m['total']}",
        "direct_tool": f"{m['direct_vs_tool_correct']}/{m['total']}",
        "tool_family": f"{m['tool_family_correct']}/{m['tool_family_total']}",
        "min_correct_margin": m["min_correct_margin"],
        "max_wrong_margin": m["max_wrong_margin"],
        "misses": [r for r in m["rows"] if not r["correct"]],
    }


def summary_md(report: dict) -> str:
    lines = [
        "# Ember v0.0.54 block2+3 frozen-router final confirmation",
        "",
        "200 untouched prompts, balanced 40/class. The previously failed 150-case final was not imported or scored.",
        "Representation, architecture, training corpus, and CV policy were frozen before this test.",
        "Ember language weights remained unchanged.",
        "",
        "| Mode | 5-way | Direct/tool | Tool family | Min correct margin | Exact |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in ("full", "int4"):
        m = report["routers"][key]["final"]
        lines.append(
            f"| {key} | {m['five_way_correct']}/200 | {m['direct_vs_tool_correct']}/200 | "
            f"{m['tool_family_correct']}/160 | {m['min_correct_margin']:.6f} | {report['routers'][key]['exact']} |"
        )
    lines += ["", f"Strict final confirmation: **{report['strict_final_confirmation']}**", "", "## Misses", ""]
    misses = []
    case_by_id = {c["id"]: c for c in FINAL}
    for key in ("full", "int4"):
        for row in report["routers"][key]["final"]["rows"]:
            if not row["correct"]:
                misses.append((key, row))
    if not misses:
        lines.append("None.")
    else:
        for key, row in misses:
            c = case_by_id[row["id"]]
            lines.append(
                f"- **{key} / {row['id']}**: `{row['truth']}` -> `{row['predicted']}`, "
                f"margin={row['margin']:.6f}; {c['user']}"
            )
    lines += ["", f"Interpretation: {report['interpretation']}", "", "No checkpoint, router integration, or production state changed.", ""]
    return "\n".join(lines)


def main() -> None:
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

    with tempfile.TemporaryDirectory(prefix="ember-block23-final-") as td:
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
        fh, fmetrics = fit_and_eval(fm, ftok, all_train, binary_cases, family_cases)

        im, itok = held.load_int4(repo, work / "int4", token)
        im.eval()
        ih, imetrics = fit_and_eval(im, itok, all_train, binary_cases, family_cases)

        fexact = exact(fmetrics)
        iexact = exact(imetrics)
        strict = fexact and iexact
        interpretation = (
            "The frozen precision-specific block2+3 router passed the untouched 200-case final exactly in full and INT4. "
            "Next: save router-head artifacts and test authoritative router-controlled generation while keeping Ember weights frozen."
            if strict else
            "The frozen block2+3 router missed at least one untouched final case. Preserve this result; do not tune or rerun against these cases."
        )

        def cv_meta(heads):
            return {
                "binary_cv": heads["binary_cv"],
                "family_cv": heads["family_cv"],
                "weather_time_cv": heads["weather_time_cv"],
                "pair_cv": heads["pair_cv"],
            }

        report = {
            "schema_version": 1,
            "diagnostic": "ember-v054-block23-final-confirmation-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_repo": repo,
            "model_checkpoint_sha256": held.BEST_SHA256,
            "int4_checkpoint_sha256": held.INT4_SHA256,
            "representation": list(REP),
            "final_count": len(FINAL),
            "final_counts": counts,
            "architecture_frozen_before_test": True,
            "training_frozen_before_test": True,
            "prior_failed_final_imported": False,
            "precision_specific_heads_required": True,
            "routers": {
                "full": {**cv_meta(fh), "final": fmetrics, "exact": fexact},
                "int4": {**cv_meta(ih), "final": imetrics, "exact": iexact},
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
            "event": "block23_final_confirmation_complete",
            "full": compact(fmetrics),
            "int4": compact(imetrics),
            "strict_final_confirmation": strict,
            "interpretation": interpretation,
            "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
