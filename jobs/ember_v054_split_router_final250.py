"""Untouched 250-case final confirmation for Ember v0.0.54 split router.

Frozen architecture before this test:
  * Ember v0.0.53 step-9 weights unchanged;
  * precision-specific ridge heads;
  * direct/tool gate representation = block_04;
  * tool-family + weather/time representation = block_02 + block_03;
  * expanded training corpus and train-only CV policy.

This file defines 250 new prompts, 50/class. It does not import or score either
previously failed final suite. No tuning, integration, checkpoint save, or
production change occurs here.
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
from jobs import ember_v054_expanded_router as expanded
from jobs import ember_v054_split_representation_router as split

OUT = Path("v054-split-router-final250")

S0 = (
    "You are Ember. Answer writing, editing, classification, explanation, comparison, planning, and summarization directly. "
    "Use weather for live weather, calculator for requested arithmetic, web_search for current external facts, and get_time for the present local time."
)
S1 = (
    "You are Ember. Tools are weather, calculator, web_search, get_time. Route by the requested action rather than by words in the sentence. "
    "A mention of weather, search, current, latest, calculator, clock, or time does not itself require a tool."
)
S2 = (
    "You are Ember. Choose direct response unless live weather, arithmetic, current web information, or current local time is actually required; then choose exactly the matching tool."
)
SYSTEMS = (S0, S1, S2)


def p(user: str, i: int) -> str:
    return f"<|system|>\n{SYSTEMS[i % len(SYSTEMS)]}\n<|user|>\n{user}\n<|assistant|>\n"


def d(cid: str, text: str, i: int) -> dict:
    return {"id": cid, "kind": "direct_response", "user": text, "prompt": p(text, i)}


def t(cid: str, text: str, family: str, i: int) -> dict:
    return {"id": cid, "kind": "tool_call", "user": text, "expected_tool": family, "prompt": p(text, i)}


DIRECT = [
    "Label the sentiment: The redesigned dashboard feels polished and responsive.",
    "Label the sentiment: The weather settings page is difficult to navigate.",
    "Label the sentiment: The calculator help text is clear and useful.",
    "Label the sentiment: The search button sits beside the page title.",
    "Label the sentiment: The clock icon looks slightly misaligned.",
    "Label the sentiment: The latest build passed all of its checks.",
    "Label the sentiment: The current-status wording is too vague.",
    "Label the sentiment: The release documentation looks excellent.",
    "Label the sentiment: The time-zone menu is frustrating to use.",
    "Label the sentiment: The search layout finally looks consistent.",
    "Rewrite this bug title clearly: weather card jumps around after resize.",
    "Rewrite this heading professionally: calculator keyboard help stuff.",
    "Rewrite this label more clearly: current time display options.",
    "Shorten this title without searching: find latest release information.",
    "Rewrite this sentence: search page thing is broken on small screens.",
    "Make this message friendlier: Current status is unavailable in the demo.",
    "Rewrite this heading: weather and clock display preferences.",
    "Polish this bug title: calculator result text overlaps footer.",
    "Rewrite this button label: search current documentation.",
    "Make this sentence more professional: the latest screen looks way better.",
    "Explain what a weather forecast represents without checking live conditions.",
    "Explain what a calculator memory function does without solving an expression.",
    "Explain what a time zone offset means without checking the current time.",
    "Explain what a web-search query is without performing a search.",
    "Explain the difference between current and archived information without looking anything up.",
    "Explain why a cached temperature value may become stale without checking the weather.",
    "Explain what percentage means without calculating a specific percentage.",
    "Explain why clocks can show different local times in different places without checking any clock.",
    "Explain what search ranking means in plain English without browsing.",
    "Explain the difference between a software version and a release date without finding the latest one.",
    "Summarize: the weather mockup was restyled and every visual test passed.",
    "Summarize: calculator shortcuts were documented and the review is complete.",
    "Summarize: the time-zone selector moved into preferences and behavior stayed unchanged.",
    "Summarize: search-page copy was simplified and all snapshots passed.",
    "Summarize: the current-status component was renamed without changing its data source.",
    "Compare a weather widget and a notification badge in two sentences.",
    "Compare a calculator and a spreadsheet without doing arithmetic.",
    "Compare a wall clock and a stopwatch without checking the time.",
    "Compare a search result and a browser bookmark without browsing.",
    "Compare a release note and a README in two concise sentences.",
    "Give three steps for testing a weather widget using sample data.",
    "Give two steps for testing calculator keyboard focus without calculating anything.",
    "Give three steps for testing a time-zone menu without checking current time.",
    "Give two steps for testing a search-field clear button without running a search.",
    "Give three steps for reviewing a current-status component in a mock interface.",
    "Classify as command or statement: Align the weather icon with the heading.",
    "Classify as question or statement: The calculator panel has a history section.",
    "Classify the tone as positive, negative, or neutral: The latest layout is much easier to read.",
    "Write a friendly tooltip for a search-preferences button.",
    "Write one sentence describing a clock-display settings panel.",
]

WEATHER = [
    "Akron, Ohio","Dayton, Ohio","Erie, Pennsylvania","Allentown, Pennsylvania","Roanoke, Virginia",
    "Greensboro, North Carolina","Augusta, Georgia","Pensacola, Florida","Huntsville, Alabama","Shreveport, Louisiana",
    "Lubbock, Texas","Amarillo, Texas","Colorado Springs, Colorado","Provo, Utah","Reno, Nevada",
    "Stockton, California","Santa Rosa, California","Salem, Oregon","Bellingham, Washington","Kelowna, British Columbia",
    "Red Deer, Alberta","Thunder Bay, Ontario","Windsor, Ontario","Moncton, New Brunswick","Charlottetown, Prince Edward Island",
    "Limerick, Ireland","Dundee, Scotland","Leeds, England","Bristol, England","The Hague, Netherlands",
    "Ghent, Belgium","Lille, France","Montpellier, France","Stuttgart, Germany","Leipzig, Germany",
    "Lodz, Poland","Cluj-Napoca, Romania","Varna, Bulgaria","Split, Croatia","Novi Sad, Serbia",
    "Graz, Austria","Basel, Switzerland","Genoa, Italy","Palermo, Italy","Malaga, Spain",
    "Coimbra, Portugal","Gothenburg, Sweden","Bergen, Norway","Turku, Finland","Kaunas, Lithuania",
]

TIME = [
    "Fiji","Tonga","Guam","Saipan","Palau","Dili","Bandung","Surabaya","Chiang Mai","Luang Prabang",
    "Chittagong","Hyderabad, India","Pune","Ahmedabad","Jaipur","Lahore","Islamabad","Kabul","Samarkand","Bukhara",
    "Atyrau","Batumi","Erbil","Jeddah","Medina","Aden","Manama","Nicosia","Minsk","Krakow",
    "Bratislava","Ljubljana","Zagreb","Belgrade","Sofia","Odessa","Kishinev","Reykjavik","Bermuda","Nassau",
    "Port of Spain","Bridgetown","Belize City","Guayaquil","Medellin","Cusco","Cordoba, Argentina","Rosario, Argentina","Montego Bay","Punta Cana",
]

SEARCH = [
    "Apache Beam","Ray","Dask","Polars","Pillow","Requests","httpx","Typer","Click","Rich",
    "Sphinx","MkDocs","Jinja","Werkzeug","pytest-cov","tox","nox","pre-commit","Bandit","pip-audit",
    "OpenTelemetry","Jaeger","Zipkin","Vector","Fluent Bit","Loki","Tempo","VictoriaMetrics","Zabbix","Nagios",
    "HAProxy","Traefik","Envoy","Cilium","Calico","Istio","Linkerd","K3s","MicroK8s","OpenShift",
    "Pulumi","Packer","Vagrant","Chef","Puppet","Salt","GNU Bash","Zsh","Fish shell","tmux",
]


def build() -> list[dict]:
    rows = [d(f"splitfinal_d{i+1:02d}", text, i) for i, text in enumerate(DIRECT)]
    wtpl = [
        "What is the weather in {x} right now?",
        "Give me the live temperature in {x}.",
        "Is it raining in {x} at the moment?",
        "Tell me the current weather conditions in {x}.",
        "Should I bring an umbrella in {x} based on the live weather?",
    ]
    for i, loc in enumerate(WEATHER):
        rows.append(t(f"splitfinal_w{i+1:02d}", wtpl[i % 5].format(x=loc), "weather", i + 50))

    for i in range(50):
        mode = i % 5
        if mode == 0:
            text = f"Calculate {751 + i * 23} plus {149 + i * 11}."
        elif mode == 1:
            text = f"What is {34 + i} multiplied by {21 + i * 3}?"
        elif mode == 2:
            divisor = 9 + (i % 14)
            text = f"Compute {divisor * (189 + i * 5)} divided by {divisor}."
        elif mode == 3:
            text = f"Calculate {13 + i} percent of {510 + i * 27}."
        else:
            text = f"What is {3 + (i % 10)} to the power of {2 + (i % 6)}?"
        rows.append(t(f"splitfinal_c{i+1:02d}", text, "calculator", i + 100))

    stpl = [
        "Find the newest stable {x} release.",
        "What is the current official {x} version?",
        "Look up the latest stable {x} release.",
        "Find the most recent production release of {x}.",
        "What is the newest officially released {x} version?",
    ]
    for i, item in enumerate(SEARCH):
        rows.append(t(f"splitfinal_s{i+1:02d}", stpl[i % 5].format(x=item), "web_search", i + 150))

    ttpl = [
        "What time is it in {x} right now?",
        "Give me the current local time in {x}.",
        "Tell me the present time in {x}.",
        "What is the local time in {x} at this moment?",
        "Check the current clock time in {x}.",
    ]
    for i, loc in enumerate(TIME):
        rows.append(t(f"splitfinal_t{i+1:02d}", ttpl[i % 5].format(x=loc), "get_time", i + 200))
    return rows


FINAL = build()


def validate(all_train):
    if len(FINAL) != 250:
        raise RuntimeError(f"expected 250 cases, got {len(FINAL)}")
    counts = {label: 0 for label in probe.LABELS}
    for c in FINAL:
        counts[probe.label_of(c)] += 1
    if any(counts[label] != 50 for label in probe.LABELS):
        raise RuntimeError(f"final must be 50/class: {counts}")
    ids = [c["id"] for c in FINAL]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate final ids")
    overlap = {c["id"] for c in all_train}.intersection(ids)
    if overlap:
        raise RuntimeError(f"final ids overlap training: {sorted(overlap)}")
    return counts


def fit_eval(model, tokenizer, all_train, binary_cases, family_cases):
    combined = all_train + FINAL
    rows, reps, _blocks, _head = probe.extract_representations(model, tokenizer, combined)
    needed = set(split.BINARY_REP + split.FAMILY_REP)
    if not needed.issubset(set(reps)):
        raise RuntimeError(f"required representations unavailable: {sorted(needed)} vs {reps}")
    by_id = {r["id"]: r for r in rows}
    heads = split.fit_heads(by_id, binary_cases, family_cases)
    metrics = split.eval_suite(by_id, FINAL, heads)
    return heads, metrics


def exact(m):
    return m["five_way_correct"] == 250 and m["direct_vs_tool_correct"] == 250 and m["tool_family_correct"] == 200


def compact(m):
    return {
        "five_way": f"{m['five_way_correct']}/{m['total']}",
        "direct_tool": f"{m['direct_vs_tool_correct']}/{m['total']}",
        "tool_family": f"{m['tool_family_correct']}/{m['tool_family_total']}",
        "min_correct_margin": m["min_correct_margin"],
        "max_wrong_margin": m["max_wrong_margin"],
        "misses": [r for r in m["rows"] if not r["correct"]],
    }


def summary_md(report):
    lines = [
        "# Ember v0.0.54 split-router final confirmation", "",
        "250 untouched prompts, balanced 50/class. Both earlier failed final suites were excluded.",
        "Architecture, representations, training corpus, and CV policy were frozen before evaluation.", "",
        "| Mode | 5-way | Direct/tool | Tool family | Min correct margin | Exact |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in ("full", "int4"):
        m = report["routers"][key]["final"]
        lines.append(
            f"| {key} | {m['five_way_correct']}/250 | {m['direct_vs_tool_correct']}/250 | "
            f"{m['tool_family_correct']}/200 | {m['min_correct_margin']:.6f} | {report['routers'][key]['exact']} |"
        )
    lines += ["", f"Strict final confirmation: **{report['strict_final_confirmation']}**", "", "## Misses", ""]
    by_id = {c["id"]: c for c in FINAL}
    misses = []
    for key in ("full", "int4"):
        for row in report["routers"][key]["final"]["rows"]:
            if not row["correct"]:
                misses.append((key, row))
    if not misses:
        lines.append("None.")
    else:
        for key, row in misses:
            lines.append(f"- **{key} / {row['id']}**: `{row['truth']}` -> `{row['predicted']}`, margin={row['margin']:.6f}; {by_id[row['id']]['user']}")
    lines += ["", f"Interpretation: {report['interpretation']}", "", "No router artifact, checkpoint, integration, or production state changed.", ""]
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

    with tempfile.TemporaryDirectory(prefix="ember-split-final250-") as td:
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
        fh, fmetrics = fit_eval(fm, ftok, all_train, binary_cases, family_cases)

        im, itok = held.load_int4(repo, work / "int4", token)
        im.eval()
        ih, imetrics = fit_eval(im, itok, all_train, binary_cases, family_cases)

        fexact, iexact = exact(fmetrics), exact(imetrics)
        strict = fexact and iexact
        interpretation = (
            "The frozen split-representation router passed the untouched 250-case confirmation exactly in full and INT4. Next: save precision-specific router-head artifacts and test authoritative router-controlled generation while Ember weights remain frozen."
            if strict else
            "The frozen split-representation router missed at least one untouched final case. Preserve this result and do not tune or rerun against these prompts."
        )
        report = {
            "schema_version":1,
            "diagnostic":"ember-v054-split-router-final250-v1",
            "created_at":datetime.now(timezone.utc).isoformat(),
            "model_repo":repo,
            "model_checkpoint_sha256":held.BEST_SHA256,
            "int4_checkpoint_sha256":held.INT4_SHA256,
            "binary_representation":list(split.BINARY_REP),
            "family_representation":list(split.FAMILY_REP),
            "final_count":250,
            "final_counts":counts,
            "architecture_frozen_before_test":True,
            "training_frozen_before_test":True,
            "failed_final_150_imported":False,
            "failed_final_200_imported":False,
            "routers":{
                "full":{"binary_cv":fh["binary_cv"],"family_cv":fh["family_cv"],"weather_time_cv":fh["weather_time_cv"],"final":fmetrics,"exact":fexact},
                "int4":{"binary_cv":ih["binary_cv"],"family_cv":ih["family_cv"],"weather_time_cv":ih["weather_time_cv"],"final":imetrics,"exact":iexact},
            },
            "strict_final_confirmation":strict,
            "ember_weights_changed":False,
            "router_integrated":False,
            "production_changed":False,
            "interpretation":interpretation,
            "elapsed_seconds":time.monotonic()-started,
        }
        (OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        (OUT/"summary.md").write_text(summary_md(report),encoding="utf-8")
        print(json.dumps({
            "event":"split_router_final250_complete",
            "full":compact(fmetrics),"int4":compact(imetrics),
            "strict_final_confirmation":strict,
            "interpretation":interpretation,
            "elapsed_seconds":report["elapsed_seconds"],
        }),flush=True)


if __name__ == "__main__":
    main()
