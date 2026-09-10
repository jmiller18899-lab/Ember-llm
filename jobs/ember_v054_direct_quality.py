"""Strict first-turn direct-response quality challenge for Ember v0.0.54 INT4.

The tool path (frozen router + extractive resolver v2) has already cleared exact
routing and 120/120 argument extraction. The remaining promotion blocker is the
quality of DIRECT responses from Ember's INT4 language model.

This diagnostic uses 24 fresh direct-only prompts across six task families:
  greeting/writing, rewrite, explain, summarize, classify/sentiment, plan/compare.

For every case:
- the frozen INT4 multi-layer router must choose DIRECT;
- tool/system/user/assistant markers are masked from generation;
- only the first turn is decoded and generation stops at <|endoftext|>;
- deterministic task-specific semantic checks replace the old lenient
  "nonempty and no tool call" evaluator;
- obvious training-leak/repetition patterns fail quality even if keywords match.

No LLM judge is used. Ember weights, router weights, checkpoints, and production
pointers are unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
import time
import zipfile

from huggingface_hub import HfApi
import torch

from jobs import ember_hf_eval as ev
from jobs import ember_v053_heldout_routing as held
from jobs import ember_v054_frozen_router_probe as probe
from jobs import ember_v054_router_calibration as cal
from jobs import ember_v054_multilayer_router as multi
from jobs import ember_v054_int4_controlled_generation as control

OUT = Path("v054-direct-quality")
MAX_TOKENS = 64

SYSTEM = (
    "You are Ember. Answer the user's request directly and concisely. The request does not need a tool. "
    "Return only the answer to the current user turn."
)


def prompt(user: str) -> str:
    return f"<|system|>\n{SYSTEM}\n<|user|>\n{user}\n<|assistant|>\n"


def case(cid: str, family: str, user: str, check: dict) -> dict:
    return {"id": cid, "kind": "direct_response", "family": family, "user": user, "prompt": prompt(user), "check": check}


CASES = [
    # Friendly/writing.
    case("dq_greet_1", "writing", "Say hello in one friendly sentence.", {"any_terms": [["hello", "hi", "hey"]], "max_words": 20}),
    case("dq_greet_2", "writing", "Write one short sentence thanking someone for reviewing my code.", {"any_terms": [["thank", "thanks", "appreciate"], ["review", "code"]], "max_words": 24}),
    case("dq_write_3", "writing", "Write a friendly one-sentence message saying the upload succeeded.", {"any_terms": [["upload"], ["success", "succeeded", "complete", "completed"]], "max_words": 24}),
    case("dq_write_4", "writing", "Write a short tooltip for a refresh button.", {"any_terms": [["refresh", "reload", "update"]], "max_words": 18}),

    # Rewrite/title.
    case("dq_rewrite_1", "rewrite", "Rewrite as a clear title: checkout button broken on mobile.", {"all_terms": ["checkout", "mobile"], "any_terms": [["broken", "fix", "issue", "failure", "problem"]], "max_words": 12}),
    case("dq_rewrite_2", "rewrite", "Rewrite as a professional title: settings page loads really slow.", {"all_terms": ["settings"], "any_terms": [["slow", "performance", "loading", "load"]], "max_words": 12}),
    case("dq_rewrite_3", "rewrite", "Shorten this heading: user profile picture upload accessibility improvements.", {"any_terms": [["profile"], ["upload", "picture", "image"], ["accessibility", "accessible"]], "max_words": 12}),
    case("dq_rewrite_4", "rewrite", "Rewrite clearly: latest build notes need cleanup.", {"all_terms": ["build"], "any_terms": [["notes", "release"], ["cleanup", "clean", "improve"]], "max_words": 12}),

    # Explanations.
    case("dq_explain_1", "explain", "Explain what a model checkpoint is in one sentence.", {"any_terms": [["save", "saved", "snapshot"], ["model", "training", "state", "weights"]], "max_words": 36}),
    case("dq_explain_2", "explain", "Explain what a cache does in one sentence.", {"any_terms": [["store", "stores", "stored", "save", "keeps"], ["fast", "faster", "quick", "reuse", "access"]], "max_words": 36}),
    case("dq_explain_3", "explain", "Explain why software tests are useful in one sentence.", {"any_terms": [["bug", "error", "problem", "issue", "correct"], ["find", "catch", "detect", "prevent", "verify"]], "max_words": 36}),
    case("dq_explain_4", "explain", "Explain the difference between a file and a folder in one sentence.", {"all_terms": ["file", "folder"], "any_terms": [["contain", "contains", "holds", "stores"], ["data", "files", "item", "items"]], "max_words": 40}),

    # Summaries.
    case("dq_summary_1", "summarize", "Summarize in one sentence: The tests passed, the deployment completed, and no errors were reported.", {"any_terms": [["test"], ["deploy"], ["pass", "success", "error", "errors"]], "max_words": 28}),
    case("dq_summary_2", "summarize", "Summarize: The sidebar was moved to the left, its labels were shortened, and the underlying behavior did not change.", {"any_terms": [["sidebar"], ["left", "move"], ["label"], ["behavior", "function", "functionality", "unchanged"]], "max_words": 32}),
    case("dq_summary_3", "summarize", "Summarize in one sentence: The API latency dropped from 300 ms to 120 ms after the optimization.", {"any_terms": [["latency", "speed", "faster"], ["300"], ["120"], ["optim", "improve", "drop", "reduc"]], "max_words": 30}),
    case("dq_summary_4", "summarize", "Summarize: The backup finished successfully, but the notification arrived five minutes late.", {"any_terms": [["backup"], ["success", "finished", "complete"], ["notification"], ["late", "delay", "five", "5"]], "max_words": 30}),

    # Classification/sentiment: require the requested label.
    case("dq_class_1", "classify", "Label the sentiment as positive, negative, or neutral: The new interface is much easier to use.", {"label": "positive", "forbidden_labels": ["negative", "neutral"], "max_words": 12}),
    case("dq_class_2", "classify", "Label the sentiment as positive, negative, or neutral: The app crashes every time I save.", {"label": "negative", "forbidden_labels": ["positive", "neutral"], "max_words": 12}),
    case("dq_class_3", "classify", "Classify this as a question or statement: The service is online.", {"label": "statement", "forbidden_labels": ["question"], "max_words": 12}),
    case("dq_class_4", "classify", "Classify this as success, warning, or error: Deployment failed because the configuration is invalid.", {"label": "error", "forbidden_labels": ["success", "warning"], "max_words": 12}),

    # Planning/comparison.
    case("dq_plan_1", "plan_compare", "Give exactly two short steps for testing a login form.", {"min_steps": 2, "max_words": 45, "any_terms": [["login", "credential", "password", "username"], ["test", "verify", "check", "submit"]]}),
    case("dq_plan_2", "plan_compare", "Give exactly two short steps for checking whether a link works.", {"min_steps": 2, "max_words": 45, "any_terms": [["click", "open", "link"], ["verify", "check", "page", "destination"]]}),
    case("dq_compare_3", "plan_compare", "Compare JSON and CSV in two concise sentences.", {"all_terms": ["json", "csv"], "any_terms": [["structure", "structured", "key", "nested", "table", "row", "column"]], "max_words": 48}),
    case("dq_compare_4", "plan_compare", "Compare a browser tab and a browser window in two concise sentences.", {"all_terms": ["tab", "window"], "any_terms": [["browser"], ["multiple", "contain", "inside", "separate"]], "max_words": 48}),
]

LEAK_PATTERNS = [
    "what is the difference between",
    "prepare a controlled example",
    "practical mechanism used to make a system easier",
    "recent webassembly announcements",
    "the current kubernet",
]


def normalized(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9.+-]+", " ", text.casefold()).split())


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))


def generic_quality(text: str) -> dict:
    visible = ev.visible_text(text).strip()
    norm = normalized(visible)
    words = re.findall(r"\b[a-z0-9]+\b", norm)
    repeated_triplet = bool(re.search(r"\b([a-z0-9]+)(?:\s+\1){2,}\b", norm))
    unique_ratio = len(set(words)) / len(words) if words else 0.0
    leak = next((p for p in LEAK_PATTERNS if p in norm), None)
    alphabetic = sum(ch.isalpha() for ch in visible)
    printable = sum(ch.isprintable() and not ch.isspace() for ch in visible)
    alpha_ratio = alphabetic / printable if printable else 0.0
    passed = (
        2 <= len(words) <= 64
        and not repeated_triplet
        and unique_ratio >= 0.45
        and alpha_ratio >= 0.45
        and leak is None
    )
    return {
        "passed": passed,
        "word_count": len(words),
        "unique_word_ratio": unique_ratio,
        "alpha_ratio": alpha_ratio,
        "repeated_triplet": repeated_triplet,
        "leak_pattern": leak,
    }


def count_steps(text: str) -> int:
    # Prefer explicit numbered/bulleted lines; fall back to sentence count.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    explicit = sum(bool(re.match(r"^(?:\d+[.)]|[-*•])\s*", line)) for line in lines)
    if explicit:
        return explicit
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    return len(sentences)


def semantic_check(text: str, spec: dict) -> dict:
    norm = normalized(text)
    wc = word_count(text)
    reasons = []
    max_words = int(spec.get("max_words", 64))
    if wc > max_words:
        reasons.append(f"too_long:{wc}>{max_words}")
    for term in spec.get("all_terms", []):
        if normalized(term) not in norm:
            reasons.append(f"missing:{term}")
    for alternatives in spec.get("any_terms", []):
        if not any(normalized(term) in norm for term in alternatives):
            reasons.append("missing_any:" + "/".join(alternatives))
    if "label" in spec:
        label = normalized(spec["label"])
        if label not in norm:
            reasons.append(f"missing_label:{label}")
        for bad in spec.get("forbidden_labels", []):
            if normalized(bad) in norm:
                reasons.append(f"wrong_label_present:{bad}")
    if "min_steps" in spec:
        steps = count_steps(text)
        if steps < int(spec["min_steps"]):
            reasons.append(f"too_few_steps:{steps}")
    return {"passed": not reasons, "reasons": reasons, "word_count": wc}


def summary_markdown(report):
    lines = [
        "# Ember v0.0.54 strict INT4 direct-response quality",
        "",
        "Frozen v0.0.53 INT4 base. Tool/control markers masked. First turn only, stop at EOT.",
        "Deterministic semantic checks; nonempty text alone is not a pass.",
        "",
        "| Family | Semantic pass | Generic quality | Both |",
        "| --- | ---: | ---: | ---: |",
    ]
    for family, row in report["by_family"].items():
        lines.append(f"| {family} | {row['semantic']}/{row['total']} | {row['quality']}/{row['total']} | {row['both']}/{row['total']} |")
    lines += [
        "",
        f"Router direct decisions: {report['router_direct']}/{report['total']}",
        f"Semantic passes: {report['semantic_pass']}/{report['total']}",
        f"Generic-quality passes: {report['quality_pass']}/{report['total']}",
        f"Combined strict passes: **{report['combined_pass']}/{report['total']}**",
        f"Promotion-quality gate (>=21/24 combined and 24/24 routing): **{report['strict_pass']}**",
        f"Interpretation: {report['interpretation']}",
        "",
        "No Ember/router checkpoint or production pointer was changed.",
    ]
    return "\n".join(lines) + "\n"


def main():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN required")
    if len(CASES) != 24 or len({c["id"] for c in CASES}) != 24:
        raise RuntimeError("direct quality set must contain 24 unique cases")
    torch.set_num_threads(2)
    torch.manual_seed(20260910)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="ember-direct-quality-") as td:
        work = Path(td)
        archive = ev.download_verified(ev.PACKAGE_URL, work / "ember.zip", ev.PACKAGE_SHA256)
        with zipfile.ZipFile(archive) as package:
            package.extractall(work / "src")
        import sys
        sys.path.insert(0, str(work / "src" / "ember"))

        api = HfApi(token=token)
        owner = api.whoami()["name"]
        repo = f"{owner}/{held.MODEL_NAME}"
        model, tokenizer = held.load_int4(repo, work / "int4", token)
        model.eval()

        train_cases = cal.training_cases()
        train_rows = multi.extract(model, tokenizer, train_cases)
        heads = multi.prepare_heads(train_rows)
        quality_rows = multi.extract(model, tokenizer, CASES)
        routes, margins = multi.hierarchical_predict(heads, quality_rows)
        specials = control.marker_ids(tokenizer) if hasattr(control, "marker_ids") else None
        if specials is None:
            contract = ev.special_token_contract(tokenizer)
            specials = {m: int(contract["signatures"][m][0]) for m in ev.SPECIAL_TOKENS}

        rows = []
        by_family = {}
        router_direct = semantic_pass = quality_pass = combined_pass = 0
        for c, route, margin_value in zip(CASES, routes, margins):
            routed_direct = route == "direct"
            router_direct += int(routed_direct)
            generated = control.generate_direct(model, tokenizer, c["prompt"], specials)
            text = generated["text"]
            semantic = semantic_check(text, c["check"])
            quality = generic_quality(text)
            combined = routed_direct and generated["passed"] and semantic["passed"] and quality["passed"]
            semantic_pass += int(semantic["passed"])
            quality_pass += int(quality["passed"])
            combined_pass += int(combined)
            bucket = by_family.setdefault(c["family"], {"semantic": 0, "quality": 0, "both": 0, "total": 0})
            bucket["total"] += 1
            bucket["semantic"] += int(semantic["passed"])
            bucket["quality"] += int(quality["passed"])
            bucket["both"] += int(combined)
            row = {
                "id": c["id"], "family": c["family"], "user": c["user"],
                "router": route, "router_margin": float(margin_value), "router_direct": routed_direct,
                "text": text, "generation": generated, "semantic": semantic, "quality": quality,
                "combined_pass": combined,
            }
            rows.append(row)
            print(json.dumps({
                "event": "direct_quality_case", "id": c["id"], "family": c["family"],
                "router": route, "router_margin": float(margin_value), "text": text[:220],
                "semantic": semantic, "quality": quality, "combined_pass": combined,
            }, ensure_ascii=False), flush=True)

        strict = router_direct == 24 and combined_pass >= 21
        if strict:
            interpretation = (
                "INT4 direct responses cleared the strict first-turn quality gate. The next step would be one adversarial mixed direct/tool confirmation before packaging the v0.0.54 execution architecture."
            )
        else:
            failed = [r["id"] for r in rows if not r["combined_pass"]]
            interpretation = (
                "The tool execution architecture is strong, but Ember's direct-response language quality remains below promotion level. "
                f"Failed direct cases include {', '.join(failed[:12])}. Keep the router/executor frozen and train/evaluate the direct language path separately rather than modifying tool routing."
            )

        report = {
            "schema_version": 1,
            "diagnostic": "ember-v054-int4-direct-response-quality-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_repo": repo,
            "int4_checkpoint_sha256": held.INT4_SHA256,
            "router": {"binary_representation": heads["binary"]["representation"], "family_representation": heads["family"]["representation"]},
            "total": 24,
            "router_direct": router_direct,
            "semantic_pass": semantic_pass,
            "quality_pass": quality_pass,
            "combined_pass": combined_pass,
            "by_family": by_family,
            "rows": rows,
            "strict_pass": strict,
            "ember_weights_changed": False,
            "production_changed": False,
            "interpretation": interpretation,
            "elapsed_seconds": time.monotonic() - started,
        }
        (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        (OUT / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
        print(json.dumps({
            "event": "direct_quality_complete",
            "router_direct": f"{router_direct}/24", "semantic": f"{semantic_pass}/24",
            "quality": f"{quality_pass}/24", "combined": f"{combined_pass}/24",
            "by_family": by_family, "strict_pass": strict,
            "interpretation": interpretation, "elapsed_seconds": report["elapsed_seconds"],
        }), flush=True)


if __name__ == "__main__":
    main()
