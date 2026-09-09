#!/usr/bin/env python3
"""Tokenizer-free, model-free audit of an Ember copy curriculum.

The v0.0.16 -> v0.0.25 sequence repeatedly modified the curriculum and the loss
without ever checking the one property those changes depend on: that the
training curriculum can actually *emit* the character-level structure the
held-out diagnostic asks the model to reproduce.

This audit answers that question with no GPU, no checkpoint, no tokenizer and no
network. It compares every held-out diagnostic value against the structural
shapes the curriculum generates, and reports where the first structurally
unsupported character sits.

Shape alphabet: digit -> ``9``, lowercase -> ``a``, uppercase -> ``A``; every
other character is kept literally. ``/tmp/ember/Q7M4/result.json`` therefore has
the shape ``/aaa/aaaaa/A9A9/aaaaaa.aaaa``.

Usage::

    python jobs/ember_curriculum_audit_v026.py \
        --data jobs/ember_sft_data_v015.py \
        --diagnostics jobs/ember_hf_sft_v015.py \
        --train 3600 --validation 450 \
        --out reports/ember-curriculum-audit.json

Exit status is non-zero when ``--assert-parity`` is given and any diagnostic
value has a shape the curriculum never produces.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shape(value: str) -> str:
    return "".join(
        "9" if ch.isdigit() else "a" if ch.islower() else "A" if ch.isupper() else ch
        for ch in value
    )


def compact(signature: str) -> str:
    runs: list[list] = []
    for ch in signature:
        if runs and runs[-1][0] == ch:
            runs[-1][1] += 1
        else:
            runs.append([ch, 1])
    return "".join(c if n == 1 else f"{c}{{{n}}}" for c, n in runs)


def field_signature(value: str) -> str:
    """Template-level signature: separators kept, alphanumeric fields reduced.

    The strict character shape is too strict for fields whose contents are drawn
    at random from a mixed alphanumeric alphabet: ``A9A9`` and ``9A{2}A`` are the
    same template, just different draws. Each maximal alphanumeric run is
    therefore reduced to its length plus the *set* of character classes it
    contains, while every separator is kept literally -- so
    ``openai/gpt-6-astra`` becomes ``[6a]/[3a]-[1d]-[5a]`` and never collides
    with ``openai/ember-xn4k-42b`` (``[6a]/[5a]-[4ad]-[3ad]``).

    This is the signature ``--assert-parity`` gates on: it asks whether the
    curriculum can emit the diagnostic's *template*, not whether it happened to
    draw the same character classes in the same order.
    """
    parts: list[str] = []
    i = 0
    while i < len(value):
        if value[i].isalnum():
            j = i
            classes = set()
            while j < len(value) and value[j].isalnum():
                classes.add("d" if value[j].isdigit() else "a" if value[j].islower() else "A")
                j += 1
            parts.append(f"[{j - i}{''.join(sorted(classes))}]")
            i = j
        else:
            j = i
            while j < len(value) and not value[j].isalnum():
                j += 1
            parts.append(value[i:j])
            i = j
    return "".join(parts)


def longest_shared_prefix(signature: str, pool: list[str]) -> int:
    best = 0
    for candidate in pool:
        n = 0
        limit = min(len(signature), len(candidate))
        while n < limit and signature[n] == candidate[n]:
            n += 1
        if n > best:
            best = n
            if best == len(signature):
                break
    return best


def audit_case(kind: str, value: str, shapes_by_kind: dict, templates_by_kind: dict,
               first_chars: dict) -> dict:
    signature = shape(value)
    same_kind = shapes_by_kind.get(kind, Counter())
    every_kind: Counter = Counter()
    for counter in shapes_by_kind.values():
        every_kind.update(counter)
    prefix = longest_shared_prefix(signature, list(same_kind))
    shape_covered = prefix == len(signature)

    template = field_signature(value)
    kind_templates = templates_by_kind.get(kind, Counter())
    template_rows = int(kind_templates.get(template, 0))
    template_prefix = longest_shared_prefix(template, list(kind_templates))
    return {
        "kind": kind,
        "value": value,
        "template": template,
        "template_rows_same_kind": template_rows,
        "template_supported": template_rows > 0,
        "template_prefix_supported_chars": template_prefix,
        "template_length": len(template),
        "shape": compact(signature),
        "shape_rows_same_kind": int(same_kind.get(signature, 0)),
        "shape_rows_any_kind": int(every_kind.get(signature, 0)),
        "shape_prefix_supported_chars": prefix,
        "value_length": len(value),
        "shape_fully_supported": shape_covered,
        "first_unsupported_char_index": None if shape_covered else prefix,
        "first_unsupported_char": None if shape_covered else value[prefix],
        "context_at_divergence": None if shape_covered else value[max(0, prefix - 8):prefix + 8],
        "rows_with_same_leading_char": int(first_chars.get(kind, Counter()).get(value[0], 0)),
    }


def audit(data, diagnostics: list[tuple[str, str]], train_n: int, validation_n: int) -> dict:
    train = data.build_examples("train", train_n)
    validation = data.build_examples("validation", validation_n)

    shapes_by_kind: dict[str, Counter] = defaultdict(Counter)
    templates_by_kind: dict[str, Counter] = defaultdict(Counter)
    first_chars: dict[str, Counter] = defaultdict(Counter)
    for row in train:
        shapes_by_kind[row["kind"]][shape(row["value"])] += 1
        templates_by_kind[row["kind"]][field_signature(row["value"])] += 1
        first_chars[row["kind"]][row["value"][0]] += 1

    cases = [
        audit_case(kind, value, shapes_by_kind, templates_by_kind, first_chars)
        for kind, value in diagnostics
    ]
    uncovered = [c for c in cases if not c["template_supported"]]

    train_values = {row["value"] for row in train}
    val_values = {row["value"] for row in validation}
    leaked = sorted({v for _, v in diagnostics} & (train_values | val_values))

    return {
        "curriculum": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "kinds": sorted({row["kind"] for row in train}),
            "distinct_train_shapes": {k: len(v) for k, v in sorted(shapes_by_kind.items())},
            "distinct_train_templates": {k: len(v) for k, v in sorted(templates_by_kind.items())},
            "train_validation_value_overlap": sorted(train_values & val_values)[:10],
        },
        "diagnostics": {
            "cases": len(cases),
            "template_supported": len(cases) - len(uncovered),
            "template_unsupported": len(uncovered),
            "shape_fully_supported": sum(1 for c in cases if c["shape_fully_supported"]),
            "unsupported_kinds": sorted({c["kind"] for c in uncovered}),
            "held_out_leakage": leaked,
        },
        "cases": cases,
        "verdict": "FORMAT_PARITY" if not uncovered and not leaked else "FORMAT_GAP",
    }


def render(report: dict) -> str:
    lines = []
    c = report["curriculum"]
    d = report["diagnostics"]
    lines.append(
        f"curriculum: {c['train_rows']} train / {c['validation_rows']} validation rows, "
        f"{len(c['kinds'])} kinds"
    )
    lines.append(
        f"diagnostics: {d['cases']} cases, {d['template_supported']} with a supported template, "
        f"{d['template_unsupported']} unsupported"
    )
    lines.append("")
    shown = [r for r in report["cases"] if not r["template_supported"]] or report["cases"]
    width = max(len(row["template"]) for row in shown)
    lines.append(
        f"{'kind':12s} {'template':10s} {'field signature':{width}s} {'rows':>6s} {'chars':>7s} "
        f"first structurally unsupported character"
    )
    for row in shown:
        supported = "yes" if row["template_supported"] else "NO"
        span = f"{row['shape_prefix_supported_chars']}/{row['value_length']}"
        if row["shape_fully_supported"]:
            note = "-"
        else:
            note = (
                f"#{row['first_unsupported_char_index']} "
                f"{row['first_unsupported_char']!r} in ...{row['context_at_divergence']}..."
            )
        lines.append(
            f"{row['kind']:12s} {supported:10s} {row['template']:{width}s} "
            f"{row['template_rows_same_kind']:6d} {span:>7s} {note}"
        )
    if len(shown) != len(report["cases"]):
        lines.append(f"({len(report['cases']) - len(shown)} supported cases not listed)")
    lines.append("")
    lines.append(f"verdict: {report['verdict']}")
    return "\n".join(lines)


def literal_attribute(path: Path, attr: str, *, allow_import: bool = True):
    """Read a module-level literal without importing the module.

    The trainers import ``huggingface_hub`` and ``torch`` at module scope, so the
    audit reads their diagnostic batteries statically instead. That keeps this
    tool runnable on a bare Python 3.11 with no third-party packages installed.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) and node.value else []
        )
        for target in targets:
            if isinstance(target, ast.Name) and target.id == attr:
                try:
                    return ast.literal_eval(node.value)
                except ValueError:
                    break  # computed at import time; fall through to importing
    if not allow_import:
        raise RuntimeError(f"{path} does not define a literal {attr}")
    module = load_module(path, f"ember_audit_{attr.lower()}")
    if not hasattr(module, attr):
        raise RuntimeError(f"{path} does not define {attr}")
    return getattr(module, attr)


def resolve_diagnostics(args) -> list[tuple[str, str]]:
    """Read a held-out battery, tolerating both layouts used in this repo.

    v0.0.15's trainer stores ``(value, corrupted_value)`` and carries no kind
    labels, so kinds are inferred. v0.0.26's curriculum module stores
    ``(kind, value, corrupted_value)`` and exports ``KINDS``; the presence of
    that vocabulary is what distinguishes the two.
    """
    path = Path(args.diagnostics)
    cases = literal_attribute(path, args.diagnostics_attr)
    try:
        vocabulary = set(literal_attribute(path, "KINDS", allow_import=False))
    except RuntimeError:
        vocabulary = set()

    resolved = []
    for entry in cases:
        if isinstance(entry, dict):
            resolved.append((str(entry["kind"]), str(entry["value"])))
            continue
        if len(entry) >= 3 and str(entry[0]) in vocabulary:
            resolved.append((str(entry[0]), str(entry[1])))
            continue
        resolved.append((infer_kind(str(entry[0])), str(entry[0])))
    return resolved


def infer_kind(value: str) -> str:
    """Classify a legacy diagnostic tuple that carries no explicit kind."""
    if value.startswith("http"):
        return "url"
    if value.startswith("/"):
        return "path"
    if "/" in value:
        return "model_id"
    if value.startswith("acct_"):
        return "mixed"
    if " " in value:
        return "entity"
    if any(op in value for op in "*+-") and value.replace("*", "").replace("+", "").replace("-", "").isdigit():
        return "expression" if not value.isdigit() else "digits"
    if value.isdigit():
        return "digits"
    if "-" in value:
        return "long_code"
    return "short_code"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="curriculum module exposing build_examples()")
    parser.add_argument("--diagnostics", required=True, help="module exposing the held-out battery")
    parser.add_argument("--diagnostics-attr", default="DIAGNOSTICS")
    parser.add_argument("--train", type=int, default=3600)
    parser.add_argument("--validation", type=int, default=450)
    parser.add_argument("--out", default="")
    parser.add_argument("--assert-parity", action="store_true")
    args = parser.parse_args()

    data = load_module(Path(args.data), "ember_audit_curriculum")
    diagnostics = resolve_diagnostics(args)
    report = audit(data, diagnostics, args.train, args.validation)
    print(render(report), flush=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {out}", flush=True)
    if args.assert_parity and report["verdict"] != "FORMAT_PARITY":
        print(
            "\nFORMAT GAP: the curriculum cannot emit every held-out diagnostic shape; "
            "training on it cannot close those cases.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
