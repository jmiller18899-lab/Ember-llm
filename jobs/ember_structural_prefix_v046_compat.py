"""Tokenizer-stable entry point for the v0.0.46 structural-prefix diagnostic.

SentencePiece can retokenize adjacent JSON fragments when each semantic segment
is encoded separately. This compatibility entry point tokenizes the complete
canonical non-value prefix once, then maps its fixed tokens back to semantic
stages by decoded character spans. It changes no prompt, model, cohort, prefix,
value boundary, metric, or authorization rule in the v0.0.46 experiment.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import ember_structural_prefix_v046 as v046


def _normalized(decoded: str) -> str:
    # The compact historical prefix contains no whitespace. SentencePiece may
    # surface a single synthetic leading space when a standalone string is
    # decoded, so strip only leading whitespace for alignment.
    return decoded.lstrip()


def token_plan(tokenizer, prompt: str, segments: list[dict]) -> tuple[list[int], list[dict]]:
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("empty prompt")
    prefix = "".join(segment["text"] for segment in segments)

    combined = tokenizer.encode(prompt + prefix)
    if combined[: len(prompt_ids)] == prompt_ids:
        structural_ids = list(combined[len(prompt_ids) :])
    else:
        # Fallback to a standalone prefix anchored at the atomic tool marker.
        contract = v046.base.semantic_gate.token_contract(tokenizer)
        tool_id = int(contract["signatures"][v046.prior.TOOL][0])
        encoded_prefix = tokenizer.encode(prefix)
        try:
            start = encoded_prefix.index(tool_id)
        except ValueError as exc:
            raise ValueError("canonical prefix lost the atomic tool marker") from exc
        structural_ids = list(encoded_prefix[start:])

    if not structural_ids:
        raise ValueError("canonical prefix produced no structural tokens")
    final_decoded = _normalized(tokenizer.decode(structural_ids))
    if final_decoded != prefix:
        raise ValueError(
            "canonical prefix does not round-trip after full-prefix tokenization: "
            + repr(final_decoded)
        )

    spans = []
    cursor = 0
    for segment in segments:
        start = cursor
        cursor += len(segment["text"])
        spans.append((start, cursor, segment["name"]))

    plan = []
    previous_length = 0
    for token_index, token_id in enumerate(structural_ids):
        decoded = _normalized(tokenizer.decode(structural_ids[: token_index + 1]))
        if not prefix.startswith(decoded):
            raise ValueError(
                f"decoded structural token prefix diverged at token {token_index}: {decoded!r}"
            )
        current_length = len(decoded)
        if current_length <= previous_length:
            raise ValueError("structural token made no forward decoded progress")
        touched = [
            name for start, end, name in spans
            if previous_length < end and current_length > start
        ]
        if not touched:
            raise ValueError("structural token could not be mapped to a semantic stage")
        plan.append({
            "stage": touched[-1],
            "stages_touched": touched,
            "stage_token_index": sum(1 for item in plan if item["stage"] == touched[-1]),
            "expected_token_id": int(token_id),
            "expected_token_text": v046.prior._decode_token(tokenizer, int(token_id)),
            "decoded_char_start": previous_length,
            "decoded_char_end": current_length,
        })
        previous_length = current_length

    if previous_length != len(prefix):
        raise ValueError("structural token plan did not reach the full non-value prefix boundary")
    return list(prompt_ids), plan


v046.token_plan = token_plan


if __name__ == "__main__":
    raise SystemExit(v046.main())
