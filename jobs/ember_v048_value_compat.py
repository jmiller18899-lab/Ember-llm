"""Context-aware value continuation compatibility for v0.0.48."""
from __future__ import annotations

from jobs import ember_v048_data as data


def value_continuation_ids(tokenizer, template: dict, target: str) -> list[int]:
    expected = template["prefix_text"] + target
    prefix_ids = list(template["prefix_ids"])
    contexts = [target]
    text = template["prefix_text"]
    for width in range(1, min(len(text), 24) + 1):
        contexts.append(text[-width:] + target)
    contexts.append(text + target)
    seen = set()
    for context in contexts:
        ids = tuple(int(x) for x in tokenizer.encode(context))
        if ids in seen:
            continue
        seen.add(ids)
        for start in range(len(ids)):
            suffix = list(ids[start:])
            if suffix and tokenizer.decode(prefix_ids + suffix) == expected:
                return suffix
    raise ValueError(f"cannot form exact context continuation tokens for {target!r}")


data.value_continuation_ids = value_continuation_ids
