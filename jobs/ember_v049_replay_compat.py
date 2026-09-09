"""Family-specific replay-pool sizing for the v0.0.49 protected canary."""
from __future__ import annotations

from jobs import ember_v049_replay as _base

for _name in dir(_base):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_base, _name)


def replay_value_rows(cfg: dict, family: str) -> list[dict]:
    if family not in {"tool", "copy"}:
        raise ValueError(family)
    used = _base._used_values()
    rows = []
    key = "tool_replay_values_per_variant" if family == "tool" else "copy_replay_values_per_variant"
    count = int(cfg[key])
    for kind in copy_data.KINDS:
        for variant in range(int(copy_data.VARIANTS[kind])):
            made = 0
            i = 0
            while made < count:
                seed = copy_data._digest("v049-replay", family, kind, variant, i)
                value = copy_data._render(kind, variant, seed)
                i += 1
                if value in used:
                    continue
                used.add(value)
                rows.append({"kind": kind, "variant": variant, "target": value})
                made += 1
    values = {row["target"] for row in rows}
    if values & set(copy_data.HELD_OUT_VALUES):
        raise ValueError("replay values overlap familiar held-out battery")
    return rows


_base.replay_value_rows = replay_value_rows
