"""Frozen-teacher distribution-preservation helpers for Ember v0.0.50."""
from __future__ import annotations

from statistics import mean

from jobs import ember_v049_replay as v049


def sequence_example(tokenizer, row: dict) -> tuple[list[int], list[int]]:
    return v049.sequence_example(tokenizer, row)


def _batch_tensors(tokenizer, torch, rows: list[dict], indices: list[int]):
    examples = [sequence_example(tokenizer, rows[i]) for i in indices]
    max_len = max(len(x) for x, _y in examples)
    xs, ys = [], []
    for x, y in examples:
        xs.append(x + [0] * (max_len - len(x)))
        ys.append(y + [-100] * (max_len - len(y)))
    return (
        torch.tensor(xs, dtype=torch.long, device="cpu"),
        torch.tensor(ys, dtype=torch.long, device="cpu"),
    )


def batch_teacher_kl(student, teacher, tokenizer, torch, rows: list[dict], indices: list[int], temperature: float):
    """KL(teacher || student) only at teacher-generated response-token positions."""
    import torch.nn.functional as F
    x, y = _batch_tensors(tokenizer, torch, rows, indices)
    mask = y != -100
    student_logits, _ = student(x)
    with torch.inference_mode():
        teacher_logits, _ = teacher(x)
    s = student_logits[mask].float() / temperature
    t = teacher_logits[mask].float() / temperature
    teacher_probs = torch.softmax(t, dim=-1)
    student_log_probs = torch.log_softmax(s, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def distill_probe(student, teacher, tokenizer, torch, rows: list[dict], batch_size: int = 8, temperature: float = 1.0) -> dict:
    import torch.nn.functional as F
    total = correct = 0
    kls = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            indices = list(range(start, min(start + batch_size, len(rows))))
            x, y = _batch_tensors(tokenizer, torch, rows, indices)
            mask = y != -100
            student_logits, _ = student(x)
            teacher_logits, _ = teacher(x)
            pred = torch.argmax(student_logits, dim=-1)
            total += int(mask.sum().item())
            correct += int(((pred == y) & mask).sum().item())
            s = student_logits[mask].float() / temperature
            t = teacher_logits[mask].float() / temperature
            kl = F.kl_div(
                torch.log_softmax(s, dim=-1),
                torch.softmax(t, dim=-1),
                reduction="batchmean",
            ) * (temperature ** 2)
            kls.append(float(kl.item()))
    return {
        "examples": len(rows),
        "tokens": total,
        "token_top1": correct,
        "token_top1_rate": correct / total,
        "teacher_kl": mean(kls),
    }


def selection_checks(before_entry: dict, after_entry: dict, before_place: dict, after_place: dict, tool_probe: dict, copy_probe: dict, cfg: dict) -> dict:
    gate = cfg["selection_gate"]
    metrics = {
        "entry_top1_gain": after_entry["top1"] - before_entry["top1"],
        "placement_exact_gain": after_place["exact_top1"] - before_place["exact_top1"],
        "placement_token_top1_gain": after_place["token_top1_rate"] - before_place["token_top1_rate"],
        "tool_distill_token_top1": tool_probe["token_top1_rate"],
        "copy_distill_token_top1": copy_probe["token_top1_rate"],
        "tool_teacher_kl": tool_probe["teacher_kl"],
        "copy_teacher_kl": copy_probe["teacher_kl"],
    }
    checks = {
        "entry_top1": metrics["entry_top1_gain"] >= int(gate["minimum_entry_top1_gain"]),
        "placement_exact": metrics["placement_exact_gain"] >= int(gate["minimum_placement_exact_gain"]),
        "placement_tokens": metrics["placement_token_top1_gain"] >= float(gate["minimum_placement_token_top1_gain"]),
        "tool_top1": metrics["tool_distill_token_top1"] >= float(gate["minimum_tool_distill_token_top1"]),
        "copy_top1": metrics["copy_distill_token_top1"] >= float(gate["minimum_copy_distill_token_top1"]),
        "tool_kl": metrics["tool_teacher_kl"] <= float(gate["maximum_tool_teacher_kl"]),
        "copy_kl": metrics["copy_teacher_kl"] <= float(gate["maximum_copy_teacher_kl"]),
    }
    return {"passed": all(checks.values()), "checks": checks, "metrics": metrics}
