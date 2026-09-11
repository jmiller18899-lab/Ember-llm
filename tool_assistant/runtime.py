"""Offline inference from a verified, frozen Ember candidate bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

from . import binary, family
from .resolver import resolve_v2

SYSTEM = (
    "You are Ember. You may answer normally or use one of these tools: weather, "
    "calculator, web_search, get_time. Use a tool only when live weather, arithmetic, "
    "current web information, or current local time is actually needed. Otherwise "
    "answer directly. Tool calls use JSON arguments."
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt(user):
    return f"<|system|>\n{SYSTEM}\n<|user|>\n{user}\n<|assistant|>\n"


def hidden(model, tokenizer, rendered):
    ids = tokenizer.encode(rendered)
    if not ids or len(ids) > model.cfg.block_size:
        raise ValueError("Request exceeds Ember's context window")
    captured = []

    def hook(_module, _inputs, output):
        captured.append(output[:, -1, :].detach().float().cpu()[0])

    handle = model.blocks[4].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(torch.tensor([ids], dtype=torch.long), None)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("Expected exactly one block_04 representation")
    return captured[0].double()


def load_model(bundle, precision):
    bundle = Path(bundle)
    if precision not in ("full", "int4"):
        raise ValueError("precision must be full or int4")
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle.resolve()) or sha256(path) != expected:
            raise RuntimeError(f"Bundle checksum mismatch: {name}")
    sys.path.insert(0, str(bundle / "ember"))
    from src.model import EmberGPT, ModelConfig
    from src.tokenizer import tokenizer_from_state_dict
    from src.quantize_int4 import dequantize_tensor

    payload = torch.load(bundle / f"model-{precision}.pt", map_location="cpu", weights_only=True)
    if precision == "int4":
        state = {k: dequantize_tensor(v) for k, v in payload["quantized_state"].items()}
        state.update(payload.get("passthrough_state", {}))
    else:
        state = payload["model_state"]
    model = EmberGPT(ModelConfig(**payload["model_config"]))
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    return model, tokenizer_from_state_dict(payload["tokenizer"]), manifest


class Runtime:
    def __init__(self, bundle, precision="int4"):
        self.model, self.tokenizer, self.manifest = load_model(bundle, precision)
        self.precision = precision
        self.binary = torch.load(Path(bundle) / f"router-{precision}.pt", map_location="cpu", weights_only=True)
        self.family = torch.load(Path(bundle) / "family.pt", map_location="cpu", weights_only=True)

    def route(self, user):
        if not isinstance(user, str) or not user.strip():
            raise ValueError("Request must be a nonempty string")
        if "<|" in user:
            raise ValueError("Requests cannot contain Ember conversation markers")
        x = hidden(self.model, self.tokenizer, prompt(user)).unsqueeze(0)
        bp, _, bm = binary.predict(self.binary, x)
        features = family.transform([user], self.family["vectorizer"])
        fp, _, fm = family.predict_ridge(self.family["model"], features)
        label = "direct" if int(bp[0]) == 0 else family.FAMILIES[int(fp[0])]
        margin = float(bm[0] if label == "direct" else min(bm[0], fm[0]))
        return label, margin

    def plan(self, user):
        label, margin = self.route(user)
        if label == "direct":
            return {"route": label, "status": "direct_answer_unavailable", "call": None, "margin": margin}
        result = resolve_v2({"user": user}, label)
        return {
            "route": label,
            "status": "tool_call" if result["payload"] else "needs_clarification",
            "call": result["payload"],
            "margin": margin,
        }

    def run(self, user, handlers):
        """Dispatch only through caller-supplied tool implementations."""
        result = self.plan(user)
        if result["status"] != "tool_call":
            return result
        call = result["call"]
        handler = handlers.get(call["name"])
        if handler is None:
            return {**result, "status": "tool_unavailable"}
        try:
            value = handler(**call["arguments"])
        except Exception as exc:
            return {**result, "status": "tool_error", "error_type": type(exc).__name__}
        return {**result, "status": "tool_result", "result": value}
