"""Explicit text routing helper with the opt-in v4 argument parser."""
from pathlib import Path
import torch

from .resolver_v3 import request_text
from .resolver_v4 import REVISION as PARSER_REVISION, resolve_v4
from .routing_data_v4 import LABELS
from .routing_v5 import REVISION, predict
from .runtime import Runtime, load_model, prompt
from .runtime_v4 import RoutingRuntime


class ParserV4Control(RoutingRuntime):
    """Paired control: original v4 routing, revised argument parsing only."""
    def plan(self, user):
        label, margin = self.route(user)
        result = {"route": label, "margin": margin, "parser_revision": PARSER_REVISION}
        if label == "direct":
            return {**result, "status": "direct_answer_unavailable", "call": None}
        resolved = resolve_v4({"user": user}, label)
        if resolved["payload"] is None:
            return {**result, "status": "needs_clarification", "call": None,
                    "reason": resolved["reason"], "message": resolved["clarification"]}
        return {**result, "status": "tool_call", "call": resolved["payload"]}


class TextHelperRuntime:
    """Model-free helper for explicitly labeled development checks."""
    plan = ParserV4Control.plan
    run = Runtime.run

    def __init__(self, state):
        if state.get("revision") != REVISION or state.get("labels") != list(LABELS) or state.get("uses_ember_features") is not False:
            raise ValueError("Wrong text helper revision or labels")
        self.routing = state

    def route(self, user):
        request_text(user)
        labels, margin = predict(self.routing, [user])
        return LABELS[int(labels[0])], float(margin[0])


class RoutingV5Runtime(TextHelperRuntime):
    """Verified Ember bundle integration; the route itself uses only text.

    Both checkpoints intentionally share one head. Loading a checkpoint checks
    the frozen bundle and retains its tokenizer/context boundary, not LLM routing.
    """
    def __init__(self, bundle, precision="int4"):
        self.model, self.tokenizer, self.manifest = load_model(bundle, precision)
        self.precision = precision
        super().__init__(torch.load(Path(bundle) / "routing-v5.pt", map_location="cpu", weights_only=True))

    def route(self, user):
        request_text(user)
        ids = self.tokenizer.encode(prompt(user))
        if not ids or len(ids) > self.model.cfg.block_size:
            raise ValueError("Request exceeds Ember's context window")
        return super().route(user)
