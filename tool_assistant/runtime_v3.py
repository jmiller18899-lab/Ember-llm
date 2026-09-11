"""Opt-in argument-parser revision using the unchanged frozen router."""
from .runtime import Runtime
from .resolver_v3 import REVISION, resolve_v3


class ParserRuntime(Runtime):
    def plan(self, user):
        label, margin = self.route(user)
        result = {"route": label, "margin": margin, "parser_revision": REVISION}
        if label == "direct":
            return {**result, "status": "direct_answer_unavailable", "call": None}
        resolved = resolve_v3({"user": user}, label)
        if resolved["payload"] is None:
            return {**result, "status": "needs_clarification", "call": None,
                    "reason": resolved["reason"], "message": resolved["clarification"]}
        return {**result, "status": "tool_call", "call": resolved["payload"]}
