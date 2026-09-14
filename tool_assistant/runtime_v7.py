"""Opt-in request framing and arithmetic repair over the frozen v6 control."""
from .resolver_v7 import REVISION as PARSER_REVISION, resolve_v7
from .routing_v6 import REVISION as V6_REVISION, contextual_route
from .routing_v7 import REVISION, request_frame
from .runtime_v6 import RoutingV6Runtime, TextHelperV6Runtime


class ParserV7Mixin:
    routing_revision = V6_REVISION

    def route_source(self, user):
        return "validated_context_question" if contextual_route(user) else "frozen_v5_classifier"

    def plan(self, user):
        label, margin = self.route(user)
        source = self.route_source(user)
        result = {"route": label, "margin": margin, "routing_source": source,
                  "margin_kind": "classifier_top_two_gap" if source == "frozen_v5_classifier" else "not_applicable",
                  "routing_revision": self.routing_revision, "parser_revision": PARSER_REVISION}
        if label == "direct":
            return {**result, "status": "direct_answer_unavailable", "call": None}
        resolved = resolve_v7({"user": user}, label)
        if resolved["payload"] is None:
            return {**result, "status": "needs_clarification", "call": None,
                    "reason": resolved["reason"], "message": resolved["clarification"]}
        return {**result, "status": "tool_call", "call": resolved["payload"]}


class RequestFrameMixin(ParserV7Mixin):
    routing_revision = REVISION

    def route(self, user):
        # V6 always runs first so validation and checkpoint context bounds remain.
        original = super().route(user)
        framed = request_frame(user)
        return (framed[0], 0.0) if framed else original

    def route_source(self, user):
        framed = request_frame(user)
        return framed[1] if framed else super().route_source(user)


class ParserV7Control(ParserV7Mixin, RoutingV6Runtime):
    """Paired control: v6 routing with the arithmetic parser fix only."""


class TextParserV7Control(ParserV7Mixin, TextHelperV6Runtime):
    """Model-free parser control for explicitly labelled development checks."""


class TextHelperV7Runtime(RequestFrameMixin, TextHelperV6Runtime):
    """Model-free development runtime; not checkpoint integration evidence."""


class RoutingV7Runtime(RequestFrameMixin, RoutingV6Runtime):
    """Original archived bundle plus the separately hashed v7 source overlay."""
