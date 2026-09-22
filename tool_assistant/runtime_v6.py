"""Optional context rule with the original fitted head and parser retained."""
from .routing_v6 import REVISION, contextual_route
from .runtime_v5 import RoutingV5Runtime, TextHelperRuntime


class ContextRoutingMixin:
    def route(self, user):
        # Always retain v5 input validation and the integrated context limit.
        label, margin = super().route(user)
        contextual = contextual_route(user)
        # Zero is an API sentinel, not confidence from a fitted classifier.
        return (contextual, 0.0) if contextual is not None else (label, margin)

    def plan(self, user):
        result = super().plan(user)
        contextual = contextual_route(user)
        return {**result, "routing_revision": REVISION,
                "routing_source": "validated_context_question" if contextual else "frozen_v5_classifier",
                "margin_kind": "not_applicable" if contextual else "classifier_top_two_gap"}


class TextHelperV6Runtime(ContextRoutingMixin, TextHelperRuntime):
    """Model-free development check; not checkpoint integration evidence."""


class RoutingV6Runtime(ContextRoutingMixin, RoutingV5Runtime):
    """Load the original verified bundle, then apply the opt-in context rule."""
