"""Opt-in definition routing over the unchanged v7 runtime and parser."""
from .routing_v8 import REVISION, definition_question
from .runtime_v7 import RoutingV7Runtime, TextHelperV7Runtime


class DefinitionRoutingMixin:
    routing_revision = REVISION

    def route(self, user):
        # Preserve validation, context limits, and every existing fallback rule.
        original = super().route(user)
        return ("direct", 0.0) if definition_question(user) else original

    def route_source(self, user):
        return "definition_question" if definition_question(user) else super().route_source(user)


class TextHelperV8Runtime(DefinitionRoutingMixin, TextHelperV7Runtime):
    """Development helper; not checkpoint integration evidence."""


class RoutingV8Runtime(DefinitionRoutingMixin, RoutingV7Runtime):
    """Archived candidate plus the separately frozen definition source overlay."""
