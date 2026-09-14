"""Optional contact repair over the unchanged v8 runtime."""
from .context_v9 import REVISION, contact_call
from .runtime_v8 import RoutingV8Runtime, TextHelperV8Runtime


class ContactMixin:
    def route(self, user):
        original = super().route(user)
        call = contact_call(user)
        return (call['name'], 0.0) if call else original

    def plan(self, user):
        # Always run original validation, including integrated context limits.
        original = super().plan(user)
        call = contact_call(user)
        if call is None:
            return original
        return {'route': call['name'], 'margin': 0.0,
                'routing_source': 'validated_contact_v9',
                'margin_kind': 'not_applicable', 'routing_revision': REVISION,
                'parser_revision': 'contact-parser-v9',
                'status': 'tool_call', 'call': call}


class TextHelperV9Runtime(ContactMixin, TextHelperV8Runtime):
    """Development helper, not checkpoint integration evidence."""


class RoutingV9Runtime(ContactMixin, RoutingV8Runtime):
    """Original archived candidate with a separately frozen contact overlay."""
