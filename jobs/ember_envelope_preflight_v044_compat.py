"""Compatibility entry point for Ember v0.0.44 CPU preflight.

v0.0.43 delegates metric helpers to its v0.0.42 parent. This shim exposes
those helpers on the v0.0.43 module before invoking the unchanged v0.0.44
runner, avoiding any change to prompts, selection, gates, or model behavior.
"""
from __future__ import annotations

from jobs import ember_envelope_preflight_v044 as v044

v044.prior.per_kind = v044.prior.prior.per_kind
v044.prior.per_subtype = v044.prior.prior.per_subtype
v044.prior.regression_gate = v044.prior.prior.regression_gate


if __name__ == "__main__":
    raise SystemExit(v044.main())
