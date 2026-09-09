"""Lineage-compatible entry point for the v0.0.50 teacher-KL canary."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import ember_v049_replay_compat  # noqa: F401; patches widened v0.0.49 replay sizing
from jobs import ember_v050_canary as canary

if __name__ == "__main__":
    raise SystemExit(canary.main())
