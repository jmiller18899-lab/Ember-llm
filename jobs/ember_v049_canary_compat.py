"""Launch v0.0.49 after applying family-specific replay sizing."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import ember_v049_replay_compat  # patches replay pool sizing
from jobs import ember_v049_canary as canary


if __name__ == "__main__":
    raise SystemExit(canary.main())
