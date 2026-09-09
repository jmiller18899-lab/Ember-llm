from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import ember_v048_value_compat  # noqa: F401
from jobs import ember_v048_canary as canary

if __name__ == "__main__":
    raise SystemExit(canary.main())
