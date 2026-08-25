#!/usr/bin/env python3
"""Run an Ember script and exit as soon as it returns.

The corpus scripts stream from Hugging Face datasets, which leaves native
worker threads (``datasets`` / ``hf_xet``) alive after the work is done. On
CPython those threads can abort the process during interpreter finalization::

    Fatal Python error: PyGILState_Release: thread state ... must be current
    Python runtime state: finalizing
    Aborted (core dumped)   # exit code 134

That happens *after* the script has completed and printed its result, so a
successful run is reported as a failure. Running the script through ``runpy``
and then calling ``os._exit`` skips interpreter finalization entirely.

Real failures are unaffected: an exception propagates out of ``run_path`` and
the process exits non-zero before ``os._exit`` is reached, and an explicit
``SystemExit`` keeps its own status code.
"""
from __future__ import annotations

import os
import runpy
import sys


def _exit(code: int) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: run_ember_script.py <script.py> [args...]", file=sys.stderr)
        raise SystemExit(2)

    script = os.path.abspath(sys.argv[1])
    # Emulate `python <script> [args...]`: the script sees itself as argv[0]
    # and its own directory on sys.path.
    sys.argv = [script, *sys.argv[2:]]
    sys.path.insert(0, os.path.dirname(script))

    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        if code is None:
            code = 0
        elif not isinstance(code, int):
            print(code, file=sys.stderr)
            code = 1
        _exit(code)

    _exit(0)


if __name__ == "__main__":
    main()
