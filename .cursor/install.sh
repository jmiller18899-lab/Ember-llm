#!/usr/bin/env bash
# Idempotent Cloud Agent setup for the Ember LLM repository.
#
# Ember's authoritative source ships as a versioned `ember-v*-hf-ready.zip`
# package. Development, CI (`.github/workflows/ember-validate.yml`), and the
# Hugging Face Jobs launchers in `jobs/` all target Python 3.11 with a CPU-only
# PyTorch build. This script reproduces that CPU validation environment.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 1. Provide Python 3.11 via uv (matches the CI + HF Jobs pin). uv also runs the
#    PEP 723 `jobs/*.py` launchers as `uv run` in production.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11

# 2. Extract the newest authoritative Ember package to ./ember-src.
PKG="$(ls -1 ember-v*-hf-ready.zip | sort -V | tail -1)"
echo "Ember package: ${PKG} (version $(unzip -p "$PKG" ember/VERSION | tr -d '[:space:]'))"
rm -rf ember-src
mkdir -p ember-src
unzip -q "$PKG" -d ember-src

# 3. Create a Python 3.11 virtualenv and install CPU-only dependencies:
#    torch from the PyTorch CPU index, plus the corpus/test requirements.
uv venv --python 3.11 --clear .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu "torch>=2.4"
uv pip install --python .venv/bin/python -r ember-src/ember/requirements-corpus.txt

# 4. Sanity check: the Hugging Face Jobs launchers must byte-compile (matches CI).
.venv/bin/python -m py_compile jobs/*.py

echo "Ember CPU development environment is ready."
echo "  Interpreter: $(.venv/bin/python --version)"
echo "  Source:      ember-src/ember   (run pytest / scripts from here)"
