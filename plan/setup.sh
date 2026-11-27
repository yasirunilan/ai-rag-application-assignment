#!/usr/bin/env bash
# Story E0.1 — One-command free setup for Linux / macOS / Git Bash
set -euo pipefail
cd "$(dirname "$0")"

# pyproject requires Python >= 3.10; macOS's /usr/bin/python3 is often 3.9, so pick the newest available.
PY=""
for cand in ${PYTHON:-} python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        PY="$cand"
        break
    fi
done
if [ -z "$PY" ]; then
    echo "ERROR: Python >= 3.10 not found. Install it (e.g. 'brew install python@3.12') or set PYTHON=/path/to/python." >&2
    exit 1
fi

echo "==> [E0.1] Creating lightweight virtual environment (.venv) with $("$PY" --version) and zero PyTorch/CUDA bloat..."
if [ ! -d ".venv" ]; then
    "$PY" -m venv .venv
fi

echo "==> [E0.1] Installing mmrag light core dependencies..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[faiss]"

if ! command -v tesseract >/dev/null 2>&1; then
    echo "WARN [E1.2] Tesseract not found: scanned pages will be queued for review, not indexed."
    echo "     Install it with 'brew install tesseract' (macOS) or 'sudo apt install tesseract-ocr' (Debian/Ubuntu)."
fi

echo "==> [E0.1] Environment check (offline; run 'python -m mmrag.cli check' once OPENROUTER_API_KEY is set)..."
.venv/bin/python -m mmrag.cli check --offline

echo "==> [E0.1] Running zero-network end-to-end smoke test (ingest -> answer -> evaluate)..."
.venv/bin/python -m mmrag.cli smoke

echo "==> [E0.1] Setup & smoke test complete!"
