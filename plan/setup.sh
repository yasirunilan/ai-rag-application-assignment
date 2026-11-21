#!/usr/bin/env bash
# Story E0.1 — One-command free setup for Linux / macOS / Git Bash
set -euo pipefail

echo "==> [E0.1] Creating lightweight virtual environment (.venv) with zero PyTorch/CUDA bloat..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

echo "==> [E0.1] Installing mmrag light core dependencies..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[faiss]"

echo "==> [E0.1] Running zero-network end-to-end smoke test (ingest -> answer -> evaluate)..."
.venv/bin/python -m mmrag.cli smoke

echo "==> [E0.1] Setup & smoke test complete!"
