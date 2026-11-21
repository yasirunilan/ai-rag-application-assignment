# Story E0.1 — One-command free setup for Windows (PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "==> [E0.1] Creating lightweight virtual environment (.venv) with zero PyTorch/CUDA bloat..." -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

Write-Host "==> [E0.1] Installing mmrag light core dependencies..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[faiss]"

Write-Host "==> [E0.1] Running zero-network end-to-end smoke test (ingest -> answer -> evaluate)..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m mmrag.cli smoke

Write-Host "==> [E0.1] Setup & smoke test complete!" -ForegroundColor Green
