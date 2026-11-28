# Story E0.1 — One-command free setup for Windows (PowerShell)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# pyproject requires Python >= 3.10: prefer the py launcher, fall back to python on PATH.
$py = $null
foreach ($cand in @(@("py", "-3"), @("python"))) {
    $exe = $cand[0]; $extra = @($cand | Select-Object -Skip 1)
    if (Get-Command $exe -ErrorAction SilentlyContinue) {
        & $exe @extra -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { $py = $cand; break }
    }
}
if (-not $py) {
    Write-Host "ERROR: Python >= 3.10 not found. Install it from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}
$pyExe = $py[0]; $pyArgs = @($py | Select-Object -Skip 1)

Write-Host "==> [E0.1] Creating lightweight virtual environment (.venv) with zero PyTorch/CUDA bloat..." -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    & $pyExe @pyArgs -m venv .venv
}

Write-Host "==> [E0.1] Installing mmrag light core dependencies..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[faiss]"

$tess = (Get-Command tesseract -ErrorAction SilentlyContinue) -or (Test-Path "C:\Program Files\Tesseract-OCR\tesseract.exe")
if (-not $tess) {
    Write-Host "WARN [E1.2] Tesseract not found: scanned pages will be queued for review, not indexed." -ForegroundColor Yellow
    Write-Host "     Install it with 'winget install UB-Mannheim.TesseractOCR'." -ForegroundColor Yellow
}

Write-Host "==> [E0.1] Environment check (offline; run 'python -m mmrag.cli check' once OPENROUTER_API_KEY is set)..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m mmrag.cli check --offline
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> [E0.1] Running zero-network end-to-end smoke test (ingest -> answer -> evaluate)..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m mmrag.cli smoke
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> [E0.1] Setup & smoke test complete!" -ForegroundColor Green
