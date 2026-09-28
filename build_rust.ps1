param([string]$PythonPath)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here

$candidates = @()
if ($PythonPath) { $candidates += $PythonPath }
$candidates += ".\.venv\Scripts\python.exe"
$command = Get-Command python -ErrorAction SilentlyContinue
if ($command) { $candidates += $command.Source }

$python = $null
foreach ($candidate in $candidates) {
    if (-not (Test-Path -LiteralPath $candidate)) { continue }
    $resolved = (Resolve-Path -LiteralPath $candidate).Path
    & $resolved -c "import sys; raise SystemExit(sys.version_info[:2] != (3, 12))" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $python = $resolved
        break
    }
}
if (-not $python) {
    throw "Python 3.12 was not found. Create .venv with 'py -3.12 -m venv .venv' or pass -PythonPath <path>."
}

$rustBin = Join-Path $env:USERPROFILE ".cargo\bin"
if (-not (Get-Command rustc -ErrorAction SilentlyContinue) -and (Test-Path $rustBin)) {
    $env:PATH = "$rustBin;$env:PATH"
}
if (-not (Get-Command rustc -ErrorAction SilentlyContinue)) {
    throw "Rust was not found. Install rustup from https://rustup.rs/ and reopen PowerShell."
}
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw "Cargo was not found. Reopen PowerShell after installing Rust with rustup."
}

Write-Host "Python: $python"
Write-Host "Rust:   $(& rustc --version)"
Write-Host "Installing pinned v2.0.0 build dependencies..."
& $python -m pip install -r .\requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$scriptsDir = & $python -c "import sysconfig; print(sysconfig.get_path('scripts'))"
$maturin = Join-Path $scriptsDir "maturin.exe"
if (-not (Test-Path $maturin)) {
    $fallback = Get-Command maturin -ErrorAction SilentlyContinue
    if ($fallback) {
        $maturin = $fallback.Source
    } else {
        throw "maturin was installed but maturin.exe could not be located."
    }
}

$dist = Join-Path $env:TEMP "qaoa_v2_0_0_dist"
if (-not $env:CARGO_TARGET_DIR) {
    $env:CARGO_TARGET_DIR = Join-Path $env:TEMP "qaoa_v2_0_0_cargo_target"
}
New-Item -ItemType Directory -Force -Path $dist | Out-Null
Get-ChildItem $dist -Filter "*.whl" -ErrorAction SilentlyContinue | Remove-Item -Force

Write-Host "Building qaoa_v2_rust in release mode..."
Push-Location .\rust_core
try {
    & $maturin build --release --interpreter $python --out $dist
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}

$wheel = Get-ChildItem $dist -Filter "qaoa_v2_rust-*.whl" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $wheel) {
    throw "maturin finished but no qaoa_v2_rust wheel was found in $dist."
}

Write-Host "Installing: $($wheel.FullName)"
& $python -m pip install --force-reinstall $wheel.FullName
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Checking Python files and Rust import..."
& $python -m py_compile .\rust_router_core.py .\hybrid_level3.py .\hybrid_level3_v2.py .\layout_seed.py .\routing_metrics.py .\qaoa_hybrid\plugin.py .\examples\basic.py .\plugin_smoke.py .\research_ibm_full.py .\research_scaling.py .\research_no_reroute.py .\research_commuting_baseline.py .\research_dense_rescue.py .\research_final_holdout.py .\research_medium24_rescue.py .\research_32plus.py .\research_32q_beam_ab.py .\research_large_smoke.py .\research_line_order_ablation.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -c "import qaoa_v2_rust; print('qaoa_v2_rust version:', qaoa_v2_rust.__version__)"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Build succeeded. Run:"
Write-Host "  $python .\research_32plus.py --help"
