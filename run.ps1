param(
    [ValidateSet("quick", "standard", "stress")]
    [string]$Profile = "quick"
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here

# Try the project/working venv first so I do not accidentally benchmark with a random Python install.
$python = $null
$candidates = @(
    "..\..\work\venv\Scripts\python.exe",
    ".\.venv\Scripts\python.exe",
    ".\venv\Scripts\python.exe"
)

foreach ($candidate in $candidates) {
    if (Test-Path $candidate) {
        $python = $candidate
        break
    }
}

if (-not $python) {
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) {
        $python = $command.Source
    }
}

if (-not $python) {
    throw "Python was not found. Open the project in the same Cursor setup you used before, or create a venv first."
}

Write-Host "Python: $python"
Write-Host "Profile: $Profile"
# This is a cheap check, so a typo fails before I waste time on the real benchmark.
Write-Host "Checking the two active Python files before the benchmark..."
& $python -m py_compile .\hybrid_level3.py .\benchmark.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Starting stock Qiskit Level 3 vs Hybrid Level 3. Both benchmark suites run automatically."
& $python .\benchmark.py --profile $Profile
exit $LASTEXITCODE
