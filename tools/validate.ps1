$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    python tools/validate_repository.py
    python -m pytest -q
}
finally {
    Pop-Location
}
