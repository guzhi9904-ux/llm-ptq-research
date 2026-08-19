param(
    [ValidateSet("runner", "fp4-reference")]
    [string]$Profile = "runner",
    [string]$EnvironmentPath = ".venv"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$targetPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $EnvironmentPath))

if (-not (Test-Path -LiteralPath $targetPath)) {
    python -m venv $targetPath
}

$envPython = Join-Path $targetPath "Scripts\python.exe"
& $envPython -m pip install --upgrade pip

if ($Profile -eq "runner") {
    & $envPython -m pip install -r (Join-Path $PSScriptRoot "runner-requirements.txt")
}
else {
    & $envPython -m pip install -r (Join-Path $PSScriptRoot "fp4-reference-requirements.txt")
}

Write-Output "环境已准备：$targetPath"
Write-Output "使用方式：& '$envPython' experiments/runners/ptq.py list"
