param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$PreviousNoBytecode = $env:PYTHONDONTWRITEBYTECODE
$env:PYTHONDONTWRITEBYTECODE = "1"

Push-Location $Root
try {
    foreach ($Script in @(
        'code/build_figure6.py',
        'code/verify_figure6.py'
    )) {
        & $Python -B $Script
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
finally {
    Pop-Location
    $env:PYTHONDONTWRITEBYTECODE = $PreviousNoBytecode
}
