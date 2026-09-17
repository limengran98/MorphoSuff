param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

Push-Location $Root
try {
    foreach ($index in 1..6) {
        $builder = Join-Path $Root "figure$index\build.ps1"
        if (-not (Test-Path -LiteralPath $builder)) {
            throw "Missing builder: $builder"
        }
        & $builder -Python $Python
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    & $Python -B validate_clean_package.py
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
