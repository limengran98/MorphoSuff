param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PreviousNoBytecode = $env:PYTHONDONTWRITEBYTECODE
$env:PYTHONDONTWRITEBYTECODE = "1"

Push-Location $PackageRoot
try {
    & $Python -B code/build_figure3.py --panel all
    if ($LASTEXITCODE -ne 0) { throw "Figure 3 build failed with exit code $LASTEXITCODE" }

    & $Python -B code/verify_figure3.py
    if ($LASTEXITCODE -ne 0) { throw "Figure 3 QA failed with exit code $LASTEXITCODE" }

    & $Python -B code/archive_manifest.py
    if ($LASTEXITCODE -ne 0) { throw "Figure 3 manifest generation failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
    $env:PYTHONDONTWRITEBYTECODE = $PreviousNoBytecode
}
