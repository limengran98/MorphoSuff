param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PreviousNoBytecode = $env:PYTHONDONTWRITEBYTECODE
$env:PYTHONDONTWRITEBYTECODE = "1"

Push-Location $PackageRoot
try {
    & $Python -B code/build_panel_ab.py
    if ($LASTEXITCODE -ne 0) { throw "Figure 4 panels a/b failed with exit code $LASTEXITCODE" }

    & $Python -B code/build_panel_c.py
    if ($LASTEXITCODE -ne 0) { throw "Figure 4 panel c failed with exit code $LASTEXITCODE" }

    & $Python -B code/build_panel_d.py
    if ($LASTEXITCODE -ne 0) { throw "Figure 4 panel d failed with exit code $LASTEXITCODE" }

    & $Python -B code/build_figure4.py
    if ($LASTEXITCODE -ne 0) { throw "Figure 4 composite failed with exit code $LASTEXITCODE" }

    & $Python -B code/verify_figure4.py
    if ($LASTEXITCODE -ne 0) { throw "Figure 4 scientific/output QA failed with exit code $LASTEXITCODE" }

    & $Python -B code/verify_figure4_archive.py
    if ($LASTEXITCODE -ne 0) { throw "Figure 4 archive QA failed with exit code $LASTEXITCODE" }

    & $Python -B code/archive_manifest.py
    if ($LASTEXITCODE -ne 0) { throw "Figure 4 manifest generation failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
    $env:PYTHONDONTWRITEBYTECODE = $PreviousNoBytecode
}
