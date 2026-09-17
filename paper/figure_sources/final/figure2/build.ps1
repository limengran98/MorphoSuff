param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Push-Location $PackageRoot
try {
    & $Python -B code/archive_manifest.py --verify-source
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -B code/rebuild_benchmark_sources.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -B code/fig2_data.py --verify-inference
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -B code/fig2_composite.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -B code/verify_printscale.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -B code/archive_manifest.py
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
