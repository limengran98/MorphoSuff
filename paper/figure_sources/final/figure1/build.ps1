param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Push-Location $PackageRoot
try {
    & $Python -B code/archive_manifest.py --verify-source
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -B build_figure1.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -B code/verify_figure1.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -B code/archive_manifest.py
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
