param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$PreviousNoBytecode = $env:PYTHONDONTWRITEBYTECODE
$env:PYTHONDONTWRITEBYTECODE = "1"

Push-Location $Root
try {
    & $Python -B code/build_panel_a.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -B code/build_panel_b.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -B c_amplitude_spectrum/code/build_panel_c.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -B code/build_panel_d.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -B code/build_panel_e.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -B code/build_figure5.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -B code/verify_figure5.py
    exit $LASTEXITCODE
}
finally {
    Pop-Location
    $env:PYTHONDONTWRITEBYTECODE = $PreviousNoBytecode
}
