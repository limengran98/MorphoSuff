param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$PreviousNoBytecode = $env:PYTHONDONTWRITEBYTECODE
$PreviousPythonPath = $env:PYTHONPATH
$Paper = Split-Path -Parent (Split-Path -Parent $Root)
$LocalPackages = Join-Path $Paper "tools\figure-python-packages"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONPATH = if ($PreviousPythonPath) { "$LocalPackages;$PreviousPythonPath" } else { $LocalPackages }

Push-Location $Root
try {
    foreach ($Script in @(
        'a_low_label_calibration/build_panel_d.py',
        'b_representation_boundary/build_panel_e.py',
        'code/build_supplementary_figure2.py'
    )) {
        & $Python -B $Script
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
finally {
    Pop-Location
    $env:PYTHONDONTWRITEBYTECODE = $PreviousNoBytecode
    $env:PYTHONPATH = $PreviousPythonPath
}
