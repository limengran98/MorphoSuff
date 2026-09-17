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
    & $Python -B code/build_supplementary_figure1.py
    exit $LASTEXITCODE
}
finally {
    Pop-Location
    $env:PYTHONDONTWRITEBYTECODE = $PreviousNoBytecode
    $env:PYTHONPATH = $PreviousPythonPath
}
