# DISPOSITION: KEEP — developer convenience helper.
#
# Activates the canonical Vetinari Python environment (.venv312) in the
# current PowerShell session without requiring a manual activation step.
# Sets VETINARI_PYTHON, prepends .venv312\Scripts to PATH, and optionally
# sets VETINARI_MODELS_DIR if the models/ directory exists.
#
# This script is NOT part of the production runtime.  It is intended for
# local development and ad-hoc operator tasks.
#
# Usage (dot-source to modify the current shell):
#   . .\enter-vetinari-shell.ps1
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvScripts = Join-Path $repoRoot ".venv312\Scripts"
$pythonExe = Join-Path $venvScripts "python.exe"
$modelsDir = Join-Path $repoRoot "models"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    Write-Error "Canonical environment not found at $pythonExe"
    return
}

$env:PATH = "$venvScripts;$env:PATH"
$env:VETINARI_PYTHON = $pythonExe
if ((Test-Path -LiteralPath $modelsDir) -and -not $env:VETINARI_MODELS_DIR) {
    $env:VETINARI_MODELS_DIR = $modelsDir
}

$version = & $pythonExe -c "import sys, vetinari; print(sys.executable); print(sys.version.split()[0]); print(vetinari.__version__)"
$lines = $version -split "`r?`n"
Write-Host "Interpreter: $($lines[0])"
Write-Host "Python: $($lines[1])"
Write-Host "Vetinari: $($lines[2])"
if ($env:VETINARI_MODELS_DIR) {
    Write-Host "Models: $env:VETINARI_MODELS_DIR"
}
