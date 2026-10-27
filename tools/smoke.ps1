$ErrorActionPreference = 'Stop'
$ProjectDirectory = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $ProjectDirectory
try {
    $PythonExecutable = Join-Path $ProjectDirectory '.venv\Scripts\python.exe'
    & $PythonExecutable -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
    & $PythonExecutable run.py experiment --config configs/smoke_orthogonal.yaml
    if ($LASTEXITCODE -ne 0) { throw 'Orthogonal dictionary smoke run failed' }
    & $PythonExecutable run.py experiment --config configs/smoke_autoencoder.yaml
    if ($LASTEXITCODE -ne 0) { throw 'Functional autoencoder smoke run failed' }
} finally {
    Pop-Location
}
