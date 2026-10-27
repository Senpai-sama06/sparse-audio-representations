param([switch]$InstallDependencies)
$ErrorActionPreference = 'Stop'
$ProjectDirectory = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $ProjectDirectory
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        if ($InstallDependencies) {
            python -m venv .venv
        } else {
            python -m venv --system-site-packages .venv
        }
        if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
    }
    # TensorBoard 2.20 still imports pkg_resources, removed in setuptools 82+.
    & '.\.venv\Scripts\python.exe' -m pip install 'setuptools==80.9.0' 'wheel==0.48.0'
    if ($LASTEXITCODE -ne 0) { throw 'Build-tool installation failed' }
    if ($InstallDependencies) {
        & '.\.venv\Scripts\python.exe' -m pip install -r requirements-lock.txt
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
    }
    & '.\.venv\Scripts\python.exe' -m pip install --no-deps --no-build-isolation -e .
    if ($LASTEXITCODE -ne 0) { throw 'Package installation failed' }
    & '.\.venv\Scripts\python.exe' -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
} finally {
    Pop-Location
}
