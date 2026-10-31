param(
    [string]$Config = "configs/study_pitchflower_librispeech.yaml",
    [ValidateSet("all", "prepare", "rvq", "gs", "predictor", "calibrate")]
    [string]$Phase = "all"
)

# Foreground execution: progress bars stay visible in this terminal. Nothing is
# detached or auto-started by preparing this file. Rerun it to reuse cached data
# and resume model/optimizer checkpoints. It leaves the L0 process untouched.
$ErrorActionPreference = "Stop"
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$pythonFile = Join-Path $workspaceRoot ".venv/Scripts/python.exe"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMBA_NUM_THREADS = "1"
Push-Location $workspaceRoot
try {
    & $pythonFile -u tools/setup_pitchflower.py
    if ($LASTEXITCODE -ne 0) { throw "PitchFlower setup failed" }
    & $pythonFile -u run.py pitchflower --config $Config --phase $Phase
    if ($LASTEXITCODE -ne 0) { throw "Campaign stopped; inspect progress.json and rerun to resume" }
} finally {
    Pop-Location
}
