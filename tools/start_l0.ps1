param(
    [string]$Config = "configs/study_l0_librispeech.yaml",
    [string]$ResumeRun = ""
)

# Launch detached and hidden, retaining PID and both logs. No wall-time limit or
# support cap is imposed. Environment limits this activity to one CPU thread.
$ErrorActionPreference = "Stop"
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$pythonFile = Join-Path $workspaceRoot ".venv/Scripts/python.exe"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ")
if ($ResumeRun) {
    $runFolder = (Resolve-Path -LiteralPath $ResumeRun).Path
    $entryFile = Join-Path $workspaceRoot "tools/resume_l0.py"
    $arguments = @("-u", ('"{0}"' -f $entryFile), "--run", ('"{0}"' -f $runFolder))
} else {
    $runFolder = Join-Path $workspaceRoot "runs/l0_librispeech_$stamp"
    $configFile = (Resolve-Path -LiteralPath $Config).Path
    $entryFile = Join-Path $workspaceRoot "run.py"
    $arguments = @("-u", ('"{0}"' -f $entryFile), "study", "--config", ('"{0}"' -f $configFile), "--output", ('"{0}"' -f $runFolder))
}
$logBase = "$runFolder.$stamp"
$searchProcess = Start-Process -FilePath $pythonFile -ArgumentList $arguments `
    -WorkingDirectory $workspaceRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$logBase.log" -RedirectStandardError "$logBase.stderr.log"
$record = [ordered]@{
    process_id = $searchProcess.Id
    run = $runFolder
    started_utc = (Get-Date).ToUniversalTime().ToString("o")
    stdout = "$logBase.log"
    stderr = "$logBase.stderr.log"
}
$record | ConvertTo-Json | Set-Content -LiteralPath "$logBase.launch.json" -Encoding UTF8
$record | ConvertTo-Json
