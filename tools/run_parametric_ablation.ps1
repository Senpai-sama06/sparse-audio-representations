param(
    [Parameter(Mandatory=$true)][string]$Audio,
    [string]$Output = 'runs/parametric_ablation',
    [int]$Port = 8765
)
# Launch hidden independent processes so the experiment survives the chat turn.
# Re-running with the same manifest resumes completed jobs. Use a new Output
# directory after changing experiment settings or source code.
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$audioPath = (Resolve-Path -LiteralPath $Audio).Path
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $taskRoot $Output))
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
if (Test-Path -LiteralPath (Join-Path $outputPath 'launch.json')) {
    $previousLaunch = Get-Content -LiteralPath (Join-Path $outputPath 'launch.json') -Raw | ConvertFrom-Json
    $existingProcess = Get-Process -Id $previousLaunch.worker_pid -ErrorAction SilentlyContinue
    if ($existingProcess -and $existingProcess.StartTime.ToUniversalTime().ToString('o') -eq $previousLaunch.worker_started) {
        throw "Experiment worker is already running: $($previousLaunch.worker_pid)"
    }
}
$pythonPath = (Get-Command python).Source
$workerArgs = '-u -m approaches.parametric.ablation --audio "{0}" --output "{1}"' -f $audioPath,$outputPath
$worker = Start-Process -FilePath $pythonPath -ArgumentList $workerArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $outputPath 'worker.log') -RedirectStandardError (Join-Path $outputPath 'worker-error.log')
$monitorArgs = '-u -m approaches.parametric.progress "{0}" --port {1}' -f $outputPath,$Port
$monitor = Start-Process -FilePath $pythonPath -ArgumentList $monitorArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $outputPath 'monitor.log') -RedirectStandardError (Join-Path $outputPath 'monitor-error.log')
$launch = @{
    worker_pid=$worker.Id; worker_started=$worker.StartTime.ToUniversalTime().ToString('o')
    monitor_pid=$monitor.Id; dashboard="http://127.0.0.1:$Port/"
    endpoint="http://127.0.0.1:$Port/progress.json"; output=$outputPath
}
$launch | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputPath 'launch.json') -Encoding utf8
$launch | ConvertTo-Json
