param(
    [Parameter(Mandatory=$true)][string]$SourceAudio,
    [string]$Output = 'runs/parametric_isolated_40ms',
    [int]$Port = 8765
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$sourcePath = (Resolve-Path -LiteralPath $SourceAudio).Path
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $taskRoot $Output))
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
if (Test-Path -LiteralPath (Join-Path $outputPath 'launch.json')) {
    $previousLaunch = Get-Content -LiteralPath (Join-Path $outputPath 'launch.json') -Raw | ConvertFrom-Json
    $existingProcess = Get-Process -Id $previousLaunch.worker_pid -ErrorAction SilentlyContinue
    if ($existingProcess -and $existingProcess.StartTime.ToUniversalTime().ToString('o') -eq $previousLaunch.worker_started) {
        throw "Isolated worker is already running: $($previousLaunch.worker_pid)"
    }
}
$pythonPath = (Get-Command python).Source
$workerArgs = '-u -m approaches.parametric.isolated --source "{0}" --output "{1}"' -f $sourcePath,$outputPath
$worker = Start-Process -FilePath $pythonPath -ArgumentList $workerArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $outputPath 'worker.log') -RedirectStandardError (Join-Path $outputPath 'worker-error.log')
$monitorArgs = '-u -m approaches.parametric.progress "{0}" --port {1}' -f $outputPath,$Port
$monitor = Start-Process -FilePath $pythonPath -ArgumentList $monitorArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $outputPath 'monitor.log') -RedirectStandardError (Join-Path $outputPath 'monitor-error.log')
$launch = @{
    worker_pid=$worker.Id; worker_started=$worker.StartTime.ToUniversalTime().ToString('o')
    monitor_pid=$monitor.Id; dashboard="http://127.0.0.1:$Port/"
    report="http://127.0.0.1:$Port/isolated_40ms_shareable.html"; output=$outputPath
}
$launch | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputPath 'launch.json') -Encoding utf8
$launch | ConvertTo-Json
