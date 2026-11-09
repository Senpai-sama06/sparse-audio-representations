param(
    [string]$Source = 'runs/parametric_learned_representation_pilot',
    [string]$Output = 'runs/parametric_symbolic_imf_pilot',
    [int]$Port = 8770
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$sourcePath = (Resolve-Path -LiteralPath (Join-Path $taskRoot $Source)).Path
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $taskRoot $Output))
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$launchPath = Join-Path $outputPath 'launch.json'
if (Test-Path -LiteralPath $launchPath) {
    $previous = Get-Content -LiteralPath $launchPath -Raw | ConvertFrom-Json
    $existing = Get-Process -Id $previous.worker_pid -ErrorAction SilentlyContinue
    if ($existing -and $existing.StartTime.ToUniversalTime().ToString('o') -eq $previous.worker_started) {
        throw "Expression discovery already running: $($existing.Id)"
    }
}
$pythonPath = (Get-Command python).Source
$workerArgs = '-u -m approaches.parametric.symbolic.experiment --source "{0}" --output "{1}"' -f $sourcePath,$outputPath
$worker = Start-Process -FilePath $pythonPath -ArgumentList $workerArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $outputPath 'worker.log') -RedirectStandardError (Join-Path $outputPath 'worker-error.log')
$monitorArgs = '-u -m approaches.parametric.progress "{0}" --port {1}' -f $outputPath,$Port
$monitor = Start-Process -FilePath $pythonPath -ArgumentList $monitorArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $outputPath 'monitor.log') -RedirectStandardError (Join-Path $outputPath 'monitor-error.log')
@{worker_pid=$worker.Id;worker_started=$worker.StartTime.ToUniversalTime().ToString('o');monitor_pid=$monitor.Id;
  dashboard="http://127.0.0.1:$Port/";report="http://127.0.0.1:$Port/symbolic_imf_shareable.html";output=$outputPath
} | ConvertTo-Json | Set-Content -LiteralPath $launchPath -Encoding utf8
Get-Content -LiteralPath $launchPath
