param(
    [Parameter(Mandatory=$true)][string]$CorpusRoot,
    [string]$Output = 'runs/parametric_learned_representation',
    [int]$Port = 8768,
    [int]$ArchivePort = 8769
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$corpusPath = (Resolve-Path -LiteralPath $CorpusRoot).Path
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $taskRoot $Output))
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
if (Test-Path -LiteralPath (Join-Path $outputPath 'launch.json')) {
    $previousLaunch = Get-Content -LiteralPath (Join-Path $outputPath 'launch.json') -Raw | ConvertFrom-Json
    $existing = Get-Process -Id $previousLaunch.worker_pid -ErrorAction SilentlyContinue
    if ($existing -and $existing.StartTime.ToUniversalTime().ToString('o') -eq $previousLaunch.worker_started) {
        throw "Training already running: $($existing.Id)"
    }
}
$pythonPath = (Get-Command python).Source
$workerArgs = '-u -m approaches.parametric.learned --corpus-root "{0}" --output "{1}"' -f $corpusPath,$outputPath
$worker = Start-Process -FilePath $pythonPath -ArgumentList $workerArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $outputPath 'worker.log') -RedirectStandardError (Join-Path $outputPath 'worker-error.log')
$monitorArgs = '-u -m approaches.parametric.progress "{0}" --port {1}' -f $outputPath,$Port
$monitor = Start-Process -FilePath $pythonPath -ArgumentList $monitorArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $outputPath 'monitor.log') -RedirectStandardError (Join-Path $outputPath 'monitor-error.log')
$archiveArgs = '-u -m approaches.parametric.progress runs/parametric_reports --port {0}' -f $ArchivePort
$archive = Start-Process -FilePath $pythonPath -ArgumentList $archiveArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $outputPath 'archive-server.log') -RedirectStandardError (Join-Path $outputPath 'archive-server-error.log')
$launch = @{
    worker_pid=$worker.Id; worker_started=$worker.StartTime.ToUniversalTime().ToString('o'); monitor_pid=$monitor.Id; archive_pid=$archive.Id
    dashboard="http://127.0.0.1:$Port/"; report="http://127.0.0.1:$Port/learned_representation_shareable.html"
    archive="http://127.0.0.1:$ArchivePort/index.html"; output=$outputPath
}
$launch | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputPath 'launch.json') -Encoding utf8
$launch | ConvertTo-Json
