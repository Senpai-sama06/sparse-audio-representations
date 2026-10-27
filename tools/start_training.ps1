param(
    [string]$Config = "configs/librispeech_sparse_autoencoder_10h.yaml",
    [switch]$SkipPrepare
)

$ErrorActionPreference = "Stop"
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$configPath = if ([IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $workspaceRoot $Config }
$configFile = (Resolve-Path -LiteralPath $configPath).Path
$pythonFile = Join-Path $workspaceRoot ".venv/Scripts/python.exe"
$entryFile = Join-Path $workspaceRoot "run.py"
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ")
$stem = [IO.Path]::GetFileNameWithoutExtension($configFile)
$logBase = Join-Path $workspaceRoot "runs/$($stem)_$($stamp)"
New-Item -ItemType Directory -Path (Join-Path $workspaceRoot "runs") -Force | Out-Null

$arguments = @("-u", ('"{0}"' -f $entryFile), "experiment", "--config", ('"{0}"' -f $configFile))
if ($SkipPrepare) { $arguments += "--skip-prepare" }
$trainingProcess = Start-Process -FilePath $pythonFile -ArgumentList $arguments `
    -WorkingDirectory $workspaceRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$logBase.log" -RedirectStandardError "$logBase.stderr.log"

$record = [ordered]@{
    process_id = $trainingProcess.Id
    started_utc = (Get-Date).ToUniversalTime().ToString("o")
    config = $configFile
    stdout = "$logBase.log"
    stderr = "$logBase.stderr.log"
    skip_prepare = [bool]$SkipPrepare
    command = "python run.py experiment --config $Config" + $(if ($SkipPrepare) { " --skip-prepare" } else { "" })
}
$record | ConvertTo-Json | Set-Content -LiteralPath "$logBase.launch.json" -Encoding UTF8
$record | ConvertTo-Json
