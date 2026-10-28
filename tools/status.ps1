param([string]$Study = "runs/concurrent_training.json", [switch]$Test)

$ErrorActionPreference = "Stop"
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$studyFile = if ([IO.Path]::IsPathRooted($Study)) { $Study } else { Join-Path $workspaceRoot $Study }
$runs = (Get-Content -LiteralPath $studyFile -Raw -Encoding UTF8 | ConvertFrom-Json).runs
$rows = foreach ($run in $runs) {
    $runPath = if ([IO.Path]::IsPathRooted($run.path)) { $run.path } else { Join-Path $workspaceRoot $run.path }
    if ($Test) {
        $testProgressFile = Join-Path $runPath "evaluation/clean/complete/progress.json"
        if (Test-Path -LiteralPath $testProgressFile) {
            $evaluation = Get-Content -LiteralPath $testProgressFile -Raw -Encoding UTF8 | ConvertFrom-Json
            $done = ($evaluation.setting_index - 1) * $evaluation.total_examples + $evaluation.examples_completed
            $total = $evaluation.total_settings * $evaluation.total_examples
            [pscustomobject]@{
                Model = $run.name; Status = $evaluation.status
                Budget = $evaluation.representation_setting
                Examples = "$($evaluation.examples_completed)/$($evaluation.total_examples)"
                Percent = [math]::Round(100 * $done / $total, 1)
            }
        } else {
            [pscustomobject]@{ Model = $run.name; Status = "not started or auditing"; Budget = "-"; Examples = "-"; Percent = "-" }
        }
        continue
    }
    $progressFile = Join-Path $runPath "progress.json"
    if (Test-Path -LiteralPath $progressFile) {
        $progress = Get-Content -LiteralPath $progressFile -Raw -Encoding UTF8 | ConvertFrom-Json
        $status = $progress.status
        if ($status -eq "complete" -and -not (Test-Path -LiteralPath (Join-Path $runPath "report/index.html"))) {
            $status = "training complete"
        }
        [pscustomobject]@{
            Model = $run.name
            Status = $status
            Updates = "$($progress.step)/$($progress.total_steps)"
            Percent = if ($progress.total_steps) { [math]::Round(100 * $progress.step / $progress.total_steps, 1) } else { "-" }
            BestValidation = if ($null -ne $progress.best_validation) { [math]::Round($progress.best_validation, 5) } else { "-" }
        }
    } else {
        [pscustomobject]@{ Model = $run.name; Status = "not started or auditing"; Updates = "-"; Percent = "-"; BestValidation = "-" }
    }
}
$rows | Format-Table -AutoSize
