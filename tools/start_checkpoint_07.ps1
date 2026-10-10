<#
Checkpoint 07 unattended three-branch launcher (CatBoost, Local07, Prithvi).
  .\tools\start_checkpoint_07.ps1 -Check
  .\tools\start_checkpoint_07.ps1 -Smoke
  .\tools\start_checkpoint_07.ps1 -Start
  .\tools\start_checkpoint_07.ps1 -Status
A reboot/sign-out may terminate the detached job; -Start resumes it.
#>
[CmdletBinding(DefaultParameterSetName = "Check")]
param(
    [Parameter(ParameterSetName = "Check")]
    [switch]$Check,
    [Parameter(ParameterSetName = "Start", Mandatory = $true)]
    [switch]$Start,
    [Parameter(ParameterSetName = "Smoke", Mandatory = $true)]
    [switch]$Smoke,
    [Parameter(ParameterSetName = "Status", Mandatory = $true)]
    [switch]$Status,
    [int]$CatBoostTrials = 48,
    [int]$LocalTrials = 48,
    [int]$Threads = 8,
    [ValidateSet("300", "600")]
    [string]$PrithviModel = "300",
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $repo "reports\checkpoint_07_three"
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory = Join-Path $repo $OutputDirectory
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$runner = "tools\run_checkpoint_07_three.py"
$arguments = @(
    "--output", $OutputDirectory,
    "--catboost-trials", "$CatBoostTrials",
    "--local-trials", "$LocalTrials",
    "--threads", "$Threads",
    "--prithvi-model", $PrithviModel
)
$python = (Get-Command python -ErrorAction Stop).Source

if ($Status) {
    & $python $runner --status @arguments
    $pidFile = Join-Path $OutputDirectory "worker.pid"
    if (Test-Path $pidFile) {
        $workerId = [int](Get-Content $pidFile -Raw)
        $process = Get-CimInstance Win32_Process -Filter "ProcessId = $workerId" -ErrorAction SilentlyContinue
        $active = ($null -ne $process -and $process.CommandLine -match "run_checkpoint_07_three.py")
        Write-Host ("Worker process active: " + $active + " (PID " + $workerId + ")")
    }
    exit $LASTEXITCODE
}
Write-Host "Repo: $repo"
Write-Host "Python: $python"
Write-Host "Output: $OutputDirectory"
if ($env:CONDA_DEFAULT_ENV -ne "geocebada") {
    throw "Activate conda first: conda activate geocebada"
}
& $python $runner --check @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint 07 preflight failed; no training started."
}
if ($Smoke) {
    & $python $runner --smoke @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "GPU/Prithvi smoke failed; full run NOT started."
    }
    Write-Host "Three-branch smoke tests passed."
    exit 0
}
if (-not $Start) {
    Write-Host "Checks passed. Start with: .\tools\start_checkpoint_07.ps1 -Start"
    exit 0
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$pidFile = Join-Path $OutputDirectory "worker.pid"
if (Test-Path $pidFile) {
    $workerId = [int](Get-Content $pidFile -Raw)
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $workerId" -ErrorAction SilentlyContinue
    if ($null -ne $process -and $process.CommandLine -match "run_checkpoint_07_three.py") {
        throw "Worker already running with PID $workerId. Do not start twice."
    }
}
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$out = Join-Path $OutputDirectory "worker_$stamp.out.log"
$err = Join-Path $OutputDirectory "worker_$stamp.err.log"
$quotedOutput = '"' + $OutputDirectory + '"'
$startArgs = @(
    "-u", $runner, "--run", "--output", $quotedOutput,
    "--catboost-trials", "$CatBoostTrials",
    "--local-trials", "$LocalTrials",
    "--threads", "$Threads",
    "--prithvi-model", $PrithviModel
)
$job = Start-Process -FilePath $python -ArgumentList $startArgs -WorkingDirectory $repo -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
$job.Id | Set-Content -Encoding ascii $pidFile
Write-Host ("Checkpoint 07 detached run started. PID: " + $job.Id)
Write-Host "CatBoost, Local07 and Prithvi will run sequentially, automatically."
Write-Host "Worker log: $out"
Write-Host "Errors: $err"
Write-Host 'Status: .\tools\start_checkpoint_07.ps1 -Status'
Write-Host "Leave the VM running: reboot/sign-out may stop the worker."
