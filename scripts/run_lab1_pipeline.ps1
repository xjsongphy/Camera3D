param(
    [string[]]$Tasks = @("task1", "task2", "task3"),
    [switch]$Force,
    [switch]$DryRun,
    [switch]$SkipYolo
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir
$PowerShellExe = if (Get-Command pwsh -ErrorAction SilentlyContinue) { "pwsh" } else { "powershell" }

# Valid task list
$validTasks = @("task1", "task2", "task3")

# Normalize task inputs so both -Tasks task2 task3 and -Tasks task2,task3 work.
$Tasks = @($Tasks | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })

# Validate tasks
foreach ($task in $Tasks) {
    if ($task -notin $validTasks) {
        Write-Host "Invalid task: $task" -ForegroundColor Red
        Write-Host ""
        Write-Host "Valid tasks: task1, task2, task3" -ForegroundColor Cyan
        exit 1
    }
}

# Display what will run
Write-Host "Tasks to run: $($Tasks -join ', ')" -ForegroundColor Green

# Task definitions
$task1Info = @{
    Script = "scripts/task1_fps_sweep_full.ps1"
    Desc = "Task1 FPS sweep"
}
$task2Info = @{
    Script = "scripts/task2_full_pipeline.ps1"
    Desc = "Task2 with optimized subsequences"
}
$task3Info = @{
    Script = "scripts/task3_full_pipeline.ps1"
    Desc = "Task3 with masks"
}

function Invoke-Task {
    param(
        [string]$Name,
        [string]$Script,
        [string]$Desc,
        [bool]$Force,
        [bool]$DryRun,
        [bool]$SkipYolo
    )

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Task: $Name - $Desc" -ForegroundColor Cyan
    Write-Host "========================================"

    # Build command
    $cmdList = @("-ExecutionPolicy", "Bypass", "-File", $Script)
    if ($Force) { $cmdList += "--force" }
    if ($DryRun) { $cmdList += "--dry-run" }
    if ($SkipYolo -and $Name -eq "task3") { $cmdList += "--skip-yolo" }
    Write-Host "Running: $PowerShellExe $($cmdList -join ' ')" -ForegroundColor Yellow

    if ($DryRun) {
        return $true
    }

    # Execute
    $process = Start-Process -FilePath $PowerShellExe -ArgumentList $cmdList -Wait -PassThru -NoNewWindow
    return $process.ExitCode -eq 0
}

# Main execution
Write-Host ""
Write-Host "Camera3D Lab1 Pipeline Runner"
Write-Host ""
Write-Host "Configuration:"
Write-Host "  Tasks:     $($Tasks -join ', ')"
Write-Host "  Force:     $($Force.IsPresent)"
Write-Host "  Dry Run:   $($DryRun.IsPresent)"
Write-Host "  Skip YOLO: $($SkipYolo.IsPresent)"
Write-Host ""

$failedTasks = @()
$successCount = 0

foreach ($task in $Tasks) {
    $info = $null
    switch ($task) {
        "task1" { $info = $task1Info }
        "task2" { $info = $task2Info }
        "task3" { $info = $task3Info }
    }

    if (Invoke-Task -Name $task -Script $info.Script -Desc $info.Desc -Force $Force.IsPresent -DryRun $DryRun.IsPresent -SkipYolo $SkipYolo.IsPresent) {
        $successCount++
    } else {
        $failedTasks += $task
    }
}

# Summary
Write-Host ""
Write-Host "========================================"
Write-Host "Summary"
Write-Host "========================================"
Write-Host "Total: $($Tasks.Count)"
Write-Host "Success: $successCount"
Write-Host "Failed: $($failedTasks.Count)"

if ($failedTasks.Count -gt 0) {
    Write-Host ""
    Write-Host "Failed tasks:" -ForegroundColor Red
    foreach ($t in $failedTasks) {
        Write-Host "  - $t" -ForegroundColor Red
    }
    exit 1
}

Write-Host ""
Write-Host "All tasks completed!" -ForegroundColor Green
Write-Host ""
Write-Host "Outputs:"
Write-Host "  Task1: outputs/lab1/task1/"
Write-Host "  Task2: outputs/lab1/task2/"
Write-Host "  Task3: outputs/lab1/task3/"
