<#
.SYNOPSIS
    Run Camera3D Lab1 pipeline tasks in sequence.

.DESCRIPTION
    Executes specified Lab1 tasks (task1, task2, task3) in dependency order.
    Skips already completed tasks unless --force is used.

.PARAMETER Tasks
    Array of task names to run. Valid values: "task1", "task2", "task3"
    Default: @("task1", "task2", "task3")

.PARAMETER Force
    Force rerun all tasks even if already completed.

.PARAMETER DryRun
    Show what would be executed without actually running.

.PARAMETER SkipYolo
    Skip YOLO mask generation in task3.

.EXAMPLE
    # Run all tasks
    .\scripts\run_lab1_pipeline.ps1

    # Run only task2 and task3
    .\scripts\run_lab1_pipeline.ps1 -Tasks task2,task3

    # Run only task1 with dry-run
    .\scripts\run_lab1_pipeline.ps1 -Tasks task1 --dry-run
#>
param(
    [string[]$Tasks = @("task1", "task2", "task3"),
    [switch]$Force,
    [switch]$DryRun,
    [switch]$SkipYolo
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot | Split-Path -Parent
Set-Location $RootDir

# Valid task list
$validTasks = @("task1", "task2", "task3")

# Validate tasks
foreach ($task in $Tasks) {
    if ($task -notin $validTasks) {
        Write-Host "Invalid task: $task" -ForegroundColor Red
        Write-Host "`nValid tasks: $($validTasks -join ', ')" -ForegroundColor Cyan
        exit 1
    }
}

# Display what will run
Write-Host "Tasks to run: $($Tasks -join ', ')" -ForegroundColor Green

# Task definitions with their scripts and check functions
$tasks = @(
    @{
        Name = "task1"
        Description = "Task1 FPS sweep (4, 8, 16, 30 fps) for S1-1, S1-2, S1-3"
        Script = "scripts/task1_fps_sweep_full.ps1"
        Args = @()
        CheckOutput = {
            param($Root)
            # Check if all task1 fps sweep outputs exist
            $fpsList = @("4", "8", "16", "30")
            $videos = @("S1-1", "S1-2", "S1-3")
            foreach ($video in $videos) {
                foreach ($fps in $fpsList) {
                    $caseRoot = Join-Path $Root "outputs/lab1/task1/${video}_fps${fps}"
                    $requiredFiles = @(
                        (Join-Path $caseRoot "sparse/0/images.txt"),
                        (Join-Path $caseRoot "sparse/0/points3D.txt"),
                        (Join-Path $caseRoot "trajectory.png"),
                        (Join-Path $caseRoot "sparse_points.png")
                    )
                    foreach ($file in $requiredFiles) {
                        if (!(Test-Path $file)) { return $false }
                    }
                }
            }
            return $true
        }
    },
    @{
        Name = "task2"
        Description = "Task2 with optimized default subsequences (return_mid, scan_stable, return_long)"
        Script = "scripts/task2_full_pipeline.ps1"
        Args = @()
        CheckOutput = {
            param($Root)
            $summaryCsv = Join-Path $Root "outputs/lab1/task2/S1-2_fps30/summary.csv"
            return (Test-Path $summaryCsv)
        }
    },
    @{
        Name = "task3"
        Description = "Task3 with raw, default, motion, yolo masks"
        Script = "scripts/task3_full_pipeline.ps1"
        Args = @()
        CheckOutput = {
            param($Root)
            # Check if at least raw and one mask method completed
            $videos = @("S2-1", "S2-2")
            foreach ($video in $videos) {
                $caseRoot = Join-Path $Root "outputs/lab1/task3/${video}_fps30"
                $rawSparse = Join-Path $caseRoot "raw/sparse/0/images.txt"
                $maskDefaultSparse = Join-Path $caseRoot "mask_default/sparse/0/images.txt"
                if (!(Test-Path $rawSparse)) { return $false }
                if (!(Test-Path $maskDefaultSparse)) { return $false }
            }
            return $true
        }
    }
)

function Invoke-Task {
    param(
        [string]$Name,
        [string]$Description,
        [string]$Script,
        [string[]]$Args,
        [scriptblock]$CheckOutput,
        [bool]$Force,
        [bool]$DryRun
    )

    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Task: $Name" -ForegroundColor Cyan
    Write-Host "Description: $Description" -ForegroundColor Cyan
    Write-Host "========================================`n"

    # Check if already completed
    if (-not $Force) {
        $isComplete = & $CheckOutput $RootDir
        if ($isComplete) {
            Write-Host "✓ Task already completed. Use --force to rerun." -ForegroundColor Green
            return $true
        }
    }

    # Build command arguments
    $commandArgs = @("pwsh", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $RootDir $Script))
    if ($Force) { $commandArgs += "--force" }
    if ($DryRun) { $commandArgs += "--dry-run" }
    if ($Args) { $commandArgs += $Args }

    # Display command
    Write-Host "Running: " -NoNewline -ForegroundColor Yellow
    Write-Host "$Script $($commandArgs[-2..($commandArgs.Length-1)] -join ' ')" -ForegroundColor DarkGray

    if ($DryRun) {
        Write-Host "[DRY RUN] Would execute task: $Name" -ForegroundColor Cyan
        return $true
    }

    # Execute
    try {
        $process = Start-Process -FilePath "pwsh" -ArgumentList $commandArgs -Wait -PassThru -NoNewWindow
        if ($process.ExitCode -ne 0) {
            Write-Host "✗ Task failed with exit code: $($process.ExitCode)" -ForegroundColor Red
            return $false
        }
        Write-Host "✓ Task completed successfully" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "✗ Task failed: $_" -ForegroundColor Red
        return $false
    }
}

# Main execution
Write-Host ""
Write-Host "╔══════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   Camera3D Complete Lab1 Pipeline Runner    ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor White
Write-Host "  Force:   $($Force.IsPresent)" -ForegroundColor White
Write-Host "  Dry Run: $($DryRun.IsPresent)" -ForegroundColor White
Write-Host "  Skip YOLO: $($SkipYolo.IsPresent)" -ForegroundColor White
Write-Host ""

$failedTasks = @()
$successCount = 0
$skippedCount = 0

foreach ($task in $tasks) {
    # Add --skip-yolo for task3 if needed
    $taskArgs = $task.Args
    if ($task.Name -eq "task3_full" -and $SkipYolo) {
        $taskArgs += "--skip-yolo"
    }

    $result = Invoke-Task `
        -Name $task.Name `
        -Description $task.Description `
        -Script $task.Script `
        -Args $taskArgs `
        -CheckOutput $task.CheckOutput `
        -Force $Force.IsPresent `
        -DryRun $DryRun.IsPresent

    if ($result) {
        $successCount++
    }
    else {
        $failedTasks += $task.Name
    }
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Pipeline Execution Summary" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Total tasks: $($tasks.Count)" -ForegroundColor White
Write-Host "  Successful: $successCount" -ForegroundColor Green
Write-Host "  Skipped:   $skippedCount" -ForegroundColor Yellow
Write-Host "  Failed:    $($failedTasks.Count)" -ForegroundColor $(if ($failedTasks.Count -gt 0) { [ConsoleColor]::Red } else { [ConsoleColor]::Green })

if ($failedTasks.Count -gt 0) {
    Write-Host "`nFailed tasks:" -ForegroundColor Red
    foreach ($taskName in $failedTasks) {
        Write-Host "  - $taskName" -ForegroundColor Red
    }
    exit 1
}

Write-Host "`n✓ All tasks completed successfully!" -ForegroundColor Green
Write-Host "`nOutput locations:" -ForegroundColor Cyan
Write-Host "  Task1: outputs/lab1/task1/" -ForegroundColor White
Write-Host "  Task2: outputs/lab1/task2/" -ForegroundColor White
Write-Host "  Task3: outputs/lab1/task3/" -ForegroundColor White
