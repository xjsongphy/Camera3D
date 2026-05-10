param(
    [switch]$Force,
    [switch]$DryRun,
    [switch]$SkipYolo
)

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $RootDir

$sources = @("default", "motion")
if (-not $SkipYolo) {
    $sources += "yolo"
}

$videos = @("S2-1", "S2-2")
$fps = "30"

Write-Host "Running full Task3 pipeline for S2-1 and S2-2."
Write-Host ("Mask sources: " + ($sources -join ", "))
if ($Force) {
    Write-Host "Force: true"
}
if ($DryRun) {
    Write-Host "Dry run: true"
}
if ($SkipYolo) {
    Write-Host "Skip YOLO: true"
} else {
    Write-Host "Ensuring YOLO dependency (uv extra: task3-yolo)..."
    uv sync --extra task3-yolo
}

foreach ($source in $sources) {
    $maskArgs = @(
        "lab1", "task3-mask",
        "--source", $source,
        "--videos"
    ) + $videos + @(
        "--fps", $fps
    )

    if ($Force) {
        $maskArgs += "--force"
    }
    if ($DryRun) {
        $maskArgs += "--dry-run"
    }

    Write-Host ""
    Write-Host ("[1/2] Generating masks: " + $source)
    uv run @maskArgs
}

$rawArgs = @(
    "lab1", "task3",
    "--videos"
) + $videos + @(
    "--fps", $fps,
    "--stage", "all",
    "--methods", "raw"
)
if ($Force) {
    $rawArgs += "--force"
}
if ($DryRun) {
    $rawArgs += "--dry-run"
}

Write-Host ""
Write-Host "[2/2] Running reconstruction: raw"
uv run @rawArgs

foreach ($source in $sources) {
    $task3Args = @(
        "lab1", "task3",
        "--videos"
    ) + $videos + @(
        "--fps", $fps,
        "--stage", "all",
        "--methods", "mask",
        "--mask-source", $source
    )

    if ($Force) {
        $task3Args += "--force"
    }
    if ($DryRun) {
        $task3Args += "--dry-run"
    }

    Write-Host ""
    Write-Host ("[2/2] Running reconstruction: mask + " + $source)
    uv run @task3Args
}

Write-Host ""
Write-Host "Task3 outputs:"
Write-Host "- outputs/lab1/task3/S2-1_fps${fps}"
Write-Host "- outputs/lab1/task3/S2-2_fps${fps}"
