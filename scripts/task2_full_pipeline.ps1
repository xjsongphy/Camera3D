param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir

function Format-ParamTag {
    param([double]$Value)
    $s = "{0:0.###}" -f $Value
    $s = $s.TrimEnd("0").TrimEnd(".")
    return "fps" + ($s -replace "\.", "p")
}

function Test-Task1ResultReady {
    param([string]$CaseRoot)
    $requiredPaths = @(
        (Join-Path $CaseRoot "images"),
        (Join-Path $CaseRoot "frame_map.csv"),
        (Join-Path $CaseRoot "sparse/0/images.txt"),
        (Join-Path $CaseRoot "sparse/0/cameras.txt"),
        (Join-Path $CaseRoot "sparse/0/points3D.txt"),
        (Join-Path $CaseRoot "trajectory.png")
    )
    foreach ($p in $requiredPaths) {
        if (!(Test-Path $p)) { return $false }
    }
    $imageCount = (Get-ChildItem (Join-Path $CaseRoot "images") -Filter *.jpg -File -ErrorAction SilentlyContinue).Count
    return $imageCount -gt 0
}

Write-Host "Running Task2 full pipeline for S1-2."
Write-Host "Source FPS: 30"
Write-Host "Force: $($Force.IsPresent)"

$commonArgs = @()
if ($Force) {
    $commonArgs += "--force"
}
$SourceFps = 30.0
$paramTag = Format-ParamTag -Value $SourceFps
$task1CaseRoot = Join-Path "outputs/lab1/task1" "S1-2_$paramTag"

Write-Host ""
Write-Host "=== Step 1/4: Build full-sequence task1 result for S1-2 ==="
if ($Force -or !(Test-Task1ResultReady -CaseRoot $task1CaseRoot)) {
    uv run lab1 task1 --videos S1-2 --fps $SourceFps --stage all @commonArgs
} else {
    Write-Host "Reuse existing task1 result: $task1CaseRoot"
}

Write-Host ""
Write-Host "=== Step 2/4: Prepare task2 subsequences (method A subset + method B frames) ==="
uv run lab1 task2 --source-fps $SourceFps --stage prepare @commonArgs

Write-Host ""
Write-Host "=== Step 3/4: Run task2 subset SfM (method B) ==="
uv run lab1 task2 --source-fps $SourceFps --stage sfm @commonArgs

Write-Host ""
Write-Host "=== Step 4/4: Analyze alignment and ATE ==="
uv run lab1 task2 --source-fps $SourceFps --stage analyze @commonArgs

$task2Root = Join-Path "outputs/lab1/task2" "S1-2_$paramTag"
$summaryCsv = Join-Path $task2Root "summary.csv"

if (!(Test-Path $summaryCsv)) {
    throw "Missing summary.csv: $summaryCsv"
}

$rows = Import-Csv $summaryCsv
if ($rows.Count -eq 0) {
    throw "summary.csv has no rows: $summaryCsv"
}

Write-Host ""
Write-Host "=== Task2 ATE Summary ==="
foreach ($row in $rows) {
    $subseq = $row.subseq
    $subsetFrames = $row.subset_frames
    $commonReg = $row.common_registered
    $ate = [double]$row.ate
    $scale = [double]$row.scale
    Write-Host ("{0} | subset={1}, common={2}, ATE={3:F6}, scale={4:F6}" -f $subseq, $subsetFrames, $commonReg, $ate, $scale)
}

Write-Host ""
Write-Host "Task2 outputs:"
Write-Host ("- Root: " + (Resolve-Path $task2Root).Path)
Write-Host ("- Summary CSV: " + (Resolve-Path $summaryCsv).Path)
Write-Host "- Per-subsequence files: method_a/, method_b/, trajectory_overlay.png, metrics.txt, timing.csv"
