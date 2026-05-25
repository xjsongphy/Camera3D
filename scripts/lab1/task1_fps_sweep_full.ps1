param(
    [string[]]$Videos = @("S1-1", "S1-2", "S1-3")
)

$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir

$FpsList = @(30)
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SummaryDir = "outputs/lab1/task1/benchmarks"
$SummaryCsv = Join-Path $SummaryDir "task1_full_sweep_${Stamp}.csv"

New-Item -ItemType Directory -Path $SummaryDir -Force | Out-Null

$rows = @()

function Get-TimingValue {
    param(
        [string]$CsvPath,
        [string]$Stage
    )
    $entry = Import-Csv $CsvPath | Where-Object { $_.stage -eq $Stage } | Select-Object -First 1
    if ($null -eq $entry) { return "" }
    return $entry.seconds
}

function Get-RegisteredFramesCount {
    param([string]$ImagesTxtPath)
    $count = 0
    $lines = Get-Content $ImagesTxtPath
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = $lines[$i].Trim()
        if ($line.Length -eq 0 -or $line.StartsWith("#")) {
            continue
        }
        if ($line -match '^\d+\s+') {
            $count += 1
            $i += 1
        }
    }
    return $count
}

Write-Host "Running task1 full pipeline sweep sequentially (no parallelism)."
Write-Host ("Videos: " + ($Videos -join ", "))
Write-Host ("FPS: " + ($FpsList -join ", "))
Write-Host ("Summary: " + $SummaryCsv)

foreach ($video in $Videos) {
    foreach ($fps in $FpsList) {
        Write-Host ""
        Write-Host "=== task1 / video=$video / fps=$fps / stage=all ==="
        uv run lab1 task1 --videos $video --fps $fps --stage all --force
        uv run lab1 task1 cloud --videos $video --fps $fps --force

        $paramTag = "fps$fps"
        $caseRoot = Join-Path "outputs/lab1/task1" "${video}_${paramTag}"
        $timingCsv = Join-Path $caseRoot "timing.csv"
        $trajPng = Join-Path $caseRoot "trajectory.png"
        $cloudPng = Join-Path $caseRoot "sparse_points.png"
        $imagesTxt = Join-Path $caseRoot "sparse/0/images.txt"

        if (!(Test-Path $timingCsv)) { throw "Missing timing file: $timingCsv" }
        if (!(Test-Path $imagesTxt)) { throw "Missing sparse poses file: $imagesTxt" }
        if (!(Test-Path $cloudPng)) { throw "Missing sparse point cloud plot: $cloudPng" }

        $extract = Get-TimingValue -CsvPath $timingCsv -Stage "extract"
        $feature = Get-TimingValue -CsvPath $timingCsv -Stage "feature_extractor"
        $matcher = Get-TimingValue -CsvPath $timingCsv -Stage "sequential_matcher"
        $hmapper = Get-TimingValue -CsvPath $timingCsv -Stage "hierarchical_mapper"
        $converter = Get-TimingValue -CsvPath $timingCsv -Stage "model_converter"
        $sfm = Get-TimingValue -CsvPath $timingCsv -Stage "sfm_total"
        $registered = Get-RegisteredFramesCount -ImagesTxtPath $imagesTxt
        $trajOk = if (Test-Path $trajPng) { "yes" } else { "no" }

        $rows += [pscustomobject]@{
            video = $video
            fps = $fps
            extract_s = $extract
            feature_extractor_s = $feature
            sequential_matcher_s = $matcher
            hierarchical_mapper_s = $hmapper
            model_converter_s = $converter
            sfm_total_s = $sfm
            registered_frames = $registered
            trajectory_png = $trajOk
            timing_csv = $timingCsv
        }
    }
}

$rows | Export-Csv -Path $SummaryCsv -NoTypeInformation -Encoding UTF8
Write-Host ""
Write-Host "Sweep finished."
Write-Host ("Summary CSV: " + $SummaryCsv)
