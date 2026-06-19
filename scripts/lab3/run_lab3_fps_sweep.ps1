$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RootDir

# All parameters live in this block; edit a value and run the script directly.
# No command-line args, no JSON config. discover_inputs() recurses into
# InputDir and processes every image/video it finds, so just point InputDir
# at the folder containing your captures.
$Sweep = [ordered]@{
    InputDir        = "input/lab3_dormitory_input"
    SceneName       = "dormitory"
    OutputRoot      = "outputs/lab3"
    Methods         = @("sfm", "3dgs", "nerf")
    FpsList         = @(2.0, 5.0)
    TestRatio       = 0.1
    ImageLimit      = $null
    FfmpegBin       = "ffmpeg"
    ColmapBin       = "colmap"
    DgsRepo         = "gaussian-splatting"
    DgsIterations   = 15000
    NerfIterations  = 60000
    SharePoses      = $true
    Evaluate        = $true
    Geometry        = $true
    Qualitative     = $true
    Lpips           = $true
    Force           = $false
    DryRun          = $false
}

$script:RunSummary = @()

function Format-FpsTag {
    param([double]$Fps)
    $text = $Fps.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    return ($text -replace "\.", "p")
}

function Build-Lab3Args {
    param(
        [hashtable]$Cfg,
        [double]$Fps,
        [bool]$DryRunMode
    )

    $sceneTag = "{0}_fps{1}" -f $Cfg.SceneName, (Format-FpsTag $Fps)
    $argsList = @(
        "--input-dir", $Cfg.InputDir,
        "--scene-name", $sceneTag,
        "--output-root", $Cfg.OutputRoot,
        "--methods"
    )
    $argsList += $Cfg.Methods
    $argsList += @(
        "--fps", $Fps.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--test-ratio", $Cfg.TestRatio.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--ffmpeg-bin", $Cfg.FfmpegBin,
        "--colmap-bin", $Cfg.ColmapBin,
        "--dgs-repo", $Cfg.DgsRepo,
        "--dgs-iterations", "$($Cfg.DgsIterations)",
        "--nerf-iterations", "$($Cfg.NerfIterations)"
    )

    if ($null -ne $Cfg.ImageLimit) {
        $argsList += @("--image-limit", "$($Cfg.ImageLimit)")
    }
    if ($Cfg.SharePoses) { $argsList += "--share-poses" } else { $argsList += "--no-share-poses" }
    if ($Cfg.Evaluate) { $argsList += "--evaluate" } else { $argsList += "--no-evaluate" }
    if ($Cfg.Geometry) { $argsList += "--geometry" } else { $argsList += "--no-geometry" }
    if ($Cfg.Qualitative) { $argsList += "--qualitative" } else { $argsList += "--no-qualitative" }
    if ($Cfg.Lpips) { $argsList += "--lpips" } else { $argsList += "--no-lpips" }
    if ($Cfg.Force) { $argsList += "--force" }
    if ($DryRunMode) { $argsList += "--dry-run" }

    return ,$argsList
}

function Invoke-Lab3SweepRun {
    param(
        [hashtable]$Cfg,
        [double]$Fps,
        [bool]$DryRunMode
    )

    $sceneTag = "{0}_fps{1}" -f $Cfg.SceneName, (Format-FpsTag $Fps)
    $argsList = Build-Lab3Args -Cfg $Cfg -Fps $Fps -DryRunMode $DryRunMode

    Write-Host ""
    Write-Host ("=== lab3 / scene={0} / fps={1} ===" -f $sceneTag, $Fps)
    Write-Host ("uv run lab3 " + ($argsList -join " "))

    & uv run lab3 @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "lab3 failed for scene=$sceneTag fps=$Fps"
    }

    $latestRun = $null
    if (Test-Path $Cfg.OutputRoot) {
        $latestRun = Get-ChildItem $Cfg.OutputRoot -Directory |
            Where-Object { $_.Name -match ("^\d{{8}}_\d{{6}}_{0}$" -f [regex]::Escape($sceneTag)) } |
            Sort-Object LastWriteTime |
            Select-Object -Last 1
    }

    $script:RunSummary += [pscustomobject]@{
        scene_name = $sceneTag
        fps = $Fps
        run_dir = if ($latestRun) { $latestRun.FullName } else { "" }
        dry_run = $DryRunMode
    }
}

Write-Host "Running lab3 full reconstruction sweep sequentially."
Write-Host ("InputDir: " + $Sweep.InputDir)
Write-Host ("Base scene: " + $Sweep.SceneName)
Write-Host ("FPS list: " + (($Sweep.FpsList | ForEach-Object { $_.ToString([System.Globalization.CultureInfo]::InvariantCulture) }) -join ", "))
if ($Sweep.DryRun) { Write-Host "DRY RUN - lab3 will print commands without training." }

foreach ($fps in $Sweep.FpsList) {
    Invoke-Lab3SweepRun -Cfg $Sweep -Fps $fps -DryRunMode $Sweep.DryRun
}

Write-Host ""
Write-Host "Sweep finished."
$script:RunSummary | Format-Table -AutoSize
