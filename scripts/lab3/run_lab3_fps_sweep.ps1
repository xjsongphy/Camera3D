$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RootDir

# Sweep launcher.
# Keep the main experiment settings in ConfigPath; this script only points at
# a concrete input directory and overrides the fps/scene tag per run.
$Sweep = [ordered]@{
    ConfigPath      = "configs/lab3/extra.json"
    InputDir        = "input/lab3_dormitory_input"
    SceneName       = "dormitory"
    FpsList         = @(2.0, 5.0)
    OutputRoot      = ""
    Methods         = @()
    ImageLimit      = $null
    BlurThreshold   = $null
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
        "--config", $Cfg.ConfigPath,
        "--input-dir", $Cfg.InputDir,
        "--scene-name", $sceneTag
    )
    $argsList += @(
        "--fps", $Fps.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    )

    if (-not [string]::IsNullOrWhiteSpace($Cfg.OutputRoot)) {
        $argsList += @("--output-root", $Cfg.OutputRoot)
    }
    if ($Cfg.Methods.Count -gt 0) {
        $argsList += "--methods"
        $argsList += $Cfg.Methods
    }
    if ($null -ne $Cfg.ImageLimit) {
        $argsList += @("--image-limit", "$($Cfg.ImageLimit)")
    }
    if ($null -ne $Cfg.BlurThreshold) {
        $argsList += @("--blur-threshold", "$($Cfg.BlurThreshold)")
    }
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
    $outputRoot = if ([string]::IsNullOrWhiteSpace($Cfg.OutputRoot)) { "outputs/lab3" } else { $Cfg.OutputRoot }
    if (Test-Path $outputRoot) {
        $latestRun = Get-ChildItem $outputRoot -Directory |
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
Write-Host ("Config: " + $Sweep.ConfigPath)
Write-Host ("InputDir: " + $Sweep.InputDir)
Write-Host ("Base scene: " + $Sweep.SceneName)
Write-Host ("FPS list: " + (($Sweep.FpsList | ForEach-Object { $_.ToString([System.Globalization.CultureInfo]::InvariantCulture) }) -join ", "))
if ($null -ne $Sweep.BlurThreshold) { Write-Host ("Blur threshold: " + $Sweep.BlurThreshold) }
if ($Sweep.DryRun) { Write-Host "DRY RUN - lab3 will print commands without training." }

foreach ($fps in $Sweep.FpsList) {
    Invoke-Lab3SweepRun -Cfg $Sweep -Fps $fps -DryRunMode $Sweep.DryRun
}

Write-Host ""
Write-Host "Sweep finished."
$script:RunSummary | Format-Table -AutoSize
