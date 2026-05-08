param(
    [string]$Video = "S1-2",
    [double]$Fps = 30.0,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir

$args = @("run", "lab1", "task1", "--videos", $Video, "--fps", $Fps, "--stage", "all")
if ($Force) {
    $args += "--force"
}

Write-Host "Build task1 full result: video=$Video, fps=$Fps, force=$($Force.IsPresent)"
uv @args
