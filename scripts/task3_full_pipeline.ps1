param(
    [switch]$Force,
    [switch]$DryRun,
    [string[]]$Methods = @("raw"),
    [string]$SemanticMaskRoot = ""
)

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $RootDir

$argsList = @(
    "lab1", "task3",
    "--videos", "S2-1", "S2-2",
    "--fps", "5",
    "--stage", "all",
    "--methods"
) + $Methods

if ($Force) {
    $argsList += "--force"
}
if ($DryRun) {
    $argsList += "--dry-run"
}
if ($SemanticMaskRoot) {
    $argsList += @("--semantic-mask-root", $SemanticMaskRoot)
}

Write-Host "Running Task3 for S2-1 and S2-2."
Write-Host ("Methods: " + ($Methods -join ", "))
if ($Force) {
    Write-Host "Force: true"
}
if ($DryRun) {
    Write-Host "Dry run: true"
}
if ($SemanticMaskRoot) {
    Write-Host ("Semantic mask root: " + $SemanticMaskRoot)
}

uv run @argsList

Write-Host ""
Write-Host "Task3 outputs:"
Write-Host "- outputs/lab1/task3/S2-1_fps5"
Write-Host "- outputs/lab1/task3/S2-2_fps5"
