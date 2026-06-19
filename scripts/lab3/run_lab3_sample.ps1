param(
    [string]$InputDir = "docs/lab1/assets/videos",
    [string]$SceneName = "lab1_sample",
    [string]$DgsRepo = "assets/external/code/gaussian-splatting",
    [double]$Fps = 2.0,
    [int]$DgsIterations = 7000,
    [int]$NerfIterations = 30000,
    [int]$NeusIterations = 20001,
    [switch]$DryRun
)

$argsList = @(
    "--input-dir", $InputDir,
    "--scene-name", $SceneName,
    "--methods", "sfm", "3dgs", "nerf", "neus",
    "--fps", "$Fps",
    "--dgs-repo", $DgsRepo,
    "--dgs-iterations", "$DgsIterations",
    "--nerf-iterations", "$NerfIterations",
    "--neus-iterations", "$NeusIterations"
)

if ($DryRun) {
    $argsList += "--dry-run"
}

uv run lab3 @argsList
