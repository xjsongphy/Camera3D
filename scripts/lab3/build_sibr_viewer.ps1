param(
    [string]$SibrSource = "gaussian-splatting/SIBR_viewers",
    [string]$BuildDir = "",
    [string]$Generator = "Auto",
    [string]$Platform = "x64",
    [string]$Config = "RelWithDebInfo",
    [string]$VsDevCmd = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat",
    [string]$GeneratorInstance = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools",
    [string]$MsvcDir = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.44.35207",
    [string]$WindowsSdkDir = "C:\Program Files (x86)\Windows Kits\10",
    [string]$WindowsSdkVersion = "10.0.22621.0",
    [string]$SevenZipCmd = "",
    [switch]$NoAutoInstall7Zip,
    [switch]$Clean,
    [switch]$ForceClean,
    [switch]$OpenSolution
)

$ErrorActionPreference = "Stop"

# MSVC cl.exe localizes its diagnostics and emits them in the console's active
# codepage (GBK/936 on zh-CN; UTF-8 under chcp 65001). .NET's process-output
# reader and Write-Host must use that SAME codepage, otherwise the captured and
# displayed Chinese turns to mojibake (e.g. "" -> "鈥滃弬鏁扳€"). Read the
# console's real codepage (chcp) and pin both the decode encoding (used by the
# StreamReader below) and the display encoding ([Console]::OutputEncoding) to
# it; cl emits in whatever chcp is, so this stays correct on 936 and 65001 alike.
$consoleCodePage = 0
foreach ($token in (cmd /c chcp) -split '\s') {
    if ($token -match '^\d+$') { $consoleCodePage = [int]$token; break }
}
if ($consoleCodePage -le 0) { $consoleCodePage = [System.Console]::OutputEncoding.CodePage }
$script:BuildEncoding = [System.Text.Encoding]::GetEncoding($consoleCodePage)
[System.Console]::OutputEncoding = $script:BuildEncoding
$script:SuppressedIncludeCount = 0
$script:SuppressedEncodingWarningCount = 0
$script:SuppressedUpToDateCount = 0
$script:SuppressedRedistWarningCount = 0
$script:LastProgressKey = $null

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$sibrRoot = Resolve-Path (Join-Path $repoRoot $SibrSource)

function Get-BuildTreeProcesses {
    # Find build-tool processes whose command line references $Path. A previous
    # build_sibr_viewer run (or a hand-launched cmake) still alive here is what
    # turns `Remove-Item -Recurse` into "file in use by another process".
    param([Parameter(Mandatory)][string]$Path)

    $patBack = $Path
    $patFwd = $Path -replace '\\', '/'
    $names = @(
        'cmake.exe', 'cmd.exe', 'nmake.exe', 'cl.exe', 'nvcc.exe',
        'link.exe', 'msbuild.exe', 'ctest.exe',
        'SIBR_gaussianViewer_app.exe',
        'SIBR_gaussianViewer_app_rwdi.exe',
        'SIBR_gaussianViewer_app_config.exe'
    )
    # NOTE: WQL has no `IN` operator -- build an OR chain of `=` comparisons,
    # otherwise the filter silently matches nothing.
    $filter = ($names | ForEach-Object { "Name = '$_'" }) -join ' OR '
    $procs = Get-CimInstance Win32_Process -Filter $filter -ErrorAction SilentlyContinue

    return @($procs | Where-Object {
        $_.CommandLine -and ($_.CommandLine -like "*$patBack*" -or $_.CommandLine -like "*$patFwd*")
    } | Select-Object ProcessId, Name, CommandLine)
}

function Stop-BuildTreeProcesses {
    # taskkill /T /F tears down the whole tree (parent cmd -> cmake -> cl/nvcc).
    param([Parameter(Mandatory)]$Holders)

    foreach ($h in $Holders) {
        $null = & taskkill.exe /PID $h.ProcessId /T /F 2>$null
    }
}

function Remove-BuildTree {
    param(
        [Parameter(Mandatory)][string]$Path,
        [int]$MaxRetries = 6,
        [int]$DelayMs = 500,
        [switch]$KillHolders
    )

    if (-not (Test-Path $Path)) { return }

    $holders = @(Get-BuildTreeProcesses -Path $Path)
    if ($holders.Count -gt 0) {
        if ($KillHolders) {
            Write-Warning "[build_sibr_viewer] -Clean: killing running build that holds $Path ..."
            $holders | ForEach-Object { Write-Host ("  kill PID {0} ({1})" -f $_.ProcessId, $_.Name) -ForegroundColor DarkYellow }
            Stop-BuildTreeProcesses -Holders $holders
            Start-Sleep -Milliseconds 1200
        } else {
            Write-Host "[build_sibr_viewer] -Clean aborted: build tree is locked by a running build." -ForegroundColor Red
            Write-Host "[build_sibr_viewer] These processes reference $Path :" -ForegroundColor Yellow
            $holders | ForEach-Object { Write-Host ("  PID {0}  {1}" -f $_.ProcessId, $_.Name) -ForegroundColor Yellow }
            Write-Host "[build_sibr_viewer] Close that build (or re-run with -ForceClean to kill it), then retry." -ForegroundColor Red
            throw "Build tree '$Path' is locked by a running build (see PIDs above). Re-run with -ForceClean to kill it."
        }
    }

    for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {
        try {
            Remove-Item -Recurse -Force -LiteralPath $Path -ErrorAction Stop
            return
        } catch {
            if ($attempt -ge $MaxRetries) {
                Write-Host "[build_sibr_viewer] Failed to remove $Path after $MaxRetries attempts." -ForegroundColor Red
                Write-Host ("[build_sibr_viewer] Last error: {0}" -f $_.Exception.Message) -ForegroundColor Red
                $still = @(Get-BuildTreeProcesses -Path $Path)
                if ($still.Count -gt 0) {
                    Write-Host "[build_sibr_viewer] Still held by:" -ForegroundColor Yellow
                    $still | ForEach-Object { Write-Host ("  PID {0}  {1}" -f $_.ProcessId, $_.Name) -ForegroundColor Yellow }
                }
                throw
            }
            Start-Sleep -Milliseconds $DelayMs
        }
    }
}

if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    $buildPath = Join-Path $sibrRoot "build"
} else {
    $buildPath = Join-Path $repoRoot $BuildDir
}

if ($Clean -and (Test-Path $buildPath)) {
    Remove-BuildTree -Path $buildPath -KillHolders:$ForceClean
}

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw "cmake not found in PATH."
}
if (($Generator -eq "Auto" -or $Generator -like "Visual Studio*") -and -not (Test-Path $GeneratorInstance)) {
    throw "Visual Studio generator instance not found: $GeneratorInstance"
}
if (($Generator -eq "Auto" -or $Generator -eq "NMake Makefiles") -and -not (Test-Path $VsDevCmd)) {
    throw "VsDevCmd.bat not found: $VsDevCmd"
}

Write-Host "[build_sibr_viewer] Source: $sibrRoot"
Write-Host "[build_sibr_viewer] Build:  $buildPath"
Write-Host "[build_sibr_viewer] Config: $Config"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Reset-TransientOutputState {
    $script:SuppressedIncludeCount = 0
    $script:SuppressedEncodingWarningCount = 0
    $script:SuppressedUpToDateCount = 0
    $script:SuppressedRedistWarningCount = 0
    $script:LastProgressKey = $null
}

function Flush-ProgressLine {
    if ($script:LastProgressKey) {
        Write-Host ""
        $script:LastProgressKey = $null
    }
}

function Format-CommandLine {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList = @()
    )

    $parts = @($FilePath) + $ArgumentList
    return ($parts | ForEach-Object {
        if ($_ -match '\s') { '"{0}"' -f $_ } else { $_ }
    }) -join " "
}

function Write-CommandSummary {
    param(
        [string]$LogPath,
        [int]$ExitCode
    )

    Flush-ProgressLine
    if ($script:SuppressedIncludeCount -gt 0) {
        Write-Host ("[build_sibr_viewer] Suppressed include-trace lines: {0}" -f $script:SuppressedIncludeCount) -ForegroundColor DarkGray
    }
    if ($script:SuppressedEncodingWarningCount -gt 0) {
        Write-Host ("[build_sibr_viewer] Suppressed repeated encoding warnings: {0}" -f $script:SuppressedEncodingWarningCount) -ForegroundColor DarkGray
    }
    if ($script:SuppressedUpToDateCount -gt 0) {
        Write-Host ("[build_sibr_viewer] Suppressed install 'Up-to-date' lines: {0}" -f $script:SuppressedUpToDateCount) -ForegroundColor DarkGray
    }
    if ($script:SuppressedRedistWarningCount -gt 0) {
        Write-Host ("[build_sibr_viewer] Suppressed repeated MSVC redist warnings: {0}" -f $script:SuppressedRedistWarningCount) -ForegroundColor DarkGray
    }
    Write-Host ("[build_sibr_viewer] Raw log: {0}" -f $LogPath) -ForegroundColor DarkGray
    if ($ExitCode -ne 0) {
        Write-Host ("[build_sibr_viewer] Command failed with exit code {0}" -f $ExitCode) -ForegroundColor Red
    }
}

function Write-PrettyLine {
    param([string]$Line)

    if ($null -eq $Line) {
        return
    }

    if ($Line -match '^\-\- \[download (\d+)% complete\]$') {
        $percent = $Matches[1]
        $progressKey = "download"
        $script:LastProgressKey = $progressKey
        Write-Host ("`r[build_sibr_viewer] Downloading... {0}%" -f $percent) -NoNewline -ForegroundColor DarkCyan
        return
    }

    if ($Line -match '包含文件|including file|敞鎰|鍖呭惈鏂囦欢|^\?[^\r\n]*:\s+[A-Za-z]:\\') {
        $script:SuppressedIncludeCount += 1
        return
    }

    if ($Line -match 'warning C4819') {
        $script:SuppressedEncodingWarningCount += 1
        return
    }

    if ($Line -match '^-- Up-to-date: ') {
        $script:SuppressedUpToDateCount += 1
        return
    }

    if ($Line -match 'MSVC_REDIST_DIR-NOTFOUND|Microsoft\.VC143\.CRT') {
        $script:SuppressedRedistWarningCount += 1
        return
    }

    Flush-ProgressLine

    if ($Line -match '^CMake Error') {
        Write-Host $Line -ForegroundColor Red
    } elseif ($Line -match '^CMake Warning|^WARNING:') {
        Write-Host $Line -ForegroundColor Yellow
    } elseif ($Line -match '^\[build_sibr_viewer\]') {
        Write-Host $Line -ForegroundColor Gray
    } elseif ($Line -match '^==> ') {
        Write-Host $Line -ForegroundColor Cyan
    } else {
        Write-Host $Line
    }
}

function Invoke-PrettyCommand {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList = @(),
        [string]$WorkingDirectory = $repoRoot,
        [string]$LogPath,
        [string]$DisplayName = ""
    )

    if ([string]::IsNullOrWhiteSpace($DisplayName)) {
        $DisplayName = Split-Path $FilePath -Leaf
    }

    New-Item -ItemType Directory -Force -Path (Split-Path $LogPath -Parent) | Out-Null
    Reset-TransientOutputState
    Write-Host ("[build_sibr_viewer] Running: {0}" -f $DisplayName) -ForegroundColor Gray

    $commandLine = Format-CommandLine -FilePath $FilePath -ArgumentList $ArgumentList
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "cmd.exe"
    $psi.Arguments = '/d /c "{0} 2>&1"' -f $commandLine
    $psi.WorkingDirectory = $WorkingDirectory
    $psi.UseShellExecute = $false
    $psi.StandardOutputEncoding = $script:BuildEncoding
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $false
    $psi.CreateNoWindow = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi

    $writer = [System.IO.StreamWriter]::new($LogPath, $false, [System.Text.UTF8Encoding]::new($false))
    try {
        $null = $process.Start()
        while (-not $process.StandardOutput.EndOfStream) {
            $line = $process.StandardOutput.ReadLine()
            $writer.WriteLine($line)
            Write-PrettyLine -Line $line
        }
        $process.WaitForExit()
        $writer.Flush()
        Write-CommandSummary -LogPath $LogPath -ExitCode $process.ExitCode
        return $process.ExitCode
    } finally {
        $writer.Dispose()
        $process.Dispose()
    }
}

function Invoke-PrettyCmdLine {
    param(
        [string]$CommandLine,
        [string]$WorkingDirectory = $repoRoot,
        [string]$LogPath,
        [string]$DisplayName = "command"
    )

    New-Item -ItemType Directory -Force -Path (Split-Path $LogPath -Parent) | Out-Null
    Reset-TransientOutputState
    Write-Host ("[build_sibr_viewer] Running: {0}" -f $DisplayName) -ForegroundColor Gray

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "cmd.exe"
    $psi.Arguments = '/d /c "{0} 2>&1"' -f $CommandLine
    $psi.WorkingDirectory = $WorkingDirectory
    $psi.UseShellExecute = $false
    $psi.StandardOutputEncoding = $script:BuildEncoding
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $false
    $psi.CreateNoWindow = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi

    $writer = [System.IO.StreamWriter]::new($LogPath, $false, [System.Text.UTF8Encoding]::new($false))
    try {
        $null = $process.Start()
        while (-not $process.StandardOutput.EndOfStream) {
            $line = $process.StandardOutput.ReadLine()
            $writer.WriteLine($line)
            Write-PrettyLine -Line $line
        }
        $process.WaitForExit()
        $writer.Flush()
        Write-CommandSummary -LogPath $LogPath -ExitCode $process.ExitCode
        return $process.ExitCode
    } finally {
        $writer.Dispose()
        $process.Dispose()
    }
}

function Find-SevenZip {
    param([string]$PreferredPath = "")

    $sevenZipCandidates = @(
        $PreferredPath,
        (Get-Command 7z -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
        "C:\Program Files\7-Zip\7z.exe",
        "C:\Program Files (x86)\7-Zip\7z.exe"
    ) | Where-Object { $_ }

    return ($sevenZipCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1)
}

function Install-SevenZip {
    Write-Host "[build_sibr_viewer] 7-Zip not found; attempting automatic install..."

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        & $winget.Source install --id 7zip.7zip -e --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -eq 0) {
            return
        }
        Write-Warning "[build_sibr_viewer] winget install failed with exit code $LASTEXITCODE."
    }

    $choco = Get-Command choco -ErrorAction SilentlyContinue
    if ($choco) {
        & $choco.Source install 7zip -y
        if ($LASTEXITCODE -eq 0) {
            return
        }
        Write-Warning "[build_sibr_viewer] choco install failed with exit code $LASTEXITCODE."
    }

    throw "Unable to install 7-Zip automatically. Install it manually or pass -SevenZipCmd 'C:\path\to\7z.exe'."
}

function Remove-IncompleteWin3rdParty {
    param([string]$ExtlibsRoot)

    if (-not (Test-Path $ExtlibsRoot)) {
        return
    }

    Get-ChildItem -Path $ExtlibsRoot -Directory | ForEach-Object {
        $items = @(Get-ChildItem -Path $_.FullName -Force)
        if ($items.Count -eq 1 -and $items[0].Name -eq "Win3rdPartyUrl") {
            Write-Warning "[build_sibr_viewer] Removing incomplete dependency cache: $($_.FullName)"
            Remove-Item -Recurse -Force $_.FullName
        }
    }
}

function Ensure-VsCudaShim {
    param(
        [string]$VsDevCmdPath,
        [string]$ActualMsvcDir,
        [string]$ShimRoot
    )

    $buildDir = Split-Path $VsDevCmdPath -Parent
    $vsCommon7 = Split-Path $buildDir -Parent
    $vsRoot = Split-Path $vsCommon7 -Parent
    $vcAuxBuildDir = Join-Path $vsRoot "VC\Auxiliary\Build"
    $vcvarsall = Join-Path $vcAuxBuildDir "vcvarsall.bat"
    $actualVcToolsRoot = Split-Path $ActualMsvcDir -Parent
    $actualMsvcVersion = Split-Path $ActualMsvcDir -Leaf

    $shimMsvcDir = Join-Path $ShimRoot "VC\Tools\MSVC\$actualMsvcVersion"
    $shimCl = Join-Path $shimMsvcDir "bin\HostX64\x64\cl.exe"
    $shimVcvars64 = Join-Path $ShimRoot "VC\Auxiliary\Build\vcvars64.bat"
    $shimVcvarsall = Join-Path $ShimRoot "VC\Auxiliary\Build\vcvarsall.bat"

    if (-not (Test-Path $shimCl)) {
        New-Item -ItemType Directory -Force -Path (Split-Path $shimMsvcDir -Parent) | Out-Null
        cmd /c "mklink /J `"$shimMsvcDir`" `"$ActualMsvcDir`"" | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create MSVC junction for CUDA shim: $shimMsvcDir"
        }
    }

    New-Item -ItemType Directory -Force -Path (Split-Path $shimVcvars64 -Parent) | Out-Null
    $shim = "@echo off`r`ncall `"$vcvarsall`" x64 %*`r`n"
    Set-Content -Path $shimVcvars64 -Value $shim -Encoding ASCII
    Set-Content -Path $shimVcvarsall -Value $shim -Encoding ASCII

    return [string]$shimCl
}

$SevenZipCmd = Find-SevenZip -PreferredPath $SevenZipCmd
if (([string]::IsNullOrWhiteSpace($SevenZipCmd) -or -not (Test-Path $SevenZipCmd)) -and -not $NoAutoInstall7Zip) {
    Install-SevenZip
    $SevenZipCmd = Find-SevenZip
}
if ([string]::IsNullOrWhiteSpace($SevenZipCmd) -or -not (Test-Path $SevenZipCmd)) {
    throw "7-Zip not found. Install 7-Zip, let the script auto-install it, or pass -SevenZipCmd 'C:\path\to\7z.exe'."
}

Write-Host "[build_sibr_viewer] 7-Zip:  $SevenZipCmd"

$extlibsRoot = Join-Path $sibrRoot "extlibs"
Remove-IncompleteWin3rdParty -ExtlibsRoot $extlibsRoot

$vsCudaShimRoot = Join-Path $buildPath "_vs_cuda_shim"
$shimCl = Ensure-VsCudaShim -VsDevCmdPath $VsDevCmd -ActualMsvcDir $MsvcDir -ShimRoot $vsCudaShimRoot
$shimClCmake = $shimCl -replace "\\", "/"
$logDir = Join-Path $buildPath "logs"
$logStamp = Get-Date -Format "yyyyMMdd_HHmmss"

$cudaFlags = "--allow-unsupported-compiler"

$includePath = @(
    (Join-Path $MsvcDir "include"),
    (Join-Path $WindowsSdkDir "Include\$WindowsSdkVersion\ucrt"),
    (Join-Path $WindowsSdkDir "Include\$WindowsSdkVersion\shared"),
    (Join-Path $WindowsSdkDir "Include\$WindowsSdkVersion\um"),
    (Join-Path $WindowsSdkDir "Include\$WindowsSdkVersion\winrt"),
    (Join-Path $WindowsSdkDir "Include\$WindowsSdkVersion\cppwinrt")
) -join ";"

$libPath = @(
    (Join-Path $MsvcDir "lib\x64"),
    (Join-Path $WindowsSdkDir "Lib\$WindowsSdkVersion\ucrt\x64"),
    (Join-Path $WindowsSdkDir "Lib\$WindowsSdkVersion\um\x64")
) -join ";"

$toolPath = @(
    (Join-Path $MsvcDir "bin\HostX64\x64"),
    (Join-Path $WindowsSdkDir "bin\$WindowsSdkVersion\x64"),
    $env:PATH
) -join ";"
if ($Generator -eq "Auto") {
    $Generator = "Visual Studio 17 2022"
}

Write-Host "[build_sibr_viewer] Generator: $Generator"

if ($Generator -like "Visual Studio*") {
    Write-Step "Configure with Visual Studio generator"
    $configureArgs = @(
        "--fresh",
        "-S", $sibrRoot,
        "-B", $buildPath,
        "-G", $Generator,
        "-A", $Platform,
        "-DCMAKE_GENERATOR_INSTANCE=$GeneratorInstance",
        "-DSEVEN_ZIP_CMD:FILEPATH=$SevenZipCmd",
        "-DCMAKE_CUDA_FLAGS=$cudaFlags",
        "-DCMAKE_C_COMPILER=$shimClCmake",
        "-DCMAKE_CXX_COMPILER=$shimClCmake",
        "-DCMAKE_CUDA_HOST_COMPILER=$shimClCmake"
    )
    $vsConfigureExit = Invoke-PrettyCommand -FilePath "cmake" -ArgumentList $configureArgs -WorkingDirectory $repoRoot -LogPath (Join-Path $logDir "configure-visualstudio-$logStamp.log") -DisplayName "cmake configure (Visual Studio)"
    if ($vsConfigureExit -ne 0) {
        Write-Warning "[build_sibr_viewer] Visual Studio generator failed; falling back to NMake Makefiles."
        $Generator = "NMake Makefiles"
    }
}

if ($Generator -eq "NMake Makefiles") {
    Write-Step "Configure and build with NMake"
    $configureCmd = 'cmake --fresh -S "{0}" -B "{1}" -G "{2}" -DCMAKE_BUILD_TYPE={3} -DCMAKE_TRY_COMPILE_CONFIGURATION=Release -DSEVEN_ZIP_CMD:FILEPATH="{4}" -DCMAKE_CUDA_FLAGS="{5}" -DCMAKE_C_COMPILER="{6}" -DCMAKE_CXX_COMPILER="{6}" -DCMAKE_CUDA_HOST_COMPILER="{6}"' -f $sibrRoot, $buildPath, $Generator, $Config, $SevenZipCmd, $cudaFlags, $shimClCmake
    $buildCmd = 'cmake --build "{0}" --target install --config {1}' -f $buildPath, $Config
    $cmdScript = 'call "{0}" -arch={1} && set "INCLUDE={2}" && set "LIB={3}" && set "PATH={4}" && {5} && {6}' -f $VsDevCmd, $Platform, $includePath, $libPath, $toolPath, $configureCmd, $buildCmd

    $nmakeExit = Invoke-PrettyCmdLine -CommandLine $cmdScript -WorkingDirectory $repoRoot -LogPath (Join-Path $logDir "build-nmake-$logStamp.log") -DisplayName "cmake configure+build (NMake)"
    if ($nmakeExit -ne 0) {
        throw "SIBR configure/build failed with exit code $nmakeExit."
    }
} elseif ($Generator -like "Visual Studio*") {
    Write-Step "Build and install"
    $buildArgs = @(
        "--build", $buildPath,
        "--target", "install",
        "--config", $Config
    )
    $vsBuildExit = Invoke-PrettyCommand -FilePath "cmake" -ArgumentList $buildArgs -WorkingDirectory $repoRoot -LogPath (Join-Path $logDir "build-visualstudio-$logStamp.log") -DisplayName "cmake build (Visual Studio)"
    if ($vsBuildExit -ne 0) {
        throw "CMake build/install failed with exit code $vsBuildExit."
    }
}

$viewerExe = Join-Path $sibrRoot "install\bin\SIBR_gaussianViewer_app.exe"
$viewerConfigExe = Join-Path $sibrRoot "install\bin\SIBR_gaussianViewer_app_config.exe"
$viewerRwdiExe = Join-Path $sibrRoot "install\bin\SIBR_gaussianViewer_app_rwdi.exe"
$solutionPath = Join-Path $buildPath "sibr_projects.sln"

if (Test-Path $viewerExe) {
    Write-Host "[build_sibr_viewer] Built viewer: $viewerExe"
} elseif (Test-Path $viewerRwdiExe) {
    Write-Host "[build_sibr_viewer] Built viewer: $viewerRwdiExe"
} elseif (Test-Path $viewerConfigExe) {
    Write-Host "[build_sibr_viewer] Built viewer: $viewerConfigExe"
} else {
    Write-Warning "[build_sibr_viewer] Build finished, but SIBR_gaussianViewer_app(.exe) was not found under install/bin."
}

if ($OpenSolution -and (Test-Path $solutionPath)) {
    Start-Process $solutionPath | Out-Null
}
