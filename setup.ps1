[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Executable $($Arguments -join ' ')" }
}

# Keep all managed files in this project, and restore the caller's environment.
$setupEnvironment = @{
    PYTHONUTF8 = '1'
    UV_CACHE_DIR = (Join-Path $PSScriptRoot '.cache\uv')
    UV_PYTHON_INSTALL_DIR = (Join-Path $PSScriptRoot '.python')
    UV_PROJECT_ENVIRONMENT = (Join-Path $PSScriptRoot '.venv')
    UV_INSTALL_DIR = (Join-Path $PSScriptRoot '.tools\uv')
    UV_NO_MODIFY_PATH = '1'
}
$previousEnvironment = @{}
foreach ($name in $setupEnvironment.Keys) {
    $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    [Environment]::SetEnvironmentVariable($name, $setupEnvironment[$name], 'Process')
}

Push-Location $PSScriptRoot
try {
    Write-Host '[1/4] Locating uv...'
    $uvCommand = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
    $uvExe = if ($uvCommand) { $uvCommand.Source } else { $null }
    if (-not $uvExe) {
        foreach ($candidate in @(
            (Join-Path $env:UV_INSTALL_DIR 'uv.exe'),
            (Join-Path $env:USERPROFILE '.local\bin\uv.exe')
        )) {
            if (Test-Path -LiteralPath $candidate) { $uvExe = $candidate; break }
        }
    }
    if (-not $uvExe) {
        Write-Host 'Installing uv from astral.sh into .tools\uv (no administrator rights needed)...'
        New-Item -ItemType Directory -Force -Path $env:UV_CACHE_DIR | Out-Null
        $installer = Join-Path $env:UV_CACHE_DIR 'install-uv.ps1'
        Invoke-WebRequest -UseBasicParsing -Uri 'https://astral.sh/uv/install.ps1' -OutFile $installer
        Invoke-Checked 'powershell.exe' @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $installer)
        $uvExe = Join-Path $env:UV_INSTALL_DIR 'uv.exe'
        if (-not (Test-Path -LiteralPath $uvExe)) { throw 'uv installation did not produce uv.exe.' }
    }
    Invoke-Checked $uvExe @('--version')

    Write-Host '[2/4] Preparing managed Python 3.12...'
    Invoke-Checked $uvExe @('python', 'install', '--no-bin', '3.12')

    Write-Host '[3/4] Synchronizing .venv from uv.lock (including test tools)...'
    Invoke-Checked $uvExe @('sync', '--locked', '--managed-python', '--python', '3.12', '--group', 'dev')

    Write-Host '[4/4] Checking FFmpeg and local transcription dependencies...'
    foreach ($directory in @('input', 'output', 'models', 'logs')) {
        New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot $directory) | Out-Null
    }
    $venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    Invoke-Checked $venvPython @('transcribe.py', '--self-test')
    Write-Host ''
    Write-Host 'Setup completed. Double-click run-ui.bat to drag recordings into the desktop app.'
    Write-Host 'For command-line batch processing, put recordings in input and use run.bat.'
    Write-Host 'The first transcription downloads its model. Later runs use the local cache.'
} catch {
    Write-Host "[ERROR] $($_.Exception.Message)"
    Write-Host 'Setup can be run again after interruption. Check the failing step above.'
    exit 1
} finally {
    Pop-Location
    foreach ($name in $previousEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], 'Process')
    }
}
