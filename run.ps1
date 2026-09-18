$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $env:PYTHONUTF8 = '1'
    $pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonExe)) {
        Write-Host 'Run: powershell -ExecutionPolicy Bypass -File setup.ps1'
        exit 1
    }
    & $pythonExe (Join-Path $PSScriptRoot 'transcribe.py') @args
    $runExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $runExitCode

