@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run this command first:
    echo powershell -ExecutionPolicy Bypass -File setup.ps1
    pause
    exit /b 1
)
".venv\Scripts\python.exe" "transcribe.py" %*
set "result=%errorlevel%"
echo.
pause
exit /b %result%

