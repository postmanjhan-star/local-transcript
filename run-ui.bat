@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
if not exist ".venv\Scripts\pythonw.exe" (
    echo Please run: powershell -ExecutionPolicy Bypass -File setup.ps1
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "%~dp0gui.py" %*

