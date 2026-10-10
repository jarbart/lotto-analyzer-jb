@echo off
cd /d "%~dp0"

if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Python virtual environment not found.
    echo Create the environment and install requirements first.
    pause
    exit /b 1
)

"%~dp0.venv\Scripts\python.exe" "%~dp0app.py"

if errorlevel 1 (
    echo.
    echo Error while starting Lotto Analyzer JB.
    pause
)
