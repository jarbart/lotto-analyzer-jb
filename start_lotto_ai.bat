@echo off
cd /d "%~dp0"

"%~dp0.venv\Scripts\python.exe" "%~dp0app.py"

if errorlevel 1 (
    echo.
    echo Wystapil blad podczas uruchamiania LottoAI.
    pause
)
