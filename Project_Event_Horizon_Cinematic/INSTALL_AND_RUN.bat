@echo off
setlocal
cd /d "%~dp0"

echo ================================================
echo   PROJECT EVENT HORIZON - CINEMATIC EDITION
echo ================================================

where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher 'py' was not found. Install Python 3.10+ and enable PATH.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating project virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 goto :error
)

call ".venv\Scripts\activate.bat"
echo Installing / checking Python requirements...
python -m pip install --upgrade pip
if errorlevel 1 goto :error
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Launching Event Horizon Cinematic Edition...
python event_horizon_cinematic.py
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo Setup or launch failed. Read README.md for troubleshooting.
pause
exit /b 1
