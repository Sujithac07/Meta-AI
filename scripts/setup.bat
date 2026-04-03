@echo off
setlocal
cd /d "%~dp0\.."

echo Meta AI - local setup
echo.

if not exist "data" mkdir data
if not exist "data\vectordb" mkdir data\vectordb
if not exist "data\sessions" mkdir data\sessions
if not exist "data\uploads" mkdir data\uploads
if not exist "exports" mkdir exports

where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found on PATH. Install Python 3.10+ and re-run this script.
  pause
  exit /b 1
)

echo Installing Python dependencies from requirements.txt ...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo pip install failed.
  pause
  exit /b 1
)

echo.
echo Setup complete.
echo Next: run "python quick_start.py" or double-click RUN_GRADIO_APP.bat in the repo root.
echo.
pause
endlocal
