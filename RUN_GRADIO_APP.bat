@echo off
setlocal
title Meta AI - Dashboard
cd /d "%~dp0"

echo Meta AI - starting dashboard (quick_start.py)
echo URL will be printed below (often http://127.0.0.1:7860 or similar).
echo Press Ctrl+C to stop.
echo.

if exist ".venv312\Scripts\python.exe" (
  set "PY=.venv312\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

"%PY%" quick_start.py

pause
endlocal
