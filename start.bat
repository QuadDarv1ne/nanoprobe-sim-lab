@echo off
REM Nanoprobe Sim Lab Launcher for Windows
REM Uses Python 3.13 with all dependencies installed

set PYTHON_PATH=C:\Users\maksi\AppData\Local\Programs\Python\Python313\python.exe

echo ======================================================================
echo   Nanoprobe Sim Lab - Windows Launcher
echo ======================================================================
echo.
echo Available modes:
echo   1. flask   - Flask Dashboard (http://localhost:5000)
echo   2. nextjs  - Next.js Dashboard (http://localhost:3000)
echo   3. api     - Backend API only (http://localhost:8000/docs)
echo.

if "%1"=="" (
    %PYTHON_PATH% start.py
) else (
    %PYTHON_PATH% start.py %1
)

pause
