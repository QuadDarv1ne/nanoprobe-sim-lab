@echo off
REM Nanoprobe Sim Lab - Universal Launcher for Windows
REM Uses Python 3.13 with all dependencies installed

set PYTHON_PATH=C:\Users\maksi\AppData\Local\Programs\Python\Python313\python.exe

echo ======================================================================
echo   Nanoprobe Sim Lab - Universal Launcher v2.0
echo ======================================================================
echo.
echo Доступные режимы:
echo   1. flask   - Flask + FastAPI (http://localhost:5000)
echo   2. nextjs  - Next.js + FastAPI (http://localhost:3000)
echo   3. api     - Backend API only (http://localhost:8000/docs)
echo   4. full    - Flask + FastAPI + Sync Manager
echo.
echo Или запустите без параметров для интерактивного выбора.
echo.

if "%1"=="" (
    %PYTHON_PATH% start_universal.py
) else (
    %PYTHON_PATH% start_universal.py %1
)

pause
