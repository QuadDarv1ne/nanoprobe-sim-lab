@echo off
REM Nanoprobe Sim Lab - Universal Launcher v3.0 (Unified)
REM Uses Python 3.13 with all dependencies installed

set PYTHON_PATH=C:\Users\maksi\AppData\Local\Programs\Python\Python313\python.exe

echo ======================================================================
echo   Nanoprobe Sim Lab - Universal Launcher v3.0 (Unified)
echo ======================================================================
echo.
echo Доступные режимы:
echo   1. flask   - Flask + FastAPI + Sync (http://localhost:5000)
echo   2. nextjs  - Next.js + FastAPI + Sync (http://localhost:3000)
echo   3. api     - Backend API only (http://localhost:8000/docs)
echo   4. full    - Full Stack (Flask + FastAPI + Sync Manager)
echo   5. dev     - Development mode (Flask + reload)
echo.
echo Или запустите без параметров для интерактивного выбора.
echo.

if "%1"=="" (
    %PYTHON_PATH% main.py
) else (
    %PYTHON_PATH% main.py %1
)

pause
