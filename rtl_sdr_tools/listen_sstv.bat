@echo off
REM SSTV Ground Station - Quick Launch
REM Захват сигнала с МКС и декодирование через MMSSTV

echo ========================================
echo SSTV Ground Station (145.800 MHz)
echo ========================================
echo.

cd /d "%~dp0.."

echo [1/2] Checking RTL-SDR...
where rtl_fm.exe >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo     Checking alternative paths...
    if exist "C:\rtl-sdr\bin\x64\rtl_fm.exe" (
        echo     Found: C:\rtl-sdr\bin\x64\rtl_fm.exe
    ) else (
        echo     ERROR: rtl_fm.exe not found!
        echo     Install RTL-SDR drivers and tools
        pause
        exit /b 1
    )
) else (
    echo     Found: rtl_fm.exe
)

echo.
echo [2/2] Starting SSTV capture...
echo.

python rtl_sdr_tools\sstv_mmsstv_capture.py %*

echo.
echo ========================================
echo Done!
echo ========================================
echo.
echo To view recent recordings:
echo     python rtl_sdr_tools\sstv_mmsstv_capture.py --list
echo.

pause
