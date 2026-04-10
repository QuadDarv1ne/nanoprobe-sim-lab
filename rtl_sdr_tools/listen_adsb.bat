@echo off
REM ADS-B Aircraft Tracker - Quick Launch
REM Uses dump1090-win + adsb_tracker.py

echo ========================================
echo ADS-B Aircraft Tracker (1090 MHz)
echo ========================================
echo.

cd /d "%~dp0.."

echo [1/3] Checking dump1090...
if exist "tools\Dump1090-main\dump1090.exe" (
    echo     Found: tools\Dump1090-main\dump1090.exe
) else (
    echo     ERROR: dump1090.exe not found!
    echo     Run: setup_adsb.bat
    pause
    exit /b 1
)

echo.
echo [2/3] Starting dump1090 + aircraft tracker...
echo     Press Ctrl+C to stop
echo.

python rtl_sdr_tools\adsb_tracker.py --mode decode --duration 30

echo.
echo [3/3] Done!
echo.
echo To view recent sightings:
echo     python rtl_sdr_tools\adsb_tracker.py --mode list
echo.
echo To view stats:
echo     python rtl_sdr_tools\adsb_tracker.py --mode stats
echo.

pause
