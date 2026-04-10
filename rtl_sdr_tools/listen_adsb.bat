@echo off
REM ADS-B Aircraft Tracker (1090 MHz)
REM Отслеживание самолётов через RTL-SDR
echo ========================================
echo ADS-B Aircraft Tracker - 1090 MHz
echo ========================================
echo.

cd /d "%~dp0"
python adsb_receiver.py --freq 1090 --gain 30 --map
pause
