@echo off
echo =====================================================
echo  Nanoprobe Sim Lab - RTL-SDR V4 Setup
echo =====================================================
echo.
echo This script downloads required RTL-SDR drivers and tools.
echo.
echo 1. Zadig (WinUSB driver installer)
echo 2. RTL-SDR Blog Windows binaries
echo.
pause

echo.
echo [1/2] Downloading Zadig...
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri 'https://github.com/pbatard/libwdi/releases/download/v1.5.0/zadig-2.9.exe' -OutFile 'zadig.exe' -UseBasicParsing; echo 'Zadig downloaded successfully.' } catch { echo 'Failed: ' + $_.Exception.Message }"

echo.
echo [2/2] Downloading RTL-SDR Blog Windows tools...
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri 'https://github.com/QuadDarv1ne/nanoprobe-sim-lab/releases/download/v1.0.0/rtl-sdr-blog-windows.zip' -OutFile 'rtl-sdr-blog.zip' -UseBasicParsing; echo 'RTL-SDR tools downloaded.' } catch { echo 'Failed: ' + $_.Exception.Message }"

if exist "rtl-sdr-blog.zip" (
    echo Extracting RTL-SDR tools...
    powershell -Command "Expand-Archive -Path 'rtl-sdr-blog.zip' -DestinationPath 'rtl-sdr-bin' -Force"
    echo Done!
)

echo.
echo Setup complete. Run fix_rtlsdr_driver.bat to install drivers.
pause
