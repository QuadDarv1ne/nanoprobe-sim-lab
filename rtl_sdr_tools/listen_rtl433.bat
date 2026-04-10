@echo off
REM RTL_433 Scanner (433 MHz)
REM Сканирование метеостанций и датчиков
echo ========================================
echo RTL_433 Scanner - 433 MHz ISM Band
echo ========================================
echo.

cd /d "%~dp0"

REM Проверяем наличие rtl_433
if exist "..\tools\rtl_433\rtl_433.exe" (
    set RTL433=..\tools\rtl_433\rtl_433.exe
) else if exist "C:\rtl_433\rtl_433.exe" (
    set RTL433=C:\rtl_433\rtl_433.exe
) else (
    echo [ERROR] rtl_433.exe не найден!
    echo.
    echo Установка:
    echo   Windows: https://github.com/merbanan/rtl_433/releases
    echo   Linux:   sudo apt install rtl-433
    echo.
    pause
    exit /b 1
)

echo [*] Запуск сканирования 433.92 MHz...
echo [*] Длительность: 60 секунд
echo.

%RTL433% -f 433920000 -g 40 -C customary -F json
echo.
pause
