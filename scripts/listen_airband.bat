@echo off
chcp 65001 >nul
REM ============================================
✈️ Прослушивание авиасвязи через RTL-SDR V4
Частота: %1 MHz (по умолчанию 118.5)
AM модуляция
============================================

set PATH=C:\rtl-sdr\bin\x64;%PATH%

if "%1"=="" (
    set FREQ=118.5
) else (
    set FREQ=%1
)

echo ✈️ Приём авиасвязи на частоте %FREQ% MHz
echo 💡 Нажми Ctrl+C для остановки
echo.
echo 🎧 Вывод звука через ffplay...
echo.

rtl_fm -f %FREQ%M -M am -s 12k -r 48k -g 40 -l 0 | ffplay -ar 48000 -f s16le -autoexit -nodisp -volume 100
