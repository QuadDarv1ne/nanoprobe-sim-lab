@echo off
chcp 65001 >nul
REM ============================================
FM Прослушивание через RTL-SDR V4
Частота: %1 MHz (по умолчанию 101.7)
============================================

set PATH=C:\rtl-sdr\bin\x64;%PATH%

if "%1"=="" (
    set FREQ=101.7
) else (
    set FREQ=%1
)

echo 📻 Приём FM радио на частоте %FREQ% MHz
echo 💡 Нажми Ctrl+C для остановки
echo.
echo 🎧 Вывод звука через ffplay...
echo.

rtl_fm -f %FREQ%M -M wbfm -s 256k -r 48k -g 49.6 -F 9 | ffplay -ar 48000 -f s16le -autoexit -nodisp -volume 100
