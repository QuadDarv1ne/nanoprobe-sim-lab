@echo off
chcp 65001 >nul
REM ============================================
📡 Сканирование авиационного диапазона (118-137 MHz)
Поиск активных частот авиасвязи
============================================

set PATH=C:\rtl-sdr\bin\x64;%PATH%

echo ✈️ Сканирование авиационного диапазона 118-137 MHz
echo 📊 Шаг: 0.05 MHz (50 kHz)
echo ⏱️  Длительность: ~10 сек на частоту
echo.
echo 🔍 Обнаруженные сигналы:
echo ==========================================

REM Сканирование с шагом 50 kHz
rtl_power -f 118M:137M:50k -i 10 -g 40 -e 3m aviation_scan.csv

if exist aviation_scan.csv (
    echo.
    echo ==========================================
    ✅ Сканирование завершено!
    💾 Результаты: aviation_scan.csv

    echo.
    📊 Топ-10 самых сильных сигналов:
    echo.
    python -c "import csv; data=list(csv.reader(open('aviation_scan.csv'))); data=[(float(r[2]),float(r[3])) for r in data if len(r)>=4]; data.sort(key=lambda x: x[1], reverse=True); [print(f'  {d[0]/1e6:.3f} MHz - {d[1]:.2f} dB') for d in data[:10]]"
) else (
    echo ❌ Ошибка сканирования
)
