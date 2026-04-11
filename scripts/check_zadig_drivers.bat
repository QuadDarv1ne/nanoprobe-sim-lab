@echo off
echo ================================================================
echo   RTL-SDR V4 - Проверка и установка драйверов Zadig
echo ================================================================
echo.
echo [Шаг 1] Проверка видимости устройства...
echo.

python -c "from rtlsdr import RtlSdr; print('Устройств найдено:', RtlSdr.get_device_count())" 2>nul

if %errorlevel% neq 0 (
    echo.
    echo [ВНИМАНИЕ] Устройство не найдено или ошибка драйверов!
    echo.
    echo [Решение]
    echo 1. Скачайте Zadig: https://zadig.akeo.ie/
    echo 2. Запустите Zadig от имени администратора
    echo 3. Options -^> List All Devices
    echo 4. Выберите RTL2838UHIDIR или Bulk-In, Interface
    echo 5. Драйвер: WinUSB (v6.x.x.x)
    echo 6. Нажмите Replace Driver
    echo 7. Переподключите RTL-SDR
    echo.
    pause
    exit /b 1
)

echo.
echo [Шаг 2] Проверка открытия устройства...
echo.

python -c "from rtlsdr import RtlSdr; sdr = RtlSdr(); print('✅ Устройство открыто!'); sdr.close()" 2>nul

if %errorlevel% neq 0 (
    echo.
    echo [ОШИБКА] Не удалось открыть устройство!
    echo.
    echo [Причина] Драйверы Zadig не установлены или неправильный драйвер
    echo.
    echo [Решение]
    echo 1. Запустите Zadig от имени администратора
    echo 2. Options -^> List All Devices
    echo 3. Отключите Ignore Hubs or Composite Parent Devices
    echo 4. Выберите RTL2838UHIDIR
    echo 5. Убедитесь что выбран WinUSB
    echo 6. Нажмите Replace Driver
    echo 7. Переподключите RTL-SDR
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================================
echo   ✅ ОТЛИЧНО! RTL-SDR V4 полностью готов к работе!
echo ================================================================
echo.
echo [Следующий шаг]
echo Запустите: python test_rtlsdr_full_power.py
echo.
pause
