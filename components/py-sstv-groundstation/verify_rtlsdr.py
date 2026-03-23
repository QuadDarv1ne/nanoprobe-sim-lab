#!/usr/bin/env python3
"""
RTL-SDR V4 Verification Script
Быстрая проверка RTL-SDR устройства перед первым использованием
"""

import sys
import subprocess


def check_rtlsdr_library():
    """Проверка установки библиотеки rtlsdr."""
    print("=" * 60)
    print("1. Проверка библиотеки rtlsdr")
    print("-" * 60)

    try:
        import rtlsdr
        print("✓ rtlsdr установлен")
        print(f"  Версия: {rtlsdr.__version__ if hasattr(rtlsdr, '__version__') else 'unknown'}")
        return True
    except ImportError:
        print("✗ rtlsdr не установлен")
        print("\nУстановите:")
        print("  pip install pyrtlsdr")
        print("\nТакже установите librtlsdr драйверы:")
        print("  Windows: Zadig (см. RTL_SDR_SETUP.md)")
        print("  Linux: sudo apt-get install librtlsdr-dev")
        print("  macOS: brew install librtlsdr")
        return False


def check_devices():
    """Проверка подключенных RTL-SDR устройств."""
    print("\n" + "=" * 60)
    print("2. Поиск RTL-SDR устройств")
    print("-" * 60)

    try:
        from rtlsdr import RtlSdr

        num_devices = RtlSdr.get_device_count()
        print(f"Найдено устройств: {num_devices}")

        if num_devices == 0:
            print("\n⚠ RTL-SDR устройства не обнаружены")
            print("\nПроверьте:")
            print("  1. Устройство подключено к USB порту")
            print("  2. Драйверы установлены (Zadig для Windows)")
            print("  3. USB кабель исправен")
            return False

        return num_devices
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        return False


def identify_devices(num_devices):
    """Идентификация типа устройств."""
    print("\n" + "=" * 60)
    print("3. Идентификация устройств")
    print("-" * 60)

    from rtlsdr import RtlSdr

    v4_found = False

    for i in range(num_devices):
        try:
            sdr = RtlSdr(device_index=i)

            # Получаем информацию
            device_name = sdr.get_device_name() if hasattr(sdr, 'get_device_name') else 'Unknown'
            serial = sdr.get_serial_number() if hasattr(sdr, 'get_serial_number') else 'Unknown'
            manufacturer = sdr.get_manufacturer() if hasattr(sdr, 'get_manufacturer') else 'Unknown'

            print(f"\nУстройство #{i}:")
            print(f"  Название: {device_name}")
            print(f"  Серийный номер: {serial}")
            print(f"  Производитель: {manufacturer}")

            # Определяем RTL-SDR V4
            device_upper = device_name.upper()
            if 'R828D' in device_upper or 'V4' in device_upper:
                print(f"  ✓✓✓ RTL-SDR V4 ОБНАРУЖЕН ✓✓✓")
                v4_found = True
            elif 'R820T' in device_upper:
                print(f"  → RTL-SDR V3 (R820T/R820T2)")
            else:
                print(f"  → RTL-SDR (классический)")

            sdr.close()

        except Exception as e:
            print(f"\nУстройство #{i}: Ошибка - {e}")

    return v4_found


def test_basic_functionality(device_index=0):
    """Базовый тест функциональности."""
    print("\n" + "=" * 60)
    print("4. Базовый тест функциональности")
    print("-" * 60)

    try:
        from rtlsdr import RtlSdr
        import numpy as np

        sdr = RtlSdr(device_index=device_index)

        # Настройка
        sdr.sample_rate = 2.4e6  # 2.4 MSPS для V4
        sdr.center_freq = 145.8e6  # ISS частота
        sdr.gain = 30

        print(f"✓ Устройство инициализировано")
        print(f"  Sample Rate: {sdr.sample_rate / 1e6:.1f} MSPS")
        print(f"  Center Freq: {sdr.center_freq / 1e6:.3f} MHz")
        print(f"  Gain: {sdr.gain} dB")

        # Читаем сэмплы
        print("\n  Чтение тестовых сэмплов...")
        samples = sdr.read_samples(1024)

        if samples is not None and len(samples) > 0:
            power = np.mean(np.abs(samples) ** 2)
            print(f"✓ Сэмплы получены: {len(samples)}")
            print(f"  Средняя мощность: {10 * np.log10(power + 1e-10):.1f} dB")
        else:
            print("✗ Не удалось получить сэмплы")
            sdr.close()
            return False

        sdr.close()
        print("\n✓ Базовый тест пройден успешно")
        return True

    except Exception as e:
        print(f"✗ Ошибка теста: {e}")
        return False


def print_next_steps(v4_found):
    """Выводит следующие шаги."""
    print("\n" + "=" * 60)
    print("СЛЕДУЮЩИЕ ШАГИ")
    print("=" * 60)

    if v4_found:
        print("\n✓ RTL-SDR V4 готов к работе!")
        print("\nРекомендуемые команды для тестирования:\n")

        print("1. Waterfall дисплей (спектрограмма):")
        print("   cd src")
        print("   python main.py --waterfall -f iss --duration 60\n")

        print("2. Real-time SSTV декодирование:")
        print("   python main.py --realtime-sstv -f iss --duration 120\n")

        print("3. Запись и декодирование:")
        print("   python main.py --sdr -f 145.800 --duration 30 --auto-decode\n")

        print("4. Расписание пролётов МКС:")
        print("   python main.py --schedule --lat 55.75 --lon 37.61\n")

        print("5. Автоматическая запись при пролёте:")
        print("   python main.py --auto-record --schedule-hours 24\n")

        print("Документация:")
        print("  - RTL_SDR_SETUP.md - настройка устройства")
        print("  - docs/03-rtl-sdr-sstv-recording.md - руководство по записи")
    else:
        print("\n⚠ RTL-SDR V4 не обнаружен")
        print("\nЕсли устройство подключено:")
        print("  1. Проверьте драйверы (Zadig для Windows)")
        print("  2. Переподключите устройство")
        print("  3. Попробуйте другой USB порт")
        print("  4. Запустите скрипт снова")


def main():
    """Основная функция проверки."""
    print("\n" + "=" * 60)
    print("RTL-SDR V4 VERIFICATION SCRIPT")
    print("Проверка готовности RTL-SDR устройства")
    print("=" * 60)

    # Шаг 1: Проверка библиотеки
    if not check_rtlsdr_library():
        sys.exit(1)

    # Шаг 2: Поиск устройств
    num_devices = check_devices()
    if not num_devices:
        print_next_steps(False)
        sys.exit(1)

    # Шаг 3: Идентификация
    v4_found = identify_devices(num_devices)

    # Шаг 4: Базовый тест
    if num_devices > 0:
        test_basic_functionality(device_index=0)

    # Следующие шаги
    print_next_steps(v4_found)

    print("\n" + "=" * 60)
    print("✓ Проверка завершена")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
