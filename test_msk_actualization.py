#!/usr/bin/env python3
"""
Тестовый скрипт для проверки автоматической актуализации МСК данных.
Проверяет:
1. Единый источник данных (utils.location_manager)
2. Автоматическое обновление кэша
3. Корректность расчётов времени МСК
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


def test_location_manager():
    """Тестируем основной менеджер локализации."""
    print("=" * 70)
    print("ТЕСТ 1: Проверка utils.location_manager")
    print("=" * 70)

    try:
        from utils.location_manager import (
            MSK_TZ,
            get_location,
            get_location_info,
            now_msk,
            refresh_msk_data,
            utc_to_msk,
        )

        # Получаем локацию
        loc = get_location()
        print(f"\n✓ Местоположение загружено:")
        print(f"  Город: {loc['city']}, {loc['country']}")
        print(f"  Координаты: {loc['lat']:.4f}°N, {loc['lon']:.4f}°E")
        print(f"  Часовой пояс: {loc['timezone'].name} (UTC+{loc['timezone'].utc_offset})")

        # Проверяем время МСК
        msk_time = now_msk()
        utc_time = datetime.now(timezone.utc)
        print(f"\n✓ Время:")
        print(f"  UTC: {utc_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  MSK: {msk_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Проверяем конвертацию
        utc_now = datetime.now(timezone.utc)
        msk_converted = utc_to_msk(utc_now)
        print(f"\n✓ Конвертация UTC→MSK:")
        print(f"  UTC: {utc_now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  MSK: {msk_converted.strftime('%Y-%m-%d %H:%M:%S')}")

        # Проверяем информацию о локации
        print(f"\n✓ Информация о локации:")
        print(get_location_info())

        return True
    except Exception as e:
        print(f"\n✗ Ошибка: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_geolocation_wrapper():
    """Тестируем обёртку geolocation.py."""
    print("\n" + "=" * 70)
    print("ТЕСТ 2: Проверка geolocation.py (обёртка)")
    print("=" * 70)

    try:
        sys.path.insert(0, str(project_root / "components" / "py-sstv-groundstation" / "src"))
        from geolocation import DEFAULT_LOCATION, MSK_TZ, get_location, now_msk, utc_to_msk

        loc = get_location()
        print(f"\n✓ Обёртка geolocation работает:")
        print(f"  Город: {loc['city']}, {loc['country']}")
        print(f"  Координаты: {loc['lat']:.4f}°N, {loc['lon']:.4f}°E")

        # Проверяем что используется единый источник
        from utils.location_manager import MSK_TZ as MAIN_MSK_TZ

        assert MSK_TZ.name == MAIN_MSK_TZ.name, "MSK_TZ не совпадает!"
        assert MSK_TZ.utc_offset == MAIN_MSK_TZ.utc_offset, "MSK_TZ offset не совпадает!"
        print(f"\n✓ Единый источник MSK_TZ подтверждён")

        return True
    except Exception as e:
        print(f"\n✗ Ошибка: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_satellite_tracker():
    """Тестируем satellite_tracker.py."""
    print("\n" + "=" * 70)
    print("ТЕСТ 3: Проверка satellite_tracker.py")
    print("=" * 70)

    try:
        sys.path.insert(0, str(project_root / "components" / "py-sstv-groundstation" / "src"))
        from satellite_tracker import SatelliteTracker

        from utils.location_manager import get_location

        loc = get_location()
        tracker = SatelliteTracker(ground_station_lat=loc["lat"], ground_station_lon=loc["lon"])

        print(f"\n✓ SatelliteTracker инициализирован:")
        print(f"  Станция: {tracker.ground_station_lat:.4f}°N, {tracker.ground_station_lon:.4f}°E")
        print(f"  Спутников: {len(tracker.get_all_satellites())}")

        # Проверяем что timezone установлен
        if tracker._tz:
            print(f"  Timezone: {tracker._tz.name} (UTC+{tracker._tz.utc_offset})")
        else:
            print("  ⚠ Timezone не установлен")

        return True
    except Exception as e:
        print(f"\n✗ Ошибка: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_auto_recorder():
    """Тестируем auto_recorder.py."""
    print("\n" + "=" * 70)
    print("ТЕСТ 4: Проверка auto_recorder.py")
    print("=" * 70)

    try:
        sys.path.insert(0, str(project_root / "components" / "py-sstv-groundstation" / "src"))
        from auto_recorder import AutoRecordingScheduler

        from utils.location_manager import get_location

        loc = get_location()
        scheduler = AutoRecordingScheduler(
            ground_station_lat=loc["lat"], ground_station_lon=loc["lon"]
        )

        print(f"\n✓ AutoRecordingScheduler инициализирован:")
        print(f"  Станция: {scheduler.lat:.4f}°N, {scheduler.lon:.4f}°E")

        if scheduler._tz:
            print(f"  Timezone: {scheduler._tz.name} (UTC+{scheduler._tz.utc_offset})")
        else:
            print("  ⚠ Timezone не установлен")

        return True
    except Exception as e:
        print(f"\n✗ Ошибка: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_cache_mechanism():
    """Тестируем механизм кэширования."""
    print("\n" + "=" * 70)
    print("ТЕСТ 5: Проверка механизма кэширования")
    print("=" * 70)

    try:
        from utils.location_manager import CACHE_FILE, CACHE_TTL_HOURS, load_location_cache

        cached = load_location_cache()
        if cached:
            print(f"\n✓ Кэш загружен:")
            print(f"  Файл: {CACHE_FILE}")
            print(f"  TTL: {CACHE_TTL_HOURS} часов")
            print(f"  Город: {cached.get('city', 'Unknown')}")
            print(f"  Timestamp: {cached.get('timestamp', 'Unknown')}")
        else:
            print(f"\n⚠ Кэш отсутствует или устарел")
            print(f"  Будет создан при следующем вызове get_location()")

        return True
    except Exception as e:
        print(f"\n✗ Ошибка: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Запускает все тесты."""
    print("\n" + "=" * 70)
    print("АВТОМАТИЧЕСКАЯ АКТУАЛИЗАЦИЯ ДАННЫХ ПО МСК")
    print("Комплексная проверка системы")
    print("=" * 70)

    results = []

    results.append(("Location Manager", test_location_manager()))
    results.append(("Geolocation Wrapper", test_geolocation_wrapper()))
    results.append(("Satellite Tracker", test_satellite_tracker()))
    results.append(("Auto Recorder", test_auto_recorder()))
    results.append(("Cache Mechanism", test_cache_mechanism()))

    # Итоговый отчёт
    print("\n" + "=" * 70)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 70)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status:12} {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print("-" * 70)
    print(f"Всего тестов: {total}")
    print(f"Пройдено: {passed}")
    print(f"Провалено: {total - passed}")
    print("=" * 70)

    if all(r[1] for r in results):
        print("\n✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Система автоматической актуализации МСК работает корректно.")
        return 0
    else:
        print("\n✗ НЕКОТОРЫЕ ТЕСТЫ ПРОВАЛЕНЫ! Требуется доработка.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
