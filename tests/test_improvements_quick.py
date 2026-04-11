#!/usr/bin/env python3
"""
Быстрые тесты для новых улучшений Nanoprobe Sim Lab
Запуск без pytest
"""

import os
import sys
import time
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_schemas_validation():
    """Тесты валидации Pydantic схем"""
    print("\n=== Тесты валидации схем ===")
    from pydantic import ValidationError

    from api.schemas import LoginRequest, PaginationParams

    # Тест 1: Валидный запрос
    try:
        _ = LoginRequest(username="test_user", password="Password123")
        print("[PASS] Валидный логин")
    except Exception as e:
        print(f"[FAIL] Валидный логин: {e}")
        return False

    # Тест 2: Короткий пароль
    try:
        LoginRequest(username="test_user", password="short1")
        print("[FAIL] Короткий пароль (должен был отклонить)")
        return False
    except ValidationError:
        print("[PASS] Короткий пароль")

    # Тест 3: Без заглавных
    try:
        LoginRequest(username="test_user", password="password123")
        print("[FAIL] Без заглавных (должен был отклонить)")
        return False
    except ValidationError:
        print("[PASS] Без заглавных")

    # Тест 4: Без цифр
    try:
        LoginRequest(username="test_user", password="Passwordabc")
        print("[FAIL] Без цифр (должен был отклонить)")
        return False
    except ValidationError:
        print("[PASS] Без цифр")

    # Тест 5: Пагинация
    try:
        params = PaginationParams(page=2, page_size=50)
        assert params.offset == 50
        assert params.limit == 50
        print("[PASS] Пагинация")
    except Exception as e:
        print(f"[FAIL] Пагинация: {e}")
        return False

    return True


def test_database_cache():
    """Тесты кэширования DatabaseManager"""
    print("\n=== Тесты кэша БД ===")
    import tempfile

    from utils.database import DatabaseManager

    # Создаём временную БД
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = DatabaseManager(db_path=db_path, enable_cache=True)

        # Тест 1: Инициализация кэша
        assert hasattr(db, "_query_cache")
        assert db._cache_ttl == 60
        print("[PASS] Инициализация кэша")

        # Тест 2: Set/Get кэша
        db._set_cache("test_key", {"data": "value"})
        result = db._get_from_cache("test_key")
        assert result == {"data": "value"}
        print("[PASS] Set/Get кэша")

        # Тест 3: TTL кэша
        db._cache_ttl = 1
        db._set_cache("ttl_key", "value")
        time.sleep(1.5)
        result = db._get_from_cache("ttl_key")
        assert result is None
        print("[PASS] TTL кэша")

        # Тест 4: Инвалидация
        db._set_cache("scans:all:100:0", "val1")
        db._set_cache("scans:spm:50:0", "val2")
        db._set_cache("other:key", "val3")
        db.invalidate_cache("scans:")
        assert "scans:all:100:0" not in db._query_cache
        assert "other:key" in db._query_cache
        print("[PASS] Инвалидация кэша")

        # Тест 5: Статистика
        db._set_cache("key1", "value1")
        db._set_cache("key2", "value2")
        stats = db.get_cache_stats()
        assert stats["total_entries"] >= 2
        print("[PASS] Статистика кэша")

        # Тест 6: add_scan_result инвалидация
        scan_id = db.add_scan_result("spm", "test", 100, 100)
        scan_keys_before = [k for k in db._query_cache.keys() if "scans:" in k]
        assert len(scan_keys_before) == 0  # Кэш очищен после вставки
        print("[PASS] Инвалидация при вставке")

        # Тест 7: delete_scan инвалидация
        db.get_scan_by_id(scan_id)
        db.get_scan_results()
        db.delete_scan(scan_id)
        # Кэш должен быть очищен для scan:id:{id}
        scan_id_keys = [k for k in db._query_cache.keys() if f"scan:id:{scan_id}" in k]
        assert len(scan_id_keys) == 0
        print("[PASS] Инвалидация при удалении")

        return True

    except Exception as e:
        print(f"[FAIL] Тесты кэша: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Закрываем все пулы перед удалением файла
        try:
            DatabaseManager.close_all_pools()
            time.sleep(0.5)
            os.unlink(db_path)
        except Exception:
            pass


def test_enhanced_monitor_prometheus():
    """Тесты экспорта Prometheus метрик"""
    print("\n=== Тесты Prometheus метрик ===")
    from utils.enhanced_monitor import EnhancedSystemMonitor

    monitor = EnhancedSystemMonitor(history_size=10)
    monitor.start_monitoring()
    time.sleep(2)

    try:
        # Тест 1: Формат экспорта
        metrics = monitor.export_prometheus_metrics()
        assert isinstance(metrics, str)
        assert "# HELP" in metrics
        assert "# TYPE" in metrics
        print("[PASS] Формат Prometheus")

        # Тест 2: Наличие метрик
        assert "nanoprobe_cpu_percent" in metrics
        assert "nanoprobe_memory_percent" in metrics
        assert "nanoprobe_disk_percent" in metrics
        print("[PASS] Наличие метрик")

        # Тест 3: Типы метрик
        assert "TYPE nanoprobe_cpu_percent gauge" in metrics
        assert "TYPE nanoprobe_uptime_seconds counter" in metrics
        print("[PASS] Типы метрик")

        # Тест 4: Значения метрик
        lines = metrics.split("\n")
        # Ищем строку сразу после комментария о CPU
        for i, line in enumerate(lines):
            if "nanoprobe_cpu_percent " in line and not line.startswith("#"):
                cpu_value = float(line.split()[-1])
                assert 0 <= cpu_value <= 100
                print(f"[PASS] Значение CPU ({cpu_value:.1f}%)")
                break
        else:
            print("[FAIL] Не найдено значение CPU")
            return False

        return True

    except Exception as e:
        print(f"[FAIL] Тесты Prometheus: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        monitor.stop_monitoring()


def test_dashboard_routes():
    """Тесты dashboard routes"""
    print("\n=== Тесты Dashboard routes ===")
    from api.routes.dashboard import get_storage_stats
    from utils.enhanced_monitor import format_uptime

    try:
        # Тест 1: Статистика хранилища
        stats = get_storage_stats()
        assert "used_mb" in stats
        assert "total_mb" in stats
        assert "percent" in stats
        print("[PASS] Статистика хранилища")

        # Тест 2: Форматирование аптайма
        assert format_uptime(0) == "< 1 мин"
        assert format_uptime(60) == "1 мин"
        assert format_uptime(3661) == "1 ч 1 мин"
        assert "дн" in format_uptime(86400)
        print("[PASS] Форматирование аптайма")

        return True

    except Exception as e:
        print(f"[FAIL] Dashboard routes: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Запуск всех тестов"""
    print("=" * 60)
    print("ТЕСТЫ НОВЫХ УЛУЧШЕНИЙ NANOPROBE SIM LAB")
    print("=" * 60)

    results = {
        "Схемы валидации": test_schemas_validation(),
        "Кэш БД": test_database_cache(),
        "Prometheus метрики": test_enhanced_monitor_prometheus(),
        "Dashboard routes": test_dashboard_routes(),
    }

    print("\n" + "=" * 60)
    print("ИТОГИ")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} - {name}")

    print(f"\nВсего: {passed}/{total} тестов пройдено ({passed/total*100:.1f}%)")

    if passed == total:
        print("\nВсе тесты пройдены!")
        return 0
    else:
        print(f"\n{total - passed} тест(а) провалено")
        return 1


if __name__ == "__main__":
    sys.exit(main())
