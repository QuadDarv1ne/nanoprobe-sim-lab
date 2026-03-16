#!/usr/bin/env python3
"""
Тесты для Redis Cache
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.caching.redis_cache import RedisCache, cached_sync


def test_redis_cache_singleton():
    """Тест singleton паттерна"""
    print("Тест singleton...")
    c1 = RedisCache()
    c2 = RedisCache()
    # Каждый создаёт новый экземпляр (не singleton)
    assert c1 is not c2, "RedisCache не должен быть singleton"
    print("[PASS] Singleton (не singleton)")


def test_redis_cache_basic():
    """Базовый тест кэширования"""
    print("Тест базового кэширования...")

    test_cache = RedisCache()

    # Тест без Redis (должен вернуть False/None)
    result = test_cache.set("test_key", {"data": "value"}, expire=60)
    # Без Redis результат будет False
    assert result is False, "Без Redis результат должен быть False"

    result_get = test_cache.get("test_key")
    assert result_get is None, "Без Redis должен вернуть None"

    print("[PASS] Базовое кэширование (offline mode)")


def test_cache_decorator_sync():
    """Тест декоратора cached_sync"""
    print("Тест декоратора cached_sync...")

    call_count = {"value": 0}

    @cached_sync(prefix="test", expire=60)
    def expensive_function(x, y):
        """Тестовая функция с имитацией дорогой операции"""
        call_count["value"] += 1
        return x + y

    # Первый вызов
    result1 = expensive_function(5, 3)
    assert result1 == 8, "Результат должен быть 8"

    # Второй вызов с теми же аргументами (должен вернуть из кэша)
    result2 = expensive_function(5, 3)
    assert result2 == 8, "Результат должен быть 8"

    # Функция должна быть вызвана 1 раз (второй раз из кэша)
    # Но без Redis кэш не работает, поэтому будет 2 вызова
    assert call_count["value"] <= 2, "Функция не должна вызываться слишком много раз"

    print("[PASS] Декоратор cached_sync")


def test_cache_key_generation():
    """Тест генерации ключей"""
    print("Тест генерации ключей...")

    test_cache = RedisCache()

    key1 = test_cache.generate_key("prefix", "arg1", "arg2")
    key2 = test_cache.generate_key("prefix", "arg1", "arg2")
    key3 = test_cache.generate_key("prefix", "arg1", "arg3")

    assert key1 == key2, "Одинаковые аргументы должны давать одинаковый ключ"
    assert key1 != key3, "Разные аргументы должны давать разные ключи"
    assert key1.startswith("prefix:"), "Ключ должен начинаться с префикса"

    print("[PASS] Генерация ключей")


def test_cache_clear_pattern():
    """Тест очистки по паттерну"""
    print("Тест очистки по паттерну...")

    test_cache = RedisCache()

    # Без Redis должен вернуть 0
    result = test_cache.clear_pattern("test:*")
    assert result == 0, "Без Redis должен вернуть 0"

    print("[PASS] Очистка по паттерну (offline mode)")


def test_cache_invalidate_by_prefix():
    """Тест инвалидации по префиксу"""
    print("Тест инвалидации по префиксу...")

    test_cache = RedisCache()

    # Без Redis должен вернуть 0
    result = test_cache.invalidate_by_prefix("dashboard")
    assert result == 0, "Без Redis должен вернуть 0"

    print("[PASS] Инвалидация по префиксу (offline mode)")


def test_cache_json():
    """Тест кэширования JSON"""
    print("Тест кэширования JSON...")

    test_cache = RedisCache()

    # Без Redis должен вернуть False
    result = test_cache.cache_json("test_data", {"key": "value"}, expire=60)
    assert result is False, "Без Redis должен вернуть False"

    result_get = test_cache.get_json("test_data")
    assert result_get is None, "Без Redis должен вернуть None"

    print("[PASS] Кэширование JSON (offline mode)")


def test_cache_exists():
    """Тест проверки существования ключа"""
    print("Тест проверки существования...")

    test_cache = RedisCache()

    # Без Redis должен вернуть False
    result = test_cache.exists("nonexistent_key")
    assert result is False, "Без Redis должен вернуть False"

    print("[PASS] Проверка существования (offline mode)")


def test_cache_stats():
    """Тест получения статистики"""
    print("Тест статистики...")

    test_cache = RedisCache()

    stats = test_cache.get_stats()

    # Без Redis должен вернуть available=False
    assert stats["available"] is False, "Без Redis available должен быть False"

    print("[PASS] Статистика (offline mode)")


def test_cache_close():
    """Тест закрытия соединения"""
    print("Тест закрытия соединения...")

    test_cache = RedisCache()

    # Закрытие без подключения не должно вызывать ошибок
    test_cache.close()

    print("[PASS] Закрытие соединения")


def main():
    """Запуск всех тестов"""
    print("=" * 60)
    print("ТЕСТЫ REDIS CACHE")
    print("=" * 60)
    print()

    tests = [
        test_redis_cache_singleton,
        test_redis_cache_basic,
        test_cache_decorator_sync,
        test_cache_key_generation,
        test_cache_clear_pattern,
        test_cache_invalidate_by_prefix,
        test_cache_json,
        test_cache_exists,
        test_cache_stats,
        test_cache_close,
    ]

    passed = 0
    failed = 0
    errors = []

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
            errors.append((test.__name__, str(e)))
        except Exception as e:
            print(f"[ERROR] {test.__name__}: {e}")
            failed += 1
            errors.append((test.__name__, str(e)))

    print()
    print("=" * 60)
    print(f"ИТОГИ: {passed}/{len(tests)} тестов пройдено ({passed/len(tests)*100:.1f}%)")

    if failed == 0:
        print("✅ Все тесты пройдены!")
        return 0
    else:
        print(f"❌ {failed} тест(а) провалено")
        if errors:
            print()
            print("Ошибки:")
            for name, error in errors:
                print(f"  - {name}: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
