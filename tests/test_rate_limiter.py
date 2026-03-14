#!/usr/bin/env python3
"""
Тесты для Rate Limiter с прогрессивной блокировкой
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.rate_limiter import RateLimiter, limiter


def test_rate_limiter_singleton():
    """Тест singleton паттерна"""
    print("Тест singleton...")
    l1 = RateLimiter()
    l2 = RateLimiter()
    assert l1 is l2, "Singleton не работает"
    print("[PASS] Singleton")


def test_rate_limiter_basic():
    """Базовый тест rate limiting"""
    print("Тест базового rate limiting...")
    limiter.reset("test_basic")

    key = "test_basic"
    max_requests = 5
    window = 10

    # Разрешить 5 запросов
    for i in range(max_requests):
        assert limiter.is_allowed(key, max_requests, window), f"Запрос {i+1} должен быть разрешён"

    # 6-й должен быть заблокирован
    assert not limiter.is_allowed(key, max_requests, window), "6-й запрос должен быть заблокирован"

    print("[PASS] Базовый rate limiting")


def test_rate_limiter_window():
    """Тест скользящего окна"""
    print("Тест скользящего окна...")
    limiter.reset("test_window")

    key = "test_window"
    max_requests = 3
    window = 2  # 2 секунды

    # Отключаем прогрессивную блокировку для этого теста
    limiter.progressive_blocking = False

    # Исчерпать лимит
    for _ in range(max_requests):
        limiter.is_allowed(key, max_requests, window)

    # Должен быть заблокирован
    assert not limiter.is_allowed(key, max_requests, window)

    # Ждём истечения окна
    time.sleep(window + 0.5)

    # Должен снова разрешить
    assert limiter.is_allowed(key, max_requests, window), "После истечения окна должен разрешить"

    # Включаем обратно
    limiter.progressive_blocking = True

    print("[PASS] Скользящее окно")


def test_progressive_blocking():
    """Тест прогрессивной блокировки"""
    print("Тест прогрессивной блокировки...")
    limiter.reset("test_progressive")

    key = "test_progressive"
    max_requests = 2
    window = 60

    limiter.progressive_blocking = True

    # Первое нарушение - блокировка на 2 минуты
    for _ in range(max_requests):
        limiter.is_allowed(key, max_requests, window)

    assert not limiter.is_allowed(key, max_requests, window)

    status = limiter.get_status(key, max_requests, window)
    assert status["violation_count"] == 1, "Должно быть 1 нарушение"
    assert status["blocked"] is True, "Должен быть заблокирован"

    # Блокировка должна быть > 100 секунд (2 минуты)
    assert status["retry_after"] > 100, f"Блокировка должна быть > 100 сек, получено {status['retry_after']}"

    print("[PASS] Прогрессивная блокировка")


def test_get_status():
    """Тест получения статуса"""
    print("Тест получения статуса...")
    limiter.reset("test_status")

    key = "test_status"
    max_requests = 10
    window = 60

    # Начальное состояние
    status = limiter.get_status(key, max_requests, window)
    assert status["requests_made"] == 0
    assert status["requests_remaining"] == 10
    assert status["blocked"] is False

    # Сделать 5 запросов
    for _ in range(5):
        limiter.is_allowed(key, max_requests, window)

    status = limiter.get_status(key, max_requests, window)
    assert status["requests_made"] == 5
    assert status["requests_remaining"] == 5

    print("[PASS] Получение статуса")


def test_get_request_count():
    """Тест подсчёта запросов"""
    print("Тест подсчёта запросов...")
    limiter.reset("test_count")

    key = "test_count"
    window = 60

    # Сделать 7 запросов
    for _ in range(7):
        limiter.is_allowed(key, 10, window)

    count = limiter.get_request_count(key, window)
    assert count == 7, f"Должно быть 7 запросов, получено {count}"

    print("[PASS] Подсчёт запросов")


def test_cleanup_old_entries():
    """Тест очистки старых записей"""
    print("Тест очистки старых записей...")
    limiter.reset("test_cleanup")

    key = "test_cleanup"
    window = 1  # 1 секунда

    # Сделать запросы
    for _ in range(3):
        limiter.is_allowed(key, 10, window)

    # Ждём истечения
    time.sleep(1.5)

    # Очистка
    limiter.cleanup_old_entries(max_age_seconds=1)

    # Запись должна быть удалена
    count = limiter.get_request_count(key, window)
    assert count == 0, f"После очистки должно быть 0, получено {count}"

    print("[PASS] Очистка старых записей")


def test_reset():
    """Тест сброса"""
    print("Тест сброса...")
    limiter.reset("test_reset")

    key = "test_reset"
    window = 60

    # Сделать запросы
    for _ in range(5):
        limiter.is_allowed(key, 10, window)

    # Сброс
    limiter.reset(key)

    # Должно быть 0
    count = limiter.get_request_count(key, window)
    assert count == 0, f"После сброса должно быть 0, получено {count}"

    print("[PASS] Сброс")


def main():
    """Запуск всех тестов"""
    print("=" * 60)
    print("ТЕСТЫ RATE LIMITER")
    print("=" * 60)

    tests = [
        test_rate_limiter_singleton,
        test_rate_limiter_basic,
        test_rate_limiter_window,
        test_progressive_blocking,
        test_get_status,
        test_get_request_count,
        test_cleanup_old_entries,
        test_reset,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"ИТОГИ: {passed}/{len(tests)} тестов пройдено ({passed/len(tests)*100:.1f}%)")

    if failed == 0:
        print("Все тесты пройдены!")
        return 0
    else:
        print(f"{failed} тест(а) провалено")
        return 1


if __name__ == "__main__":
    sys.exit(main())
