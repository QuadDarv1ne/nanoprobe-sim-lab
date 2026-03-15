#!/usr/bin/env python3
"""
Тесты для Comprehensive Rate Limiting
"""

import sys
from pathlib import Path

# Добавляем корень проекта в path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.rate_limiter import (
    limiter,
    auth_limit,
    api_limit,
    write_limit,
    download_limit,
    external_limit,
    sstv_limit,
    get_retry_after,
    format_retry_after,
    whitelist_ip,
    blacklist_ip,
    is_ip_whitelisted,
    is_ip_blacklisted,
    get_ip_lists,
    inc_rate_limit_hit,
    inc_rate_limit_blocked,
    get_rate_limit_stats,
    reset_rate_limit_stats,
)


def test_rate_limiter_init():
    """Тест инициализации rate limiter"""
    print("Тест инициализации rate limiter...")
    
    assert limiter is not None, "Limiter должен быть инициализирован"
    
    print("[PASS] Инициализация rate limiter")


def test_decorators():
    """Тест декораторов rate limiting"""
    print("Тест декораторов...")
    
    # Auth limit (5/60s)
    auth_dec = auth_limit()
    assert auth_dec is not None, "auth_limit должен вернуть декоратор"
    
    # API limit (100/60s)
    api_dec = api_limit()
    assert api_dec is not None, "api_limit должен вернуть декоратор"
    
    # Write limit (30/60s)
    write_dec = write_limit()
    assert write_dec is not None, "write_limit должен вернуть декоратор"
    
    # Download limit (20/60s)
    download_dec = download_limit()
    assert download_dec is not None, "download_limit должен вернуть декоратор"
    
    # External limit (10/60s)
    external_dec = external_limit()
    assert external_dec is not None, "external_limit должен вернуть декоратор"
    
    # SSTV limit (10/60s)
    sstv_dec = sstv_limit()
    assert sstv_dec is not None, "sstv_limit должен вернуть декоратор"
    
    print("[PASS] Декораторы")


def test_custom_decorators():
    """Тест кастомных декораторов"""
    print("Тест кастомных декораторов...")
    
    # Custom auth limit
    auth_dec = auth_limit(max_requests=3, window=30)
    assert auth_dec is not None
    
    # Custom API limit
    api_dec = api_limit(max_requests=200, window=120)
    assert api_dec is not None
    
    # Custom write limit
    write_dec = write_limit(max_requests=50, window=90)
    assert write_dec is not None
    
    print("[PASS] Кастомные декораторы")


def test_format_retry_after():
    """Тест форматирования retry after"""
    print("Тест форматирования retry after...")
    
    # Секунды
    assert format_retry_after(30) == "30 сек"
    assert format_retry_after(59) == "59 сек"
    
    # Минуты
    assert format_retry_after(60) == "1 мин"
    assert format_retry_after(120) == "2 мин"
    assert format_retry_after(300) == "5 мин"
    
    # Часы
    assert format_retry_after(3600) == "1 час"
    assert format_retry_after(7200) == "2 час"
    
    print("[PASS] Форматирование retry after")


def test_ip_whitelist():
    """Тест whitelist IP"""
    print("Тест whitelist IP...")
    
    # Очистка
    from api.rate_limiter import _ip_whitelist
    _ip_whitelist.clear()
    
    # Добавление
    whitelist_ip("192.168.1.1")
    whitelist_ip("10.0.0.1")
    
    assert is_ip_whitelisted("192.168.1.1"), "IP должен быть в whitelist"
    assert is_ip_whitelisted("10.0.0.1"), "IP должен быть в whitelist"
    assert not is_ip_whitelisted("192.168.1.2"), "IP не должен быть в whitelist"
    
    # Удаление
    from api.rate_limiter import remove_from_whitelist
    remove_from_whitelist("192.168.1.1")
    assert not is_ip_whitelisted("192.168.1.1"), "IP должен быть удалён из whitelist"
    
    print("[PASS] Whitelist IP")


def test_ip_blacklist():
    """Тест blacklist IP"""
    print("Тест blacklist IP...")
    
    # Очистка
    from api.rate_limiter import _ip_blacklist
    _ip_blacklist.clear()
    
    # Добавление
    blacklist_ip("192.168.1.100")
    blacklist_ip("10.0.0.100")
    
    assert is_ip_blacklisted("192.168.1.100"), "IP должен быть в blacklist"
    assert is_ip_blacklisted("10.0.0.100"), "IP должен быть в blacklist"
    assert not is_ip_blacklisted("192.168.1.101"), "IP не должен быть в blacklist"
    
    # Удаление
    from api.rate_limiter import remove_from_blacklist
    remove_from_blacklist("192.168.1.100")
    assert not is_ip_blacklisted("192.168.1.100"), "IP должен быть удалён из blacklist"
    
    print("[PASS] Blacklist IP")


def test_get_ip_lists():
    """Тест получения списков IP"""
    print("Тест получения списков IP...")
    
    # Очистка
    from api.rate_limiter import _ip_whitelist, _ip_blacklist
    _ip_whitelist.clear()
    _ip_blacklist.clear()
    
    # Добавление
    whitelist_ip("192.168.1.1")
    blacklist_ip("192.168.1.100")
    
    lists = get_ip_lists()
    
    assert "whitelist" in lists, "Должен быть ключ 'whitelist'"
    assert "blacklist" in lists, "Должен быть ключ 'blacklist'"
    assert "192.168.1.1" in lists["whitelist"], "IP должен быть в whitelist"
    assert "192.168.1.100" in lists["blacklist"], "IP должен быть в blacklist"
    
    print("[PASS] Получение списков IP")


def test_rate_limit_stats():
    """Тест статистики rate limit"""
    print("Тест статистики rate limit...")
    
    # Сброс
    reset_rate_limit_stats()
    
    # Инкремент
    inc_rate_limit_hit("/api/v1/auth/login")
    inc_rate_limit_hit("/api/v1/auth/login")
    inc_rate_limit_blocked("/api/v1/auth/login")
    
    stats = get_rate_limit_stats()
    
    assert "/api/v1/auth/login" in stats, "Endpoint должен быть в статистике"
    assert stats["/api/v1/auth/login"]["hits"] == 2, "Должно быть 2 hits"
    assert stats["/api/v1/auth/login"]["blocked"] == 1, "Должна быть 1 блокировка"
    
    # Сброс
    reset_rate_limit_stats()
    stats = get_rate_limit_stats()
    assert len(stats) == 0, "Статистика должна быть сброшена"
    
    print("[PASS] Статистика rate limit")


def test_rate_limit_scopes():
    """Тест что декораторы работают"""
    print("Тест что декораторы работают...")
    
    # Проверка что декораторы возвращаются
    auth_dec = auth_limit()
    api_dec = api_limit()
    write_dec = write_limit()
    
    # Все декораторы должны быть не None
    assert auth_dec is not None, "auth_limit должен вернуть декоратор"
    assert api_dec is not None, "api_limit должен вернуть декоратор"
    assert write_dec is not None, "write_limit должен вернуть декоратор"
    
    print("[PASS] Декораторы работают")


def main():
    """Запуск всех тестов"""
    print("=" * 70)
    print("  Comprehensive Rate Limiting Tests")
    print("=" * 70)
    print()
    
    tests = [
        test_rate_limiter_init,
        test_decorators,
        test_custom_decorators,
        test_format_retry_after,
        test_ip_whitelist,
        test_ip_blacklist,
        test_get_ip_lists,
        test_rate_limit_stats,
        test_rate_limit_scopes,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"   ❌ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"   ⚠️  ERROR: {e}")
            failed += 1
        print()
    
    print("=" * 70)
    print(f"  Результаты: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n✅ Все тесты пройдены!")
    else:
        print(f"\n❌ {failed} тестов не пройдено")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
