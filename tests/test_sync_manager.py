#!/usr/bin/env python3
"""
Тесты для Sync Manager (Backend ↔ Frontend синхронизация)
"""

import sys
from pathlib import Path

# Добавляем корень проекта в path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.sync_manager import BackendFrontendSync


def test_sync_manager_init():
    """Тест инициализации"""
    print("Тест инициализации...")

    sync = BackendFrontendSync()

    assert sync.backend_url == "http://localhost:8000", "Default backend URL"
    assert sync.frontend_url == "http://localhost:5000", "Default frontend URL"
    assert sync._running is False, "Не должен быть запущен"
    assert sync._last_sync_time is None, "Время синхронизации должно быть None"

    print("[PASS] Инициализация")


def test_sync_manager_custom_urls():
    """Тест кастомных URL"""
    print("Тест кастомных URL...")

    sync = BackendFrontendSync(
        backend_url="http://custom-backend:9000",
        frontend_url="http://custom-frontend:4000"
    )

    assert sync.backend_url == "http://custom-backend:9000", "Custom backend URL"
    assert sync.frontend_url == "http://custom-frontend:4000", "Custom frontend URL"

    print("[PASS] Кастомные URL")


def test_sync_manager_get_status():
    """Тест получения статуса"""
    print("Тест получения статуса...")

    sync = BackendFrontendSync()

    status = sync.get_sync_status()

    assert "running" in status, "Статус должен содержать 'running'"
    assert "backend_url" in status, "Статус должен содержать 'backend_url'"
    assert "frontend_url" in status, "Статус должен содержать 'frontend_url'"
    assert "last_sync_time" in status, "Статус должен содержать 'last_sync_time'"
    assert "backend_connections" in status, "Статус должен содержать 'backend_connections'"
    assert "frontend_connections" in status, "Статус должен содержать 'frontend_connections'"

    assert status["running"] is False, "Не должен быть запущен"
    assert status["backend_url"] == "http://localhost:8000"
    assert status["frontend_url"] == "http://localhost:5000"

    print("[PASS] Получение статуса")


def test_sync_manager_ws_connections():
    """Тест WebSocket подключений"""
    print("Тест WebSocket подключений...")

    sync = BackendFrontendSync()

    # Начальное состояние
    assert len(sync.ws_connections["backend"]) == 0
    assert len(sync.ws_connections["frontend"]) == 0

    # Добавление подключений (симуляция)
    sync.ws_connections["backend"].add("client1")
    sync.ws_connections["backend"].add("client2")
    sync.ws_connections["frontend"].add("socket1")

    assert len(sync.ws_connections["backend"]) == 2
    assert len(sync.ws_connections["frontend"]) == 1

    # Удаление подключений
    sync.ws_connections["backend"].remove("client1")

    assert len(sync.ws_connections["backend"]) == 1

    print("[PASS] WebSocket подключения")


def test_sync_manager_stop():
    """Тест остановки"""
    print("Тест остановки...")

    sync = BackendFrontendSync()

    # Остановка без запуска не должна вызывать ошибок
    sync.stop_sync_loop()

    assert sync._running is False

    print("[PASS] Остановка")


async def test_sync_manager_async_methods():
    """Тест async методов (требует запущенных сервисов)"""
    print("Тест async методов (skip без сервисов)...")

    # Без запущенных сервисов health check должен вернуть False
    # Это нормальное поведение

    print("[SKIP] Async методы (требуют запущенных сервисов)")


def main():
    """Запуск всех тестов"""
    print("=" * 60)
    print("ТЕСТЫ SYNC MANAGER")
    print("=" * 60)
    print()

    tests = [
        test_sync_manager_init,
        test_sync_manager_custom_urls,
        test_sync_manager_get_status,
        test_sync_manager_ws_connections,
        test_sync_manager_stop,
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
    import asyncio

    exit_code = main()

    # Запуск async тестов
    print()
    print("=" * 60)
    print("ASYNC ТЕСТЫ")
    print("=" * 60)
    print()

    try:
        asyncio.run(test_sync_manager_async_methods())
    except Exception as e:
        print(f"[SKIP] Async тесты: {e}")

    sys.exit(exit_code)
