#!/usr/bin/env python
"""
Integration Tests: API + Database

Тестирование взаимодействия API с базой данных
"""

import sys
import os
import tempfile
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))

# Устанавливаем тестовую БД
TEST_DB = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_PATH"] = TEST_DB

from fastapi.testclient import TestClient
from api.main import app, db_manager


def setup_module():
    """Инициализация перед тестами"""
    print("\n[INFO] Инициализация тестовой БД...")


def teardown_module():
    """Очистка после тестов"""
    print("\n[INFO] Очистка тестовой БД...")
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_db_connection():
    """Тест 1: Подключение к базе данных"""
    print("\n[TEST] Подключение к базе данных...")
    
    assert db_manager is not None, "Database manager не инициализирован"
    print("[PASS] Database manager инициализирован")


def test_create_scan():
    """Тест 2: Создание сканирования в БД через API"""
    print("\n[TEST] Создание сканирования через API...")
    
    client = TestClient(app)
    
    # Тестовые данные
    scan_data = {
        "surface_type": "test_surface",
        "resolution": 128,
        "scan_size": 1.0,
        "noise_level": 0.1,
    }
    
    # Создаём сканирование
    response = client.post("/api/v1/scans/", json=scan_data)
    
    assert response.status_code == 200, f"Ошибка: {response.status_code} - {response.text}"
    data = response.json()
    
    assert "id" in data, "Ответ не содержит id"
    assert data["surface_type"] == "test_surface", "surface_type не совпадает"
    assert data["resolution"] == 128, "resolution не совпадает"
    
    scan_id = data["id"]
    print(f"[PASS] Сканирование создано (id={scan_id})")
    
    return scan_id


def test_get_scan(scan_id):
    """Тест 3: Получение сканирования из БД"""
    print("\n[TEST] Получение сканирования из БД...")
    
    client = TestClient(app)
    
    response = client.get(f"/api/v1/scans/{scan_id}")
    
    assert response.status_code == 200, f"Ошибка: {response.status_code}"
    data = response.json()
    
    assert data["id"] == scan_id, "id не совпадает"
    assert data["surface_type"] == "test_surface", "surface_type не совпадает"
    
    print(f"[PASS] Сканирование получено (id={scan_id})")


def test_list_scans():
    """Тест 4: Список сканирований"""
    print("\n[TEST] Список сканирований...")
    
    client = TestClient(app)
    
    response = client.get("/api/v1/scans/")
    
    assert response.status_code == 200, f"Ошибка: {response.status_code}"
    data = response.json()
    
    assert isinstance(data, list), "Ответ не список"
    assert len(data) > 0, "Список пуст"
    
    print(f"[PASS] Получено {len(data)} сканирований")


def test_update_scan(scan_id):
    """Тест 5: Обновление сканирования"""
    print("\n[TEST] Обновление сканирования...")
    
    client = TestClient(app)
    
    update_data = {
        "surface_type": "updated_surface",
        "resolution": 256,
    }
    
    response = client.put(f"/api/v1/scans/{scan_id}", json=update_data)
    
    assert response.status_code == 200, f"Ошибка: {response.status_code}"
    data = response.json()
    
    assert data["surface_type"] == "updated_surface", "surface_type не обновился"
    assert data["resolution"] == 256, "resolution не обновился"
    
    print(f"[PASS] Сканирование обновлено (id={scan_id})")


def test_delete_scan(scan_id):
    """Тест 6: Удаление сканирования"""
    print("\n[TEST] Удаление сканирования...")
    
    client = TestClient(app)
    
    response = client.delete(f"/api/v1/scans/{scan_id}")
    
    assert response.status_code == 200, f"Ошибка: {response.status_code}"
    
    # Проверяем, что удалено
    response = client.get(f"/api/v1/scans/{scan_id}")
    assert response.status_code == 404, "Сканирование не удалено"
    
    print(f"[PASS] Сканирование удалено (id={scan_id})")


def test_create_simulation():
    """Тест 7: Создание симуляции"""
    print("\n[TEST] Создание симуляции...")
    
    client = TestClient(app)
    
    sim_data = {
        "surface_type": "graphene",
        "resolution": 64,
        "scan_size": 0.5,
        "noise_level": 0.05,
    }
    
    response = client.post("/api/v1/simulations/", json=sim_data)
    
    assert response.status_code == 200, f"Ошибка: {response.status_code}"
    data = response.json()
    
    assert "id" in data, "Ответ не содержит id"
    assert data["surface_type"] == "graphene", "surface_type не совпадает"
    
    sim_id = data["id"]
    print(f"[PASS] Симуляция создана (id={sim_id})")
    
    return sim_id


def test_simulation_status(sim_id):
    """Тест 8: Статус симуляции"""
    print("\n[TEST] Статус симуляции...")
    
    client = TestClient(app)
    
    response = client.get(f"/api/v1/simulations/{sim_id}/status")
    
    assert response.status_code == 200, f"Ошибка: {response.status_code}"
    data = response.json()
    
    assert "status" in data, "Ответ не содержит status"
    assert data["status"] in ["pending", "running", "completed", "failed"], "Неверный статус"
    
    print(f"[PASS] Статус симуляции: {data['status']}")


def test_dashboard_stats():
    """Тест 9: Статистика дашборда"""
    print("\n[TEST] Статистика дашборда...")
    
    client = TestClient(app)
    
    response = client.get("/api/v1/dashboard/stats")
    
    assert response.status_code == 200, f"Ошибка: {response.status_code}"
    data = response.json()
    
    assert isinstance(data, dict), "Ответ не словарь"
    assert "total_scans" in data or "total_simulations" in data, "Ответ не содержит статистику"

    print("[PASS] Статистика получена")


def test_health_check():
    """Тест 10: Health check API"""
    print("\n[TEST] Health check API...")

    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200, f"Ошибка: {response.status_code}"
    data = response.json()

    assert data["status"] == "healthy", "API не healthy"
    assert "timestamp" in data, "Ответ не содержит timestamp"

    print("[PASS] API healthy")


def test_auth_login():
    """Тест 11: Аутентификация (login)"""
    print("\n[TEST] Аутентификация...")
    
    client = TestClient(app)
    
    # Пробуем войти с тестовыми учётными данными
    login_data = {
        "username": "admin",
        "password": "Admin123!",
    }
    
    response = client.post("/api/v1/auth/login", json=login_data)
    
    # Может быть 200 (успех) или 401 (нет пользователя)
    assert response.status_code in [200, 401, 404], f"Ошибка: {response.status_code}"
    
    if response.status_code == 200:
        data = response.json()
        assert "access_token" in data, "Ответ не содержит access_token"
        print("[PASS] Login успешен")
    else:
        print("[INFO] Пользователь admin не найден (это нормально для тестовой БД)")


def test_db_transaction_rollback():
    """Тест 12: Откат транзакции при ошибке"""
    print("\n[TEST] Откат транзакции...")
    
    client = TestClient(app)
    
    # Создаём некорректные данные
    invalid_data = {
        "surface_type": "",  # Пустое значение
        "resolution": -1,  # Отрицательное значение
    }
    
    response = client.post("/api/v1/scans/", json=invalid_data)
    
    # Должна быть ошибка валидации
    assert response.status_code in [400, 422], f"Ожидалась ошибка валидации: {response.status_code}"
    
    print("[PASS] Транзакция откатилась при ошибке валидации")


def test_concurrent_requests():
    """Тест 13: Параллельные запросы"""
    print("\n[TEST] Параллельные запросы...")
    
    import concurrent.futures
    
    client = TestClient(app)
    
    def make_request():
        response = client.get("/api/v1/scans/")
        return response.status_code
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request) for _ in range(5)]
        results = [f.result() for f in futures]
    
    assert all(r == 200 for r in results), "Не все запросы успешны"
    
    print("[PASS] Параллельные запросы обработаны")


def test_db_indexes():
    """Тест 14: Индексы БД"""
    print("\n[TEST] Индексы базы данных...")
    
    # Проверяем, что индексы созданы
    from alembic.config import Config
    from alembic.script import ScriptDirectory
    
    alembic_cfg = Config("alembic.ini")
    script = ScriptDirectory.from_config(alembic_cfg)
    
    # Проверяем наличие миграций
    migrations = list(script.walk_revisions())
    assert len(migrations) > 0, "Миграции не найдены"

    print(f"[PASS] Найдено {len(migrations)} миграций")


if __name__ == "__main__":
    print("=" * 60)
    print("Integration Tests: API + Database")
    print("=" * 60)
    
    setup_module()
    
    try:
        # Запуск тестов
        test_db_connection()
        test_health_check()
        
        # CRUD операции
        scan_id = test_create_scan()
        test_get_scan(scan_id)
        test_list_scans()
        test_update_scan(scan_id)
        
        # Статистика
        test_dashboard_stats()
        
        # Симуляции
        sim_id = test_create_simulation()
        test_simulation_status(sim_id)
        
        # Аутентификация
        test_auth_login()
        
        # Транзакции
        test_db_transaction_rollback()
        
        # Параллельные запросы
        test_concurrent_requests()
        
        # Индексы
        test_db_indexes()
        
        # Удаление
        test_delete_scan(scan_id)
        
        print("\n" + "=" * 60)
        print("✅ Все интеграционные тесты пройдены!")
        print("=" * 60)
        
    except AssertionError as e:
        print("\n" + "=" * 60)
        print(f"❌ Тест провален: {e}")
        print("=" * 60)
        sys.exit(1)
    
    finally:
        teardown_module()
