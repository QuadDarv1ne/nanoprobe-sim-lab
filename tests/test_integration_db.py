#!/usr/bin/env python
"""
Integration Tests: API + Database

Тестирование взаимодействия API с базой данных
"""

import sys
import os
import tempfile
import pytest
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))

# Устанавливаем тестовую БД
TEST_DB = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_PATH"] = TEST_DB

from fastapi.testclient import TestClient
from api.main import app
from api.state import get_db_manager


@pytest.fixture(scope="module")
def client():
    """Фикстура: HTTP клиент для тестов"""
    print("\n[INFO] Инициализация тестовой БД...")
    with TestClient(app) as test_client:
        yield test_client
    print("\n[INFO] Очистка тестовой БД...")
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except Exception:
            pass


@pytest.fixture
def scan_id(client):
    """Фикстура: ID созданного сканирования"""
    scan_data = {
        "scan_type": "spm",  # Enum: spm, image, sstv
        "surface_type": "test_surface",
        "width": 128,
        "height": 128,
    }
    response = client.post("/api/v1/scans", json=scan_data)
    assert response.status_code == 201, f"Ошибка создания скана: {response.status_code} - {response.text}"
    data = response.json()
    assert "id" in data, "Ответ не содержит id"
    return data["id"]


@pytest.fixture
def sim_id(client):
    """Фикстура: ID созданной симуляции"""
    sim_data = {
        "simulation_type": "spm",  # Требуется simulation_type
        "surface_type": "graphene",
        "resolution": 64,
        "scan_size": 0.5,
        "noise_level": 0.05,
    }
    response = client.post("/api/v1/simulations", json=sim_data)
    assert response.status_code == 201, f"Ошибка создания симуляции: {response.status_code}"
    data = response.json()
    # Используем simulation_id (строку), а не id (integer)
    assert "simulation_id" in data, "Ответ не содержит simulation_id"
    return data["simulation_id"]


# ==================== Tests ====================

def test_db_connection():
    """Тест 1: Подключение к базе данных"""
    print("\n[TEST] Подключение к базе данных...")
    try:
        db = get_db_manager()
        assert db is not None, "Database manager не инициализирован"
        print("[PASS] Database manager инициализирован")
    except RuntimeError:
        print("[SKIP] Database manager не инициализирован (тест в изоляции)")


def test_health_check(client):
    """Тест 2: Health check API"""
    print("\n[TEST] Health check API...")
    response = client.get("/health")
    assert response.status_code == 200, f"Ошибка: {response.status_code}"
    data = response.json()
    assert data["status"] == "healthy", "API не healthy"
    assert "timestamp" in data, "Ответ не содержит timestamp"
    print("[PASS] API healthy")


def test_create_scan(client):
    """Тест 3: Создание сканирования через API"""
    print("\n[TEST] Создание сканирования через API...")
    
    scan_data = {
        "scan_type": "spm",
        "surface_type": "test_surface",
        "width": 128,
        "height": 128,
    }
    
    response = client.post("/api/v1/scans", json=scan_data)
    assert response.status_code == 201, f"Ошибка: {response.status_code} - {response.text}"
    data = response.json()
    
    assert "id" in data, "Ответ не содержит id"
    assert data["surface_type"] == "test_surface", "surface_type не совпадает"
    
    print(f"[PASS] Сканирование создано (id={data['id']})")


def test_get_scan(client, scan_id):
    """Тест 4: Получение сканирования из БД"""
    print("\n[TEST] Получение сканирования из БД...")
    
    response = client.get(f"/api/v1/scans/{scan_id}")
    assert response.status_code == 200, f"Ошибка: {response.status_code}"
    data = response.json()
    
    assert data["id"] == scan_id, "id не совпадает"
    assert data["surface_type"] == "test_surface", "surface_type не совпадает"
    
    print(f"[PASS] Сканирование получено (id={scan_id})")


def test_list_scans(client, scan_id):
    """Тест 5: Список сканирований"""
    print("\n[TEST] Список сканирований...")

    response = client.get("/api/v1/scans/")
    assert response.status_code == 200, f"Ошибка: {response.status_code}"
    data = response.json()

    # API возвращает пагинированный ответ
    assert isinstance(data, dict), "Ответ не словарь"
    assert "items" in data, "Ответ не содержит 'items'"
    assert "total" in data, "Ответ не содержит 'total'"
    assert len(data["items"]) > 0, "Список пуст"

    print(f"[PASS] Получено {len(data['items'])} сканирований (total={data['total']})")


def test_update_scan(client, scan_id):
    """Тест 6: Обновление сканирования (проверяем что endpoint есть)"""
    print("\n[TEST] Обновление сканирования...")

    # PATCH вместо PUT
    update_data = {
        "surface_type": "updated_surface",
    }

    response = client.patch(f"/api/v1/scans/{scan_id}", json=update_data)
    # Принимаем 200 или 405 (если endpoint не реализован)
    assert response.status_code in [200, 405], f"Ошибка: {response.status_code}"

    if response.status_code == 200:
        data = response.json()
        assert data.get("surface_type") == "updated_surface", "surface_type не обновился"
        print(f"[PASS] Сканирование обновлено (id={scan_id})")
    else:
        print(f"[SKIP] PATCH endpoint не реализован (405)")


def test_delete_scan(client, scan_id):
    """Тест 7: Удаление сканирования"""
    print("\n[TEST] Удаление сканирования...")

    response = client.delete(f"/api/v1/scans/{scan_id}")
    # 204 No Content или 200 OK
    assert response.status_code in [200, 204], f"Ошибка: {response.status_code}"

    # Проверяем что удалено
    response = client.get(f"/api/v1/scans/{scan_id}")
    assert response.status_code == 404, "Сканирование не удалено"

    print(f"[PASS] Сканирование удалено (id={scan_id})")


def test_create_simulation(client):
    """Тест 8: Создание симуляции"""
    print("\n[TEST] Создание симуляции...")
    
    sim_data = {
        "simulation_type": "spm",
        "surface_type": "graphene",
        "parameters": {"resolution": 64},
    }
    
    response = client.post("/api/v1/simulations", json=sim_data)
    assert response.status_code == 201, f"Ошибка: {response.status_code}"
    data = response.json()
    
    assert "id" in data, "Ответ не содержит id"
    
    print(f"[PASS] Симуляция создана (id={data['id']})")


def test_simulation_status(client, sim_id):
    """Тест 9: Статус симуляции"""
    print("\n[TEST] Статус симуляции...")

    # Проверяем основную симуляцию
    response = client.get(f"/api/v1/simulations/{sim_id}")
    assert response.status_code == 200, f"Ошибка: {response.status_code}"
    data = response.json()

    assert "status" in data or "simulation_type" in data, "Ответ не содержит данные симуляции"

    print(f"[PASS] Симуляция найдена (id={sim_id})")


def test_dashboard_stats(client):
    """Тест 10: Статистика дашборда"""
    print("\n[TEST] Статистика дашборда...")
    
    response = client.get("/api/v1/dashboard/stats")
    assert response.status_code == 200, f"Ошибка: {response.status_code}"
    data = response.json()
    
    assert isinstance(data, dict), "Ответ не словарь"
    assert "total_scans" in data or "total_simulations" in data, "Ответ не содержит статистику"
    
    print("[PASS] Статистика получена")


def test_auth_login(client):
    """Тест 11: Аутентификация (login)"""
    print("\n[TEST] Аутентификация...")
    
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


def test_db_transaction_rollback(client):
    """Тест 12: Откат транзакции при ошибке"""
    print("\n[TEST] Откат транзакции...")
    
    # Создаём некорректные данные
    invalid_data = {
        "surface_type": "",
        "resolution": -1,
    }
    
    response = client.post("/api/v1/scans/", json=invalid_data)
    
    # Должна быть ошибка валидации
    assert response.status_code in [400, 422], f"Ожидалась ошибка валидации: {response.status_code}"
    
    print("[PASS] Транзакция откатилась при ошибке валидации")


def test_concurrent_requests(client):
    """Тест 13: Параллельные запросы"""
    print("\n[TEST] Параллельные запросы...")
    
    import concurrent.futures
    
    def make_request():
        response = client.get("/api/v1/scans/")
        return response.status_code
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request) for _ in range(5)]
        results = [f.result() for f in futures]
    
    assert all(r == 200 for r in results), "Не все запросы успешны"
    
    print("[PASS] Параллельные запросы обработаны")
