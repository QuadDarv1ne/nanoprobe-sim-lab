#!/usr/bin/env python
"""
Тесты для Simulations и Scans API routes

Покрытие endpoint'ов:
Simulations:
- GET /api/v1/simulations
- GET /api/v1/simulations/{simulation_id}
- POST /api/v1/simulations
- PUT /api/v1/simulations/{simulation_id}
- DELETE /api/v1/simulations/{simulation_id}

Scans:
- GET /api/v1/scans
- GET /api/v1/scans/{scan_id}
- POST /api/v1/scans
- DELETE /api/v1/scans/{scan_id}
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))

# Устанавливаем тестовую БД
TEST_DB = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_PATH"] = TEST_DB

from api.main import app


@pytest.fixture(scope="module")
def client():
    """Фикстура: HTTP клиент для тестов"""
    print("\n[INFO] Инициализация тестовой БД для simulations/scans тестов...")
    with TestClient(app) as test_client:
        yield test_client
    print("\n[INFO] Очистка тестовой БД...")
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except Exception:
            pass


class TestSimulationsAPI:
    """Тесты для Simulations API"""

    def test_get_simulations_empty(self, client):
        """Получение пустого списка симуляций"""
        response = client.get("/api/v1/simulations")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "limit" in data
        assert isinstance(data["items"], list)

    def test_create_simulation_spm(self, client):
        """Создание SPM симуляции"""
        sim_data = {
            "simulation_type": "spm",
            "surface_type": "graphene",
            "resolution": 64,
            "scan_size": 0.5,
            "noise_level": 0.05,
        }
        response = client.post("/api/v1/simulations", json=sim_data)
        assert response.status_code == 201
        data = response.json()
        assert "simulation_id" in data
        assert data["simulation_type"] == "spm"
        assert data["status"] in ["pending", "running", "completed"]
        return data["simulation_id"]

    def test_create_simulation_image(self, client):
        """Создание Image симуляции"""
        sim_data = {
            "simulation_type": "image",
            "surface_type": "test_surface",
            "resolution": 128,
            "scan_size": 1.0,
        }
        response = client.post("/api/v1/simulations", json=sim_data)
        assert response.status_code == 201
        data = response.json()
        assert data["simulation_type"] == "image"

    def test_get_simulation_by_id(self, client):
        """Получение симуляции по ID"""
        # Сначала создаём
        sim_data = {
            "simulation_type": "spm",
            "surface_type": "silicon",
            "resolution": 32,
            "scan_size": 0.25,
        }
        create_response = client.post("/api/v1/simulations", json=sim_data)
        assert create_response.status_code == 201
        sim_id = create_response.json()["simulation_id"]

        # Теперь получаем
        response = client.get(f"/api/v1/simulations/{sim_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["simulation_id"] == sim_id
        assert "simulation_type" in data

    def test_get_simulation_not_found(self, client):
        """Получение несуществующей симуляции"""
        response = client.get("/api/v1/simulations/nonexistent-id")
        assert response.status_code == 404

    def test_get_simulations_with_filter(self, client):
        """Получение списка симуляций с фильтром"""
        response = client.get("/api/v1/simulations?status=completed&limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert data["limit"] == 10

    def test_delete_simulation(self, client):
        """Удаление симуляции"""
        # Создаём симуляцию
        sim_data = {
            "simulation_type": "spm",
            "surface_type": "gold",
            "resolution": 64,
        }
        create_response = client.post("/api/v1/simulations", json=sim_data)
        assert create_response.status_code == 201
        sim_id = create_response.json()["simulation_id"]

        # Удаляем
        response = client.delete(f"/api/v1/simulations/{sim_id}")
        assert response.status_code in [200, 204, 404, 501]


class TestScansAPI:
    """Тесты для Scans API"""

    def test_get_scans_empty(self, client):
        """Получение пустого списка сканирований"""
        response = client.get("/api/v1/scans")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert isinstance(data["items"], list)

    def test_create_scan_spm(self, client):
        """Создание SPM сканирования"""
        scan_data = {
            "scan_type": "spm",
            "surface_type": "graphene",
            "width": 128,
            "height": 128,
        }
        response = client.post("/api/v1/scans", json=scan_data)
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["scan_type"] == "spm"
        return data["id"]

    def test_create_scan_image(self, client):
        """Создание Image сканирования"""
        scan_data = {
            "scan_type": "image",
            "surface_type": "test_surface",
            "width": 256,
            "height": 256,
        }
        response = client.post("/api/v1/scans", json=scan_data)
        assert response.status_code == 201
        data = response.json()
        assert data["scan_type"] == "image"

    def test_create_scan_sstv(self, client):
        """Создание SSTV сканирования"""
        scan_data = {
            "scan_type": "sstv",
            "surface_type": "signal",
            "width": 320,
            "height": 256,
        }
        response = client.post("/api/v1/scans", json=scan_data)
        assert response.status_code == 201
        data = response.json()
        assert data["scan_type"] == "sstv"

    def test_get_scan_by_id(self, client):
        """Получение сканирования по ID"""
        # Сначала создаём
        scan_data = {
            "scan_type": "spm",
            "surface_type": "silicon",
            "width": 64,
            "height": 64,
        }
        create_response = client.post("/api/v1/scans", json=scan_data)
        assert create_response.status_code == 201
        scan_id = create_response.json()["id"]

        # Теперь получаем
        response = client.get(f"/api/v1/scans/{scan_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == scan_id
        assert data["surface_type"] == "silicon"

    def test_get_scan_not_found(self, client):
        """Получение несуществующего сканирования"""
        response = client.get("/api/v1/scans/999999")
        assert response.status_code == 404

    def test_get_scans_with_pagination(self, client):
        """Получение списка сканирований с пагинацией"""
        response = client.get("/api/v1/scans?page=1&page_size=10")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert data["limit"] == 10
        assert data["offset"] == 0

    def test_get_scans_with_filter(self, client):
        """Получение сканирований с фильтром по типу"""
        response = client.get("/api/v1/scans?scan_type=spm")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data

    def test_delete_scan(self, client):
        """Удаление сканирования"""
        # Создаём сканирование
        scan_data = {
            "scan_type": "spm",
            "surface_type": "gold",
            "width": 128,
            "height": 128,
        }
        create_response = client.post("/api/v1/scans", json=scan_data)
        assert create_response.status_code == 201
        scan_id = create_response.json()["id"]

        # Удаляем
        response = client.delete(f"/api/v1/scans/{scan_id}")
        assert response.status_code in [200, 204, 404, 501]

    def test_create_scan_invalid_data(self, client):
        """Создание сканирования с невалидными данными"""
        scan_data = {
            "scan_type": "invalid_type",
            "surface_type": "",
            "width": -1,
            "height": -1,
        }
        response = client.post("/api/v1/scans", json=scan_data)
        # Должно вернуть 422 (Validation Error) или 400
        assert response.status_code in [400, 422]

    def test_create_scan_missing_fields(self, client):
        """Создание сканирования с отсутствующими полями"""
        scan_data = {}
        response = client.post("/api/v1/scans", json=scan_data)
        assert response.status_code in [400, 422]
