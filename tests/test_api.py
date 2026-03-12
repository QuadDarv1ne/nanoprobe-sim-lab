# -*- coding: utf-8 -*-
"""
Тесты для FastAPI API
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import os

# Создаём тестовую директорию для БД
os.makedirs("data", exist_ok=True)

# Инициализируем БД ДО импорта app
from utils.database import DatabaseManager
import api.main
api.main.db_manager = DatabaseManager("data/nanoprobe.db")

from api.main import app


client = TestClient(app)


class TestHealth:
    """Тесты health check"""

    def test_health_check(self):
        """Проверка health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "1.0.0"

    def test_api_root(self):
        """Проверка корневого endpoint API"""
        response = client.get("/api")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Nanoprobe Sim Lab API"
        assert data["version"] == "1.0.0"
        assert data["docs"] == "/docs"


class TestAuth:
    """Тесты аутентификации"""

    def test_login_success(self):
        """Успешный вход"""
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "Admin123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_login_invalid_credentials(self):
        """Неверные учетные данные"""
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "invalid", "password": "Wrong123"},
        )
        assert response.status_code == 401

    def test_login_empty_password(self):
        """Пустой пароль"""
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": ""},
        )
        assert response.status_code == 422  # Validation error


class TestScans:
    """Тесты сканирований"""

    def test_get_scans_empty(self):
        """Получение пустого списка сканирований"""
        response = client.get("/api/v1/scans")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data

    def test_create_scan(self):
        """Создание сканирования"""
        scan_data = {
            "scan_type": "spm",
            "surface_type": "graphite",
            "width": 256,
            "height": 256,
            "metadata": {"test": True},
        }
        response = client.post("/api/v1/scans", json=scan_data)
        assert response.status_code == 201
        data = response.json()
        assert data["scan_type"] == "spm"
        assert data["surface_type"] == "graphite"
        assert "id" in data
        assert "timestamp" in data

    def test_get_scan_not_found(self):
        """Получение несуществующего сканирования"""
        response = client.get("/api/v1/scans/999999")
        assert response.status_code == 404


class TestSimulations:
    """Тесты симуляций"""

    def test_get_simulations(self):
        """Получение списка симуляций"""
        response = client.get("/api/v1/simulations")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data

    def test_create_simulation(self):
        """Создание симуляции"""
        sim_data = {
            "simulation_type": "spm_scan",
            "parameters": {"resolution": 256, "scan_size": 100},
        }
        response = client.post("/api/v1/simulations", json=sim_data)
        assert response.status_code == 201
        data = response.json()
        assert data["simulation_type"] == "spm_scan"
        assert "simulation_id" in data


class TestAnalysis:
    """Тесты анализа дефектов"""

    def test_analyze_defects_file_not_found(self):
        """Анализ несуществующего файла"""
        response = client.post(
            "/api/v1/analysis/defects",
            json={"image_path": "/nonexistent/image.png"},
        )
        assert response.status_code == 400


class TestComparison:
    """Тесты сравнения поверхностей"""

    def test_compare_surfaces_file_not_found(self):
        """Сравнение несуществующих файлов"""
        response = client.post(
            "/api/v1/comparison",
            json={
                "image1_path": "/nonexistent/image1.png",
                "image2_path": "/nonexistent/image2.png",
            },
        )
        assert response.status_code == 400


class TestReports:
    """Тесты PDF отчётов"""

    def test_generate_surface_report(self):
        """Генерация отчёта о поверхности"""
        report_data = {
            "report_type": "surface_analysis",
            "title": "Тестовый отчёт",
            "author": "Test User",
            "data": {
                "surface_type": "graphite",
                "scan_size": 100,
                "mean_height": 10.5,
                "std_deviation": 2.3,
            },
            "images": [],
        }
        response = client.post("/api/v1/reports", json=report_data)
        assert response.status_code == 200
        data = response.json()
        assert "report_path" in data
        assert data["report_type"] == "surface_analysis"


class TestWebSocket:
    """Тесты WebSocket"""

    def test_websocket_connect(self):
        """Подключение к WebSocket"""
        with client.websocket_connect("/ws/realtime") as websocket:
            # Отправка ping
            websocket.send_json({"type": "ping"})
            data = websocket.receive_json()
            assert data["type"] == "pong"

    def test_websocket_subscribe(self):
        """Подписка на канал"""
        with client.websocket_connect("/ws/realtime") as websocket:
            websocket.send_json({"type": "subscribe", "channel": "scans"})
            data = websocket.receive_json()
            assert data["type"] == "subscribed"
            assert data["channel"] == "scans"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
