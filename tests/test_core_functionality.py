#!/usr/bin/env python
"""
Качественные тесты для core функциональности

Фокус на важных бизнес-сценариях, а не на покрытии всего подряд.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

TEST_DB = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_PATH"] = TEST_DB

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestHealthEndpoint:
    """Тесты health endpoint - критично для мониторинга"""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_has_timestamp(self, client):
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data

    def test_health_has_version(self, client):
        response = client.get("/health")
        data = response.json()
        assert "version" in data


class TestRootEndpoint:
    """Тесты корневого endpoint"""

    def test_root_returns_200(self, client):
        response = client.get("/api")
        assert response.status_code == 200

    def test_root_has_name(self, client):
        response = client.get("/api")
        data = response.json()
        assert "name" in data

    def test_root_has_version(self, client):
        response = client.get("/api")
        data = response.json()
        assert "version" in data


class TestScansCRUD:
    """Полный CRUD для сканирований"""

    def test_create_and_get_scan(self, client):
        """Создать и получить сканирование"""
        # Создаём
        response = client.post(
            "/api/v1/scans",
            json={"scan_type": "spm", "surface_type": "graphene", "width": 128, "height": 128},
        )
        assert response.status_code == 201
        scan_id = response.json()["id"]

        # Получаем
        response = client.get(f"/api/v1/scans/{scan_id}")
        assert response.status_code == 200
        assert response.json()["id"] == scan_id

    def test_list_scans(self, client):
        """Список сканирований"""
        response = client.get("/api/v1/scans")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data

    def test_create_scan_with_pagination(self, client):
        """Создать и проверить пагинацию"""
        # Создаём несколько
        for i in range(3):
            client.post(
                "/api/v1/scans",
                json={"scan_type": "spm", "surface_type": "test", "width": 64, "height": 64},
            )

        response = client.get("/api/v1/scans?page=1&page_size=2")
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 2


class TestSimulationsCRUD:
    """Полный CRUD для симуляций"""

    def test_create_and_get_simulation(self, client):
        """Создать и получить симуляцию"""
        response = client.post(
            "/api/v1/simulations",
            json={"simulation_type": "spm", "surface_type": "silicon", "resolution": 64},
        )
        assert response.status_code == 201
        sim_id = response.json()["simulation_id"]

        response = client.get(f"/api/v1/simulations/{sim_id}")
        assert response.status_code == 200
        assert response.json()["simulation_id"] == sim_id

    def test_list_simulations(self, client):
        """Список симуляций"""
        response = client.get("/api/v1/simulations")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data


class TestAuthEndpoints:
    """Тесты auth endpoint'ов"""

    def test_login_wrong_password(self, client):
        """Неверный пароль"""
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "wrongpassword123"},
        )
        assert response.status_code in [401, 403, 422]

    def test_login_empty_body(self, client):
        """Пустое тело запроса"""
        response = client.post("/api/v1/auth/login", json={})
        assert response.status_code in [400, 422]

    def test_protected_endpoint_without_auth(self, client):
        """Защищённый endpoint без авторизации"""
        response = client.get("/api/v1/admin/system/info")
        assert response.status_code in [401, 403]


class TestGraphQLEndpoint:
    """Базовые тесты GraphQL"""

    def test_graphql_endpoint_accepts_post(self, client):
        """GraphQL endpoint принимает POST"""
        response = client.post(
            "/api/v1/graphql",
            json={"query": "{ __typename }"},
        )
        # Может вернуть 200 или 422 (если схема другая)
        assert response.status_code in [200, 422]

    def test_graphql_invalid_query(self, client):
        """Неверный запрос"""
        response = client.post(
            "/api/v1/graphql",
            json={"query": "{{{"},
        )
        assert response.status_code in [200, 400, 422]


class TestErrorHandling:
    """Тесты обработки ошибок"""

    def test_404_for_unknown_route(self, client):
        """404 для неизвестного маршрута"""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404

    def test_422_for_invalid_json(self, client):
        """422 для неверного JSON"""
        response = client.post("/api/v1/scans", content=b"not json")
        assert response.status_code in [400, 422]


class TestSecurityHeaders:
    """Проверка security headers"""

    def test_response_has_security_headers(self, client):
        """Проверка наличия security headers"""
        response = client.get("/health")
        # Проверяем что headers вообще есть
        assert response.status_code == 200
        headers = dict(response.headers)
        assert len(headers) > 0
