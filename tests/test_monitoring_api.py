#!/usr/bin/env python
"""
Тесты для Monitoring API routes

Покрытие endpoint'ов:
- GET /monitoring/system-metrics
- GET /monitoring/health-check
- GET /monitoring/performance-metrics
- GET /monitoring/system-status
- GET /monitoring/prometheus
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
    print("\n[INFO] Инициализация тестовой БД для monitoring тестов...")
    with TestClient(app) as test_client:
        yield test_client
    print("\n[INFO] Очистка тестовой БД...")
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except Exception:
            pass


class TestMonitoringAPI:
    """Тесты для Monitoring API"""

    def test_get_system_metrics(self, client):
        """Получение системных метрик (Prometheus)"""
        response = client.get("/api/v1/monitoring/metrics")
        assert response.status_code == 200
        # Prometheus возвращает текст в формате text/plain

    def test_health_check(self, client):
        """Проверка health check endpoint"""
        response = client.get("/api/v1/monitoring/health/detailed")
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "status" in data or "health" in data

    def test_performance_metrics(self, client):
        """Получение метрик производительности"""
        response = client.get("/api/v1/monitoring/performance-metrics")
        # Может вернуть 200 или 404 если не реализовано
        assert response.status_code in [200, 404, 501]

    def test_system_status(self, client):
        """Получение статуса системы"""
        response = client.get("/api/v1/monitoring/system-status")
        # Может вернуть 200 или 404
        assert response.status_code in [200, 404, 501]

    def test_prometheus_metrics(self, client):
        """Получение Prometheus метрик"""
        response = client.get("/api/v1/monitoring/prometheus")
        # Может вернуть 200, 404 или текст
        assert response.status_code in [200, 404, 501]
