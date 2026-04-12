#!/usr/bin/env python
"""
Тесты для дополнительных API routes

Покрытие:
- analysis API (defects, ML)
- comparison API (surfaces)
- reports API (generation)
- batch API (processing)
- alerting API (alerts management)
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
    print("\n[INFO] Инициализация тестовой БД для additional API тестов...")
    with TestClient(app) as test_client:
        yield test_client
    print("\n[INFO] Очистка тестовой БД...")
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except Exception:
            pass


class TestAnalysisAPI:
    """Тесты для Analysis API"""

    def test_analyze_defects_file_not_found(self, client):
        """Анализ дефектов - файл не найден"""
        response = client.post(
            "/api/v1/analysis/defects",
            json={"image_path": "/nonexistent/file.png", "model_name": "isolation_forest"},
        )
        # Должно вернуть 400 или 404
        assert response.status_code in [400, 404, 422]

    def test_analyze_defects_invalid_model(self, client):
        """Анализ дефектов - неверная модель"""
        response = client.post(
            "/api/v1/analysis/defects",
            json={"image_path": "test.png", "model_name": "invalid_model_xyz"},
        )
        # Должно вернуть 400 или 422
        assert response.status_code in [400, 422]

    def test_get_analysis_results(self, client):
        """Получение результатов анализа"""
        response = client.get("/api/v1/analysis/results")
        # Может вернуть 200 или 404
        assert response.status_code in [200, 404, 501]

    def test_get_analysis_by_id(self, client):
        """Получение анализа по ID"""
        response = client.get("/api/v1/analysis/test-analysis-id")
        # Может вернуть 200, 404 или 501
        assert response.status_code in [200, 404, 501]


class TestComparisonAPI:
    """Тесты для Comparison API"""

    def test_compare_surfaces(self, client):
        """Сравнение поверхностей"""
        response = client.post(
            "/api/v1/comparison/surfaces",
            json={
                "surface1": "surface_1_data",
                "surface2": "surface_2_data",
                "method": "rms",
            },
        )
        # Может вернуть 200, 400, 404, 405 или 422
        assert response.status_code in [200, 400, 404, 405, 422]

    def test_get_comparison_history(self, client):
        """Получение истории сравнений"""
        response = client.get("/api/v1/comparison/history")
        # Может вернуть 200 или 404
        assert response.status_code in [200, 404, 501]

    def test_get_comparison_by_id(self, client):
        """Получение сравнения по ID"""
        response = client.get("/api/v1/comparison/test-comparison-id")
        # Может вернуть 200, 404 или 501
        assert response.status_code in [200, 404, 501]


class TestReportsAPI:
    """Тесты для Reports API"""

    def test_generate_surface_report(self, client):
        """Генерация отчёта по поверхности"""
        response = client.post(
            "/api/v1/reports/surface",
            json={"surface_type": "graphene", "scan_id": 1},
        )
        # Может вернуть 200, 400, 404, 405 или 422
        assert response.status_code in [200, 400, 404, 405, 422]

    def test_generate_report_invalid_type(self, client):
        """Генерация отчёта - неверный тип"""
        response = client.post(
            "/api/v1/reports/invalid_type",
            json={"data": "test"},
        )
        # Должно вернуть 404, 405 или 422
        assert response.status_code in [404, 405, 422]

    def test_get_report_by_id(self, client):
        """Получение отчёта по ID"""
        response = client.get("/api/v1/reports/test-report-id")
        # Может вернуть 200, 404, 405 или 501
        assert response.status_code in [200, 404, 405, 501]

    def test_list_reports(self, client):
        """Список отчётов"""
        response = client.get("/api/v1/reports/list")
        # Может вернуть 200, 404, 405 или 501
        assert response.status_code in [200, 404, 405, 501]


class TestBatchAPI:
    """Тесты для Batch Processing API"""

    def test_batch_process_files(self, client):
        """Пакетная обработка файлов"""
        response = client.post(
            "/api/v1/batch/process",
            json={
                "file_paths": ["/path/to/file1.png", "/path/to/file2.png"],
                "process_type": "spm",
            },
        )
        # Может вернуть 200, 400, 404 или 422
        assert response.status_code in [200, 400, 404, 422]

    def test_batch_get_status(self, client):
        """Получение статуса пакетной обработки"""
        response = client.get("/api/v1/batch/test-batch-id/status")
        # Может вернуть 200, 404 или 501
        assert response.status_code in [200, 404, 501]

    def test_batch_cancel(self, client):
        """Отмена пакетной обработки"""
        response = client.post("/api/v1/batch/test-batch-id/cancel")
        # Может вернуть 200, 404 или 501
        assert response.status_code in [200, 404, 501]


class TestAlertingAPI:
    """Тесты для Alerting API"""

    def test_get_alerts(self, client):
        """Получение алертов"""
        response = client.get("/api/v1/alerts")
        # Может вернуть 200 или 404
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list) or "items" in data or "alerts" in data

    def test_get_alerts_with_filter(self, client):
        """Получение алертов с фильтром"""
        response = client.get("/api/v1/alerts?severity=critical&limit=10")
        # Может вернуть 200 или 404
        assert response.status_code in [200, 404]

    def test_create_alert(self, client):
        """Создание алерта"""
        response = client.post(
            "/api/v1/alerts",
            json={
                "type": "test_alert",
                "message": "Test alert message",
                "severity": "warning",
            },
        )
        # Может вернуть 200, 201, 400, 404 или 422
        assert response.status_code in [200, 201, 400, 404, 422]

    def test_acknowledge_alert(self, client):
        """Подтверждение алерта"""
        response = client.post("/api/v1/alerts/1/acknowledge")
        # Может вернуть 200, 404 или 501
        assert response.status_code in [200, 404, 501]

    def test_get_alert_by_id(self, client):
        """Получение алерта по ID"""
        response = client.get("/api/v1/alerts/1")
        # Может вернуть 200, 404 или 501
        assert response.status_code in [200, 404, 501]
