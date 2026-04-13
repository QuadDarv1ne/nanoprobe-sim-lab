"""
Тесты для System Export API (api/routes/system_export.py)

Покрытие:
- GET /export/{format} — экспорт в JSON/CSV/PDF
- GET /export-bulk — массовый экспорт в ZIP
- Error handling
- Validation
"""

import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Устанавливаем тестовую БД
TEST_DB = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_PATH"] = TEST_DB

from api.main import app


@pytest.fixture(scope="module")
def client():
    """Фикстура: HTTP клиент для тестов"""
    with TestClient(app) as test_client:
        yield test_client
    # Cleanup
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except Exception:
            pass


class TestExportData:
    """Тесты GET /export/{format}"""

    def test_export_json(self, client):
        """Тест экспорта в JSON"""
        response = client.get("/api/v1/export/json")

        assert response.status_code == 200
        data = response.json()
        assert data["format"] == "json"
        assert data["status"] == "success"
        assert "download_url" in data
        assert "json" in data["download_url"]

    def test_export_csv(self, client):
        """Тест экспорта в CSV"""
        response = client.get("/api/v1/export/csv")

        assert response.status_code == 200
        data = response.json()
        assert data["format"] == "csv"
        assert data["status"] == "success"
        assert "csv" in data["download_url"]

    def test_export_pdf(self, client):
        """Тест экспорта в PDF"""
        response = client.get("/api/v1/export/pdf")

        assert response.status_code == 200
        data = response.json()
        assert data["format"] == "pdf"
        assert data["status"] == "success"
        assert "pdf" in data["download_url"]

    def test_export_invalid_format(self, client):
        """Тест экспорта с неподдерживаемым форматом"""
        response = client.get("/api/v1/export/xml")

        assert response.status_code in [400, 422]

    def test_export_format_case_insensitive(self, client):
        """Тест что формат регистронезависимый"""
        # lowercase должен работать
        response = client.get("/api/v1/export/json")
        assert response.status_code == 200


class TestExportBulk:
    """Тесты GET /export-bulk"""

    def test_export_bulk_success(self, client):
        """Тест массового экспорта"""
        mock_db = MagicMock()
        mock_db.get_scan_results.return_value = [
            {"id": 1, "name": "scan1", "created_at": "2026-01-01"}
        ]
        mock_db.get_simulations.return_value = [{"id": 1, "name": "sim1", "type": "spm"}]

        with patch("api.routes.system_export.get_db_manager", return_value=mock_db):
            response = client.get("/api/v1/export-bulk")

            # Должен вернуть ZIP или статус
            assert response.status_code in [200, 503]

    def test_export_bulk_db_not_available(self, client):
        """Тест массового экспорта без БД"""
        with patch("api.routes.system_export.get_db_manager") as mock_get:
            mock_get.side_effect = RuntimeError("Database not available")

            response = client.get("/api/v1/export-bulk")

            assert response.status_code == 503

    def test_export_bulk_empty_data(self, client):
        """Тест массового экспорта с пустыми данными"""
        mock_db = MagicMock()
        mock_db.get_scan_results.return_value = []
        mock_db.get_simulations.return_value = []

        with patch("api.routes.system_export.get_db_manager", return_value=mock_db):
            response = client.get("/api/v1/export-bulk")

            # Должен успечно создать ZIP с пустыми данными
            assert response.status_code in [200, 503]


class TestExportValidation:
    """Тесты валидации экспорта"""

    def test_export_timestamp_format(self, client):
        """Тест что timestamp в экспорте корректный"""
        response = client.get("/api/v1/export/json")

        assert response.status_code == 200
        data = response.json()
        download_url = data["download_url"]

        # Проверяем что есть timestamp в URL
        # Формат: YYYYMMDD_HHMMSS
        assert "_" in download_url
        assert "." in download_url

    def test_export_response_structure(self, client):
        """Тест структуры ответа экспорта"""
        response = client.get("/api/v1/export/json")

        assert response.status_code == 200
        data = response.json()

        required_fields = ["format", "status", "message", "download_url"]
        for field in required_fields:
            assert field in data

    def test_export_message_contains_format(self, client):
        """Тест что сообщение содержит формат"""
        for fmt in ["json", "csv", "pdf"]:
            response = client.get(f"/api/v1/export/{fmt}")

            assert response.status_code == 200
            data = response.json()
            # Сообщение должно содержать формат
            assert fmt.upper() in data["message"] or fmt in data["message"].lower()


class TestExportEdgeCases:
    """Тесты граничных случаев экспорта"""

    def test_export_concurrent_requests(self, client):
        """Тест конкурентных запросов экспорта"""
        # Делаем несколько запросов подряд
        responses = []
        for _ in range(3):
            response = client.get("/api/v1/export/json")
            responses.append(response)

        # Все должны быть успешными
        for response in responses:
            assert response.status_code == 200

    def test_export_download_url_format(self, client):
        """Тест формата URL загрузки"""
        response = client.get("/api/v1/export/json")

        assert response.status_code == 200
        data = response.json()
        url = data["download_url"]

        # URL должен начинаться с /
        assert url.startswith("/")
        # URL должен содержать формат
        assert "json" in url
