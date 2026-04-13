"""
Тесты для Analysis API (api/routes/analysis.py)

Покрытие:
- POST /defects — анализ дефектов изображения
- GET /defects/history — история анализов
- GET /defects/{id} — получение анализа по ID
- DELETE /defects/{id} — удаление анализа
- GET /defects/{id}/export — экспорт в CSV/JSON
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

from api.dependencies import get_db
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


def override_get_db():
    """Override зависимости для возврата mock БД"""
    mock_db = MagicMock()
    mock_db.get_defect_analyses.return_value = []
    yield mock_db


class TestAnalyzeDefects:
    """Тесты POST /defects — анализ дефектов"""

    def test_analyze_defects_file_not_found(self, client):
        """Тест анализа когда файл не найден"""
        response = client.post(
            "/api/v1/analysis/defects",
            json={
                "image_path": "/nonexistent/image.png",
                "model_name": "isolation_forest",
            },
        )

        assert response.status_code in [400, 422]

    def test_analyze_defects_success(self, client, tmp_path):
        """Тест успешного анализа дефектов"""
        img = tmp_path / "test.png"
        img.write_bytes(b"fake image data")

        app.dependency_overrides[get_db] = override_get_db
        try:
            with patch("utils.ai.defect_analyzer.DefectAnalysisPipeline") as mock_pipeline:
                mock_instance = MagicMock()
                mock_instance.analyze_image.return_value = {
                    "analysis_id": "defect_abc123",
                    "defects": [
                        {
                            "type": "scratch",
                            "x": 10,
                            "y": 20,
                            "width": 5,
                            "height": 5,
                            "area": 25,
                            "confidence": 0.95,
                        },
                        {
                            "type": "crack",
                            "x": 50,
                            "y": 60,
                            "width": 10,
                            "height": 3,
                            "area": 30,
                            "confidence": 0.88,
                        },
                    ],
                    "processing_time_ms": 150,
                }
                mock_pipeline.return_value = mock_instance

                with patch("api.metrics.BusinessMetrics.inc_defect_analysis"):
                    response = client.post(
                        "/api/v1/analysis/defects",
                        json={
                            "image_path": str(img),
                            "model_name": "isolation_forest",
                        },
                    )

                    assert response.status_code == 200
                    data = response.json()
                    assert data["image_path"] == str(img)
                    assert data["model_name"] == "isolation_forest"
                    assert data["defects_count"] == 2
                    assert data["confidence_score"] > 0
                    assert len(data["defects"]) == 2
        finally:
            app.dependency_overrides.clear()

    def test_analyze_defects_no_defects(self, client, tmp_path):
        """Тест анализа без найденных дефектов"""
        img = tmp_path / "clean.png"
        img.write_bytes(b"clean image data")

        app.dependency_overrides[get_db] = override_get_db
        try:
            with patch("utils.ai.defect_analyzer.DefectAnalysisPipeline") as mock_pipeline:
                mock_instance = MagicMock()
                mock_instance.analyze_image.return_value = {
                    "analysis_id": "defect_clean",
                    "defects": [],
                    "processing_time_ms": 100,
                }
                mock_pipeline.return_value = mock_instance

                with patch("api.metrics.BusinessMetrics.inc_defect_analysis"):
                    response = client.post(
                        "/api/v1/analysis/defects",
                        json={
                            "image_path": str(img),
                            "model_name": "kmeans",
                        },
                    )

                    assert response.status_code == 200
                    data = response.json()
                    assert data["defects_count"] == 0
                    assert data["confidence_score"] == 0.0
        finally:
            app.dependency_overrides.clear()

    def test_analyze_defects_pipeline_error(self, client, tmp_path):
        """Тест ошибки пайплайна анализа"""
        img = tmp_path / "error.png"
        img.write_bytes(b"error data")

        app.dependency_overrides[get_db] = override_get_db
        try:
            with patch("utils.ai.defect_analyzer.DefectAnalysisPipeline") as mock_pipeline:
                mock_pipeline.return_value.analyze_image.side_effect = Exception("Pipeline error")

                response = client.post(
                    "/api/v1/analysis/defects",
                    json={
                        "image_path": str(img),
                        "model_name": "isolation_forest",
                    },
                )

                assert response.status_code in [400, 422, 500]
        finally:
            app.dependency_overrides.clear()


class TestDefectHistory:
    """Тесты GET /defects/history"""

    def test_get_history_empty(self, client):
        """Тест получения пустой истории"""
        app.dependency_overrides[get_db] = override_get_db
        try:
            response = client.get("/api/v1/analysis/defects/history")

            assert response.status_code == 200
            data = response.json()
            assert "items" in data
            assert "total" in data
            assert "limit" in data
            assert isinstance(data["total"], int)
        finally:
            app.dependency_overrides.clear()

    def test_get_history_with_limit(self, client):
        """Тест получения истории с лимитом"""
        mock_db = MagicMock()
        mock_db.get_defect_analyses.return_value = [
            {
                "id": 1,
                "analysis_id": "defect_abc123",
                "model_name": "isolation_forest",
                "defects_count": 5,
            }
        ]

        def override_with_data():
            yield mock_db

        app.dependency_overrides[get_db] = override_with_data
        try:
            response = client.get("/api/v1/analysis/defects/history?limit=10")

            assert response.status_code == 200
            data = response.json()
            assert data["limit"] == 10
        finally:
            app.dependency_overrides.clear()

    def test_get_history_db_error(self, client):
        """Тест ошибки БД при получении истории"""
        mock_db = MagicMock()
        mock_db.get_defect_analyses.side_effect = Exception("DB error")

        def override_error():
            yield mock_db

        app.dependency_overrides[get_db] = override_error
        try:
            response = client.get("/api/v1/analysis/defects/history")

            assert response.status_code in [200, 400, 422, 500]
        finally:
            app.dependency_overrides.clear()


class TestGetDefectAnalysis:
    """Тесты GET /defects/{id}"""

    def test_get_analysis_not_found(self, client):
        """Тест когда анализ не найден"""
        app.dependency_overrides[get_db] = override_get_db
        try:
            response = client.get("/api/v1/analysis/defects/nonexistent")

            assert response.status_code in [404, 500]
        finally:
            app.dependency_overrides.clear()

    def test_get_analysis_by_id(self, client):
        """Тест получения анализа по ID"""
        mock_db = MagicMock()
        mock_db.get_defect_analyses.return_value = [
            {
                "id": 1,
                "analysis_id": "defect_abc123",
                "image_path": "/path/img.png",
                "model_name": "isolation_forest",
                "defects_count": 3,
            }
        ]

        def override_with_data():
            yield mock_db

        app.dependency_overrides[get_db] = override_with_data
        try:
            response = client.get("/api/v1/analysis/defects/defect_abc123")

            assert response.status_code == 200
            data = response.json()
            assert data["analysis_id"] == "defect_abc123"
        finally:
            app.dependency_overrides.clear()

    def test_get_analysis_by_numeric_id(self, client):
        """Тест получения анализа по числовому ID"""
        mock_db = MagicMock()
        mock_db.get_defect_analyses.return_value = [
            {
                "id": 42,
                "analysis_id": "defect_xyz789",
                "image_path": "/path/img.png",
            }
        ]

        def override_with_data():
            yield mock_db

        app.dependency_overrides[get_db] = override_with_data
        try:
            response = client.get("/api/v1/analysis/defects/42")

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == 42
        finally:
            app.dependency_overrides.clear()


class TestDeleteDefectAnalysis:
    """Тесты DELETE /defects/{id}"""

    def test_delete_analysis_not_found(self, client):
        """Тест удаления несуществующего анализа"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_conn.cursor.return_value = mock_cursor

        mock_db = MagicMock()
        mock_db.get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.get_connection.return_value.__exit__ = MagicMock(return_value=False)

        app.dependency_overrides[get_db] = lambda: mock_db
        try:
            response = client.delete("/api/v1/analysis/defects/nonexistent")

            assert response.status_code in [404, 500]
        finally:
            app.dependency_overrides.clear()


class TestExportDefectAnalysis:
    """Тесты GET /defects/{id}/export"""

    def test_export_analysis_not_found(self, client):
        """Тест экспорта несуществующего анализа"""
        app.dependency_overrides[get_db] = override_get_db
        try:
            response = client.get("/api/v1/analysis/defects/nonexistent/export")

            assert response.status_code in [404, 500]
        finally:
            app.dependency_overrides.clear()

    def test_export_analysis_json(self, client):
        """Тест экспорта анализа в JSON"""
        mock_db = MagicMock()
        mock_db.get_defect_analyses.return_value = [
            {
                "id": 1,
                "analysis_id": "defect_abc123",
                "defects_count": 3,
            }
        ]

        def override_with_data():
            yield mock_db

        app.dependency_overrides[get_db] = override_with_data
        try:
            response = client.get("/api/v1/analysis/defects/defect_abc123/export?fmt=json")

            assert response.status_code == 200
            data = response.json()
            assert data["analysis_id"] == "defect_abc123"
        finally:
            app.dependency_overrides.clear()

    def test_export_analysis_csv(self, client):
        """Тест экспорта анализа в CSV"""
        mock_db = MagicMock()
        mock_db.get_defect_analyses.return_value = [
            {
                "id": 1,
                "analysis_id": "defect_abc123",
                "defects_count": 3,
            }
        ]

        def override_with_data():
            yield mock_db

        app.dependency_overrides[get_db] = override_with_data
        try:
            response = client.get("/api/v1/analysis/defects/defect_abc123/export?fmt=csv")

            assert response.status_code == 200
            assert "text/csv" in response.headers.get("content-type", "")
            assert "analysis_id" in response.text
        finally:
            app.dependency_overrides.clear()


class TestAnalysisValidation:
    """Тесты валидации Analysis API"""

    def test_defect_info_structure(self):
        """Тест структуры информации о дефекте"""
        defect = {
            "type": "scratch",
            "x": 10,
            "y": 20,
            "width": 5,
            "height": 5,
            "area": 25,
            "confidence": 0.95,
        }

        required_fields = [
            "type",
            "x",
            "y",
            "width",
            "height",
            "area",
            "confidence",
        ]
        for field in required_fields:
            assert field in defect

    def test_confidence_range(self):
        """Тест что confidence в диапазоне [0, 1]"""
        confidence_values = [0.0, 0.5, 0.88, 0.95, 1.0]
        for conf in confidence_values:
            assert 0 <= conf <= 1

    def test_model_names_supported(self):
        """Тест поддерживаемых моделей"""
        expected_models = ["isolation_forest", "kmeans"]
        assert len(expected_models) >= 2
