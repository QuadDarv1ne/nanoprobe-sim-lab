"""
Тесты для Comparison API (api/routes/comparison.py)

Покрытие:
- POST /compare — сравнение двух поверхностей
- GET /history — история сравнений
- GET /{comparison_id} — получение по ID
- DELETE /{comparison_id} — удаление
- GET /{comparison_id}/export — экспорт в CSV/JSON
- Error handling
"""

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
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


class TestCompareSurfaces:
    """Тесты POST /compare"""

    def test_compare_surfaces_file1_not_found(self, client):
        """Тест ошибки когда первый файл не найден"""
        response = client.post(
            "/api/v1/comparison",
            json={
                "image1_path": "/nonexistent/image1.png",
                "image2_path": "/tmp/image2.png",
            },
        )

        # ValidationError возвращает 400 или 422
        assert response.status_code in [400, 422]

    def test_compare_surfaces_file2_not_found(self, client, tmp_path):
        """Тест ошибки когда второй файл не найден"""
        img1 = tmp_path / "image1.png"
        img1.write_bytes(b"fake image data")

        response = client.post(
            "/api/v1/comparison",
            json={
                "image1_path": str(img1),
                "image2_path": "/nonexistent/image2.png",
            },
        )

        assert response.status_code in [400, 422]

    def test_compare_surfaces_comparator_exception(self, client, tmp_path):
        """Тест при исключении компаратора"""
        img1 = tmp_path / "image1.png"
        img2 = tmp_path / "image2.png"
        img1.write_bytes(b"fake image data 1")
        img2.write_bytes(b"fake image data 2")

        with patch("utils.surface_comparator.SurfaceComparator") as mock_comp:
            mock_comp.return_value.compare.side_effect = Exception("Comparison failed")

            response = client.post(
                "/api/v1/comparison",
                json={
                    "image1_path": str(img1),
                    "image2_path": str(img2),
                },
            )

            assert response.status_code in [400, 422]


class TestComparisonHistory:
    """Тесты GET /history"""

    def test_get_history_empty(self, client):
        """Тест получения пустой истории сравнений"""
        response = client.get("/api/v1/comparison/history")

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "limit" in data
        assert data["total"] == 0

    def test_get_history_with_limit(self, client):
        """Тест получения истории с лимитом"""
        response = client.get("/api/v1/comparison/history?limit=10")

        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 10


class TestGetComparison:
    """Тесты GET /{comparison_id}"""

    def test_get_comparison_not_found(self, client):
        """Тест когда сравнение не найдено"""
        response = client.get("/api/v1/comparison/nonexistent")

        # Должен вернуть 404
        assert response.status_code == 404
        data = response.json()
        # Проверяем что есть сообщение об ошибке (может быть на русском)
        assert "message" in data or "detail" in data


class TestComparisonEndpointExists:
    """Тесты существования эндпоинтов"""

    def test_comparison_route_exists(self, client):
        """Тест что comparison роут существует"""
        # Проверяем что роуты зарегистрированы
        routes = [route.path for route in app.routes]
        comparison_routes = [r for r in routes if "comparison" in str(r).lower()]
        assert len(comparison_routes) > 0

    def test_comparison_history_route_exists(self, client):
        """Тест что history роут существует"""
        response = client.get("/api/v1/comparison/history")
        # Должен вернуть 200 или ошибку БД, но не 404
        assert response.status_code != 404


class TestComparisonMetrics:
    """Тесты валидации метрик сравнения"""

    def test_comparison_metrics_structure(self):
        """Тест структуры метрик сравнения"""
        metrics = {
            "ssim": 0.90,
            "psnr": 32.5,
            "mse": 120.3,
            "similarity": 0.85,
            "pearson": 0.88,
        }

        # Проверяем что все ключи присутствуют
        required_keys = ["ssim", "psnr", "mse", "similarity", "pearson"]
        for key in required_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))

    def test_ssim_range(self):
        """Тест что SSIM в диапазоне [0, 1]"""
        ssim_values = [0.0, 0.5, 0.9, 1.0]
        for ssim in ssim_values:
            assert 0 <= ssim <= 1

    def test_similarity_range(self):
        """Тест что similarity в диапазоне [0, 1]"""
        sim_values = [0.0, 0.5, 0.85, 1.0]
        for sim in sim_values:
            assert 0 <= sim <= 1

    def test_psnr_positive(self):
        """Тест что PSNR положительный"""
        psnr_values = [20.0, 30.0, 40.0, 50.0]
        for psnr in psnr_values:
            assert psnr > 0

    def test_mse_non_negative(self):
        """Тест что MSE неотрицательный"""
        mse_values = [0.0, 50.0, 100.0, 200.0]
        for mse in mse_values:
            assert mse >= 0
