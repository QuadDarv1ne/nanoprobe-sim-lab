"""
Tests for SSTV Calibration API

Тесты для API роутов калибровки PPM.
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.routes.sstv.calibration import router


@pytest.fixture
def client():
    """Создание тестового клиента."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/sstv")

    with TestClient(app) as client:
        yield client


class TestCalibrationAPI:
    """Тесты для Calibration API."""

    def test_get_calibration_status(self, client):
        """Тест получения статуса калибровки."""
        with patch("api.routes.sstv.calibration._get_calibrator") as mock_get:
            mock_calibrator = MagicMock()
            mock_calibrator.get_calibration_info.return_value = {
                "has_calibration": True,
                "ppm": 42.5,
                "method": "rtl_test",
                "confidence": 0.8,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            mock_calibrator.is_calibration_valid.return_value = True
            mock_get.return_value = mock_calibrator

            response = client.get("/api/v1/sstv/calibration/status")

            assert response.status_code == 200
            data = response.json()
            assert data["has_calibration"] is True
            assert data["ppm"] == 42.5
            assert data["is_valid"] is True

    def test_get_calibration_status_not_found(self, client):
        """Тест получения статуса без калибровки."""
        with patch("api.routes.sstv.calibration._get_calibrator") as mock_get:
            mock_calibrator = MagicMock()
            mock_calibrator.get_calibration_info.return_value = {
                "has_calibration": False,
                "ppm": None,
            }
            mock_calibrator.is_calibration_valid.return_value = False
            mock_get.return_value = mock_calibrator

            response = client.get("/api/v1/sstv/calibration/status")

            assert response.status_code == 200
            data = response.json()
            assert data["has_calibration"] is False

    def test_automated_calibration_rtl_test(self, client):
        """Тест автоматической калибровки через rtl_test."""
        with patch("api.routes.sstv.calibration._get_calibrator") as mock_get:
            mock_calibrator = MagicMock()
            mock_calibrator.calibrate_with_rtl_test.return_value = 35.2
            mock_calibrator.get_calibration_info.return_value = {
                "has_calibration": True,
                "ppm": 35.2,
                "method": "rtl_test",
                "confidence": 0.8,
            }
            mock_get.return_value = mock_calibrator

            response = client.post(
                "/api/v1/sstv/calibration/automated",
                json={"method": "rtl_test", "device_index": 0},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["ppm_error"] == 35.2
            assert data["method"] == "rtl_test"

    def test_automated_calibration_signal(self, client):
        """Тест автоматической калибровки по сигналу."""
        with patch("api.routes.sstv.calibration._get_calibrator") as mock_get:
            mock_calibrator = MagicMock()
            mock_calibrator.calibrate_with_signal.return_value = 28.5
            mock_calibrator.get_calibration_info.return_value = {
                "has_calibration": True,
                "ppm": 28.5,
                "method": "signal",
                "confidence": 0.7,
            }
            mock_get.return_value = mock_calibrator

            response = client.post(
                "/api/v1/sstv/calibration/automated",
                json={
                    "method": "signal",
                    "frequency_mhz": 100.0,
                    "device_index": 0,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["ppm_error"] == 28.5

    def test_automated_calibration_auto(self, client):
        """Тест автоматической калибровки (auto)."""
        with patch("api.routes.sstv.calibration._get_calibrator") as mock_get:
            mock_calibrator = MagicMock()
            mock_calibrator.calibrate_auto.return_value = 40.0
            mock_calibrator.get_calibration_info.return_value = {
                "has_calibration": True,
                "ppm": 40.0,
                "method": "auto",
                "confidence": 0.8,
            }
            mock_get.return_value = mock_calibrator

            response = client.post(
                "/api/v1/sstv/calibration/automated",
                json={"method": "auto", "device_index": 0},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_automated_calibration_missing_frequency(self, client):
        """Тест калибровки signal без frequency_mhz."""
        response = client.post(
            "/api/v1/sstv/calibration/automated",
            json={"method": "signal", "device_index": 0},
        )

        assert response.status_code == 422  # Validation error

    def test_automated_calibration_failure(self, client):
        """Тест неудачной калибровки."""
        with patch("api.routes.sstv.calibration._get_calibrator") as mock_get:
            mock_calibrator = MagicMock()
            mock_calibrator.calibrate_auto.return_value = None
            mock_get.return_value = mock_calibrator

            response = client.post(
                "/api/v1/sstv/calibration/automated",
                json={"method": "auto", "device_index": 0},
            )

            assert response.status_code == 500

    def test_get_current_calibration(self, client):
        """Тест получения текущей калибровки."""
        with patch("api.routes.sstv.calibration._get_calibrator") as mock_get:
            mock_calibrator = MagicMock()
            mock_calibrator.get_calibration.return_value = 45.0
            mock_calibrator.get_calibration_info.return_value = {
                "has_calibration": True,
                "ppm": 45.0,
                "method": "rtl_test",
                "confidence": 0.8,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            mock_calibrator.is_calibration_valid.return_value = True
            mock_get.return_value = mock_calibrator

            response = client.get("/api/v1/sstv/calibration/current")

            assert response.status_code == 200
            data = response.json()
            assert data["ppm"] == 45.0
            assert data["device"] == 0

    def test_get_current_calibration_not_found(self, client):
        """Тест получения калибровки которой нет."""
        with patch("api.routes.sstv.calibration._get_calibrator") as mock_get:
            mock_calibrator = MagicMock()
            mock_calibrator.get_calibration.return_value = None
            mock_get.return_value = mock_calibrator

            response = client.get("/api/v1/sstv/calibration/current")

            assert response.status_code == 404

    def test_reset_calibration(self, client):
        """Тест сброса калибровки."""
        with patch("api.routes.sstv.calibration._get_calibrator") as mock_get:
            mock_calibrator = MagicMock()
            mock_get.return_value = mock_calibrator

            response = client.post("/api/v1/sstv/calibration/reset")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            mock_calibrator.reset_calibration.assert_called_once()

    def test_get_calibration_file(self, client, tmp_path):
        """Тест получения файла калибровки."""
        cal_file = tmp_path / "device_calibration.json"
        cal_file.write_text(json.dumps({"0": {"ppm": 30.0}}))

        with patch("api.routes.sstv.calibration._get_calibration_file_path") as mock_path:
            mock_path.return_value = cal_file

            response = client.get("/api/v1/sstv/calibration/file")

            assert response.status_code == 200
            data = response.json()
            assert "0" in data

    def test_get_calibration_file_not_found(self, client):
        """Тест получения несуществующего файла."""
        from pathlib import Path

        with patch("api.routes.sstv.calibration._get_calibration_file_path") as mock_path:
            mock_path.return_value = Path("/nonexistent/file.json")

            response = client.get("/api/v1/sstv/calibration/file")

            assert response.status_code == 404

    def test_get_devices(self, client):
        """Тест получения списка устройств."""
        with patch("api.routes.sstv.calibration.get_rtl_sdr_devices") as mock_get:
            mock_get.return_value = [
                {"index": 0, "manufacturer": "Realtek", "product": "RTL2838", "serial": "001"},
            ]

            response = client.get("/api/v1/sstv/calibration/devices")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["product"] == "RTL2838"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
