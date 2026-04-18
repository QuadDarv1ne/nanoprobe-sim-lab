"""
Tests for SSTV Satellite API

Тесты для API роутов спутникового захвата.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.routes.sstv.satellites import router


@pytest.fixture
def client():
    """Создание тестового клиента."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/sstv")

    with TestClient(app) as client:
        yield client


class TestSatelliteAPI:
    """Тесты для Satellite API."""

    def test_predict_passes(self, client):
        """Тест предсказания пролётов."""
        with patch("api.routes.sstv.satellites._get_satellite_capture") as mock_get:
            mock_instance = MagicMock()
            now = datetime.now()
            mock_instance.predict_passes.return_value = [
                MagicMock(
                    satellite="NOAA-19",
                    aos=now + timedelta(hours=1),
                    los=now + timedelta(hours=2),
                    max_elevation=45.0,
                    frequency_mhz=137.1,
                    mode="APT",
                    duration_seconds=3600,
                    azimuth_aos=180.0,
                    azimuth_los=270.0,
                    time_to_aos=lambda: 3600.0,
                )
            ]
            mock_get.return_value = mock_instance

            response = client.get("/api/v1/sstv/satellites/passes?hours_ahead=24")

            assert response.status_code == 200
            data = response.json()
            assert data["total_count"] == 1
            assert data["passes"][0]["satellite"] == "NOAA-19"

    def test_predict_passes_filtered(self, client):
        """Тест предсказания с фильтром по спутнику."""
        with patch("api.routes.sstv.satellites._get_satellite_capture") as mock_get:
            mock_instance = MagicMock()
            now = datetime.now()
            mock_instance.predict_passes.return_value = [
                MagicMock(
                    satellite="NOAA-19",
                    aos=now + timedelta(hours=1),
                    los=now + timedelta(hours=2),
                    max_elevation=45.0,
                    frequency_mhz=137.1,
                    mode="APT",
                    duration_seconds=3600,
                    azimuth_aos=180.0,
                    azimuth_los=270.0,
                    time_to_aos=lambda: 3600.0,
                )
            ]
            mock_get.return_value = mock_instance

            response = client.get("/api/v1/sstv/satellites/passes?satellite=NOAA-19")

            assert response.status_code == 200
            data = response.json()
            assert data["passes"][0]["satellite"] == "NOAA-19"

    def test_get_scheduler_status(self, client):
        """Тест получения статуса планировщика."""
        with patch("api.routes.sstv.satellites._get_satellite_capture") as mock_get:
            mock_instance = MagicMock()
            mock_instance._running = False
            mock_instance.predict_passes.return_value = []
            mock_instance.get_passes_summary.return_value = {
                "total_predicted": 0,
                "upcoming": 0,
                "active_pass": None,
            }
            mock_get.return_value = mock_instance

            response = client.get("/api/v1/sstv/satellites/status")

            assert response.status_code == 200
            data = response.json()
            assert data["running"] is False
            assert data["upcoming_passes"] == 0

    def test_start_scheduler(self, client):
        """Тест запуска планировщика."""
        with patch("api.routes.sstv.satellites._get_satellite_capture") as mock_get:
            mock_instance = MagicMock()
            mock_instance.predict_passes.return_value = []
            mock_get.return_value = mock_instance

            response = client.post(
                "/api/v1/sstv/satellites/scheduler/start",
                json={"hours_ahead": 24},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            mock_instance.start_scheduler.assert_called_once()

    def test_stop_scheduler(self, client):
        """Тест остановки планировщика."""
        with patch("api.routes.sstv.satellites._get_satellite_capture") as mock_get:
            mock_instance = MagicMock()
            mock_get.return_value = mock_instance

            response = client.post("/api/v1/sstv/satellites/scheduler/stop")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            mock_instance.stop_scheduler.assert_called_once()

    def test_get_capture_config(self, client):
        """Тест получения конфигурации захвата."""
        response = client.get("/api/v1/sstv/satellites/config")

        assert response.status_code == 200
        data = response.json()
        assert data["device_index"] == 0
        assert data["sample_rate"] == 2400000
        assert data["output_dir"] == "data/satellite_captures"

    def test_update_capture_config(self, client):
        """Тест обновления конфигурации захвата."""
        response = client.post(
            "/api/v1/sstv/satellites/config",
            json={
                "device_index": 1,
                "sample_rate": 2400000,
                "output_dir": "/custom/path",
                "min_elevation": 10.0,
                "pre_record_offset": 180,
                "post_record_offset": 120,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_get_supported_satellites(self, client):
        """Тест получения списка поддерживаемых спутников."""
        response = client.get("/api/v1/sstv/satellites/supported")

        assert response.status_code == 200
        data = response.json()
        assert "satellites" in data
        assert len(data["satellites"]) > 0
        satellite_names = [s["name"] for s in data["satellites"]]
        assert "NOAA-15" in satellite_names
        assert "METEOR-M2" in satellite_names

    def test_get_captures_empty(self, client):
        """Тест получения пустого списка записей."""

        with patch("pathlib.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            mock_path.return_value.glob.return_value = []

            response = client.get("/api/v1/sstv/satellites/captures")

            assert response.status_code == 200
            data = response.json()
            assert data["captures"] == []
            assert data["total"] == 0

    def test_delete_capture_not_found(self, client):
        """Тест удаления несуществующей записи."""

        with patch("pathlib.Path") as mock_path:
            mock_instance = MagicMock()
            mock_instance.exists.return_value = False
            mock_path.return_value = mock_instance

            response = client.delete("/api/v1/sstv/satellites/captures/nonexistent.raw")

            assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
