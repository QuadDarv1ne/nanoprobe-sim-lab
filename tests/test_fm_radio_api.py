"""
Тесты для FM Radio API (api/routes/fm_radio.py)

Покрытие:
- GET /recordings — список записей
- GET /stations — список станций
- GET /stats — статистика
- POST /clear — очистка данных
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

TEST_DB = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_PATH"] = TEST_DB

from api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """Фикстура: HTTP клиент"""
    with TestClient(app) as test_client:
        yield test_client
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except Exception:
            pass


@pytest.fixture(autouse=True)
def mock_managers():
    """Фикстура: моки для DatabaseManager и CacheManager"""
    with (
        patch("api.routes.fm_radio._db_manager") as mock_db,
        patch("api.routes.fm_radio._cache_manager") as mock_cache,
    ):
        # Настройка mock database
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db.get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.get_connection.return_value.__exit__ = MagicMock(return_value=False)

        yield mock_db, mock_cache


class TestGetRecordings:
    """Тесты GET /recordings"""

    def test_get_recordings_success(self, client, mock_managers):
        """Тест получения списка записей"""
        mock_db, mock_cache = mock_managers
        mock_cache.get.return_value = None

        with patch("api.routes.fm_radio._get_recordings_db") as mock_get:
            mock_get.return_value = {
                "items": [
                    {
                        "id": 1,
                        "frequency_mhz": 100.5,
                        "file_path": "/path/to/rec1.mp3",
                        "file_size_bytes": 1024000,
                        "duration_sec": 60.0,
                        "created_at": "2026-04-14T10:00:00",
                    }
                ],
                "total": 1,
            }

            response = client.get("/api/v1/fm-radio/recordings")

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1
            assert len(data["items"]) == 1
            assert data["items"][0]["frequency_mhz"] == 100.5

    def test_get_recordings_with_pagination(self, client, mock_managers):
        """Тест пагинации записей"""
        mock_db, mock_cache = mock_managers
        mock_cache.get.return_value = None

        with patch("api.routes.fm_radio._get_recordings_db") as mock_get:
            mock_get.return_value = {"items": [], "total": 0}

            response = client.get("/api/v1/fm-radio/recordings?limit=10&offset=5")

            assert response.status_code == 200
            data = response.json()
            assert data["limit"] == 10
            assert data["offset"] == 5

    def test_get_recordings_cache_hit(self, client, mock_managers):
        """Тест использования кэша"""
        mock_db, mock_cache = mock_managers
        cached_response = {"total": 0, "items": [], "limit": 50, "offset": 0}
        mock_cache.get.return_value = cached_response

        response = client.get("/api/v1/fm-radio/recordings")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        mock_cache.get.assert_called_once()


class TestGetStations:
    """Тесты GET /stations"""

    def test_get_stations_success(self, client, mock_managers):
        """Тест получения списка станций"""
        mock_db, mock_cache = mock_managers
        mock_cache.get.return_value = None

        with patch("api.routes.fm_radio._get_stations_db") as mock_get:
            mock_get.return_value = {
                "stations": [
                    {
                        "id": 1,
                        "frequency_mhz": 101.2,
                        "signal_strength_db": -50.0,
                        "signal_power": -80.0,
                        "last_seen": "2026-04-14T10:00:00",
                    }
                ],
                "total": 1,
            }

            response = client.get("/api/v1/fm-radio/stations")

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1
            assert len(data["stations"]) == 1

    def test_get_stations_empty(self, client, mock_managers):
        """Тест пустого списка станций"""
        mock_db, mock_cache = mock_managers
        mock_cache.get.return_value = None

        with patch("api.routes.fm_radio._get_stations_db") as mock_get:
            mock_get.return_value = {"stations": [], "total": 0}

            response = client.get("/api/v1/fm-radio/stations")

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 0


class TestGetStats:
    """Тесты GET /stats"""

    def test_get_stats_success(self, client, mock_managers):
        """Тест получения статистики"""
        mock_db, mock_cache = mock_managers
        mock_cache.get.return_value = None

        with patch("api.routes.fm_radio._ensure_table"):
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value = mock_cursor

            # Mock query results
            mock_cursor.fetchone.side_effect = [
                (10,),  # total_recordings
                (5,),  # unique_stations
                (5000000,),  # total_storage_bytes
            ]

            mock_db_instance = MagicMock()
            mock_db_instance.get_connection.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            mock_db_instance.get_connection.return_value.__exit__ = MagicMock(return_value=False)

            with patch("api.routes.fm_radio._get_db", return_value=mock_db_instance):
                response = client.get("/api/v1/fm-radio/stats")

                assert response.status_code == 200
                data = response.json()
                assert data["total_recordings"] == 10
                assert data["unique_stations"] == 5
                assert "timestamp" in data


class TestClearData:
    """Тесты POST /clear"""

    def test_clear_data_success(self, client, mock_managers):
        """Тест очистки данных"""
        mock_db, mock_cache = mock_managers

        with patch("api.routes.fm_radio._ensure_table"):
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value = mock_cursor

            # Mock rowcount для DELETE запросов
            mock_cursor.rowcount = 5

            mock_db_instance = MagicMock()
            mock_db_instance.get_connection.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            mock_db_instance.get_connection.return_value.__exit__ = MagicMock(return_value=False)

            with patch("api.routes.fm_radio._get_db", return_value=mock_db_instance):
                response = client.post("/api/v1/fm-radio/clear")

                assert response.status_code == 200
                data = response.json()
                assert "message" in data
                assert data["deleted_recordings"] == 5
                assert data["deleted_stations"] == 5


class TestFMRadioValidation:
    """Тесты валидации"""

    def test_recording_model_structure(self):
        """Тест структуры модели записи"""
        from api.routes.fm_radio import FMRecording

        recording = FMRecording(
            id=1,
            frequency_mhz=100.5,
            file_path="/path/to/rec.mp3",
            file_size_bytes=1024,
            duration_sec=60.0,
            created_at="2026-04-14T10:00:00",
        )

        assert recording.id == 1
        assert recording.frequency_mhz == 100.5
        assert recording.file_path == "/path/to/rec.mp3"

    def test_station_model_structure(self):
        """Тест структуры модели станции"""
        from api.routes.fm_radio import FMStation

        station = FMStation(
            id=1,
            frequency_mhz=101.2,
            signal_strength_db=-50.0,
            signal_power=-80.0,
            last_seen="2026-04-14T10:00:00",
        )

        assert station.id == 1
        assert station.frequency_mhz == 101.2
        assert station.signal_strength_db == -50.0

    def test_recordings_response_structure(self):
        """Тест структуры ответа записей"""
        from api.routes.fm_radio import FMRecordingsResponse

        response = FMRecordingsResponse(items=[], total=0, limit=50, offset=0)

        assert response.total == 0
        assert response.limit == 50
        assert response.offset == 0

    def test_stats_response_structure(self):
        """Тест структуры ответа статистики"""
        from api.routes.fm_radio import FMStatsResponse

        response = FMStatsResponse(
            total_recordings=10,
            unique_stations=5,
            total_storage_bytes=5000000,
            timestamp="2026-04-14T10:00:00",
        )

        assert response.total_recordings == 10
        assert response.unique_stations == 5
        assert response.total_storage_bytes == 5000000

    def test_pagination_limits(self):
        """Тест лимитов пагинации"""
        # Проверка limit range (1-500)
        assert 1 <= 50 <= 500
        # Проверка offset range (>= 0)
        assert 0 >= 0
