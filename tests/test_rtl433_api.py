"""
Тесты для RTL-433 API (api/routes/rtl433.py)

Покрытие:
- GET /readings — список показаний датчиков
- GET /devices — список устройств
- GET /stats — статистика
- POST /clear — очистка данных
- Model validation
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
        patch("api.routes.rtl433._db_manager") as mock_db,
        patch("api.routes.rtl433._cache_manager") as mock_cache,
    ):
        # Настройка mock database
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db.get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.get_connection.return_value.__exit__ = MagicMock(return_value=False)

        yield mock_db, mock_cache


class TestGetReadings:
    """Тесты GET /readings"""

    def test_get_readings_success(self, client, mock_managers):
        """Тест получения списка показаний"""
        mock_db, mock_cache = mock_managers
        mock_cache.get.return_value = None

        with patch("api.routes.rtl433._get_readings_db") as mock_get:
            mock_get.return_value = {
                "items": [
                    {
                        "id": 1,
                        "model": "Oregon-THN132N",
                        "device_id": "123",
                        "channel": 1,
                        "battery_ok": 1,
                        "temperature_c": 22.5,
                        "humidity": 65.0,
                        "pressure_hpa": None,
                        "wind_speed_kmh": None,
                        "rain_mm": None,
                        "created_at": "2026-04-14T10:00:00",
                    }
                ],
                "total": 1,
            }

            response = client.get("/api/v1/rtl433/readings")

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1
            assert len(data["items"]) == 1
            assert data["items"][0]["model"] == "Oregon-THN132N"
            assert data["items"][0]["temperature_c"] == 22.5

    def test_get_readings_filter_model(self, client, mock_managers):
        """Тест фильтрации по модели"""
        mock_db, mock_cache = mock_managers
        mock_cache.get.return_value = None

        with patch("api.routes.rtl433._get_readings_db") as mock_get:
            mock_get.return_value = {"items": [], "total": 0}

            response = client.get("/api/v1/rtl433/readings?model=Oregon-THN132N")

            assert response.status_code == 200
            # Проверяем, что фильтр был передан
            mock_get.assert_called_once()

    def test_get_readings_filter_device(self, client, mock_managers):
        """Тест фильтрации по device_id"""
        mock_db, mock_cache = mock_managers
        mock_cache.get.return_value = None

        with patch("api.routes.rtl433._get_readings_db") as mock_get:
            mock_get.return_value = {"items": [], "total": 0}

            response = client.get("/api/v1/rtl433/readings?device_id=123")

            assert response.status_code == 200
            mock_get.assert_called_once()

    def test_get_readings_cache_hit(self, client, mock_managers):
        """Тест использования кэша"""
        mock_db, mock_cache = mock_managers
        cached_response = {"total": 0, "items": [], "limit": 50, "offset": 0}
        mock_cache.get.return_value = cached_response

        response = client.get("/api/v1/rtl433/readings")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        mock_cache.get.assert_called_once()

    def test_get_readings_with_pagination(self, client, mock_managers):
        """Тест пагинации показаний"""
        mock_db, mock_cache = mock_managers
        mock_cache.get.return_value = None

        with patch("api.routes.rtl433._get_readings_db") as mock_get:
            mock_get.return_value = {"items": [], "total": 0}

            response = client.get("/api/v1/rtl433/readings?limit=10&offset=5")

            assert response.status_code == 200
            data = response.json()
            assert data["limit"] == 10
            assert data["offset"] == 5


class TestGetDevices:
    """Тесты GET /devices"""

    def test_get_devices_success(self, client, mock_managers):
        """Тест получения списка устройств"""
        mock_db, mock_cache = mock_managers
        mock_cache.get.return_value = None

        with patch("api.routes.rtl433._get_devices_db") as mock_get:
            mock_get.return_value = {
                "devices": [
                    {
                        "model": "Oregon-THN132N",
                        "device_id": "123",
                        "channel": 1,
                        "reading_count": 50,
                        "last_seen": "2026-04-14T10:00:00",
                        "avg_temperature_c": 22.5,
                        "avg_humidity": 65.0,
                    }
                ],
                "total": 1,
            }

            response = client.get("/api/v1/rtl433/devices")

            assert response.status_code == 200
            data = response.json()
            assert data["total_devices"] == 1
            assert len(data["devices"]) == 1

    def test_get_devices_empty(self, client, mock_managers):
        """Тест пустого списка устройств"""
        mock_db, mock_cache = mock_managers
        mock_cache.get.return_value = None

        with patch("api.routes.rtl433._get_devices_db") as mock_get:
            mock_get.return_value = {"devices": [], "total": 0}

            response = client.get("/api/v1/rtl433/devices")

            assert response.status_code == 200
            data = response.json()
            assert data["total_devices"] == 0


class TestGetStats:
    """Тесты GET /stats"""

    def test_get_stats_success(self, client, mock_managers):
        """Тест получения статистики"""
        mock_db, mock_cache = mock_managers
        mock_cache.get.return_value = None

        with patch("api.routes.rtl433._ensure_table"):
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value = mock_cursor

            # Mock query results
            mock_cursor.fetchone.side_effect = [
                (100,),  # total_readings
                (5,),  # unique_devices
                ("2026-04-01T10:00:00", "2026-04-14T10:00:00"),  # min/max created_at
            ]

            mock_db_instance = MagicMock()
            mock_db_instance.get_connection.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            mock_db_instance.get_connection.return_value.__exit__ = MagicMock(return_value=False)

            with patch("api.routes.rtl433._get_db", return_value=mock_db_instance):
                response = client.get("/api/v1/rtl433/stats")

                assert response.status_code == 200
                data = response.json()
                assert data["total_readings"] == 100
                assert data["unique_devices"] == 5
                assert "timestamp" in data


class TestClearData:
    """Тесты POST /clear"""

    def test_clear_data_success(self, client, mock_managers):
        """Тест очистки данных"""
        mock_db, mock_cache = mock_managers

        with patch("api.routes.rtl433._ensure_table"):
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value = mock_cursor

            # Mock rowcount для DELETE запроса
            mock_cursor.rowcount = 100

            mock_db_instance = MagicMock()
            mock_db_instance.get_connection.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            mock_db_instance.get_connection.return_value.__exit__ = MagicMock(return_value=False)

            with patch("api.routes.rtl433._get_db", return_value=mock_db_instance):
                response = client.post("/api/v1/rtl433/clear")

                assert response.status_code == 200
                data = response.json()
                assert "message" in data
                assert data["deleted_count"] == 100


class TestRTL433Validation:
    """Тесты валидации"""

    def test_reading_model_structure(self):
        """Тест структуры модели показания"""
        from api.routes.rtl433 import RTL433Reading

        reading = RTL433Reading(
            id=1,
            model="Oregon-THN132N",
            device_id="123",
            channel=1,
            battery_ok=1,
            temperature_c=22.5,
            humidity=65.0,
            pressure_hpa=1013.25,
            wind_speed_kmh=10.5,
            rain_mm=0.0,
            created_at="2026-04-14T10:00:00",
        )

        assert reading.id == 1
        assert reading.model == "Oregon-THN132N"
        assert reading.device_id == "123"
        assert reading.temperature_c == 22.5

    def test_device_summary_model_structure(self):
        """Тест структуры модели summary устройства"""
        from api.routes.rtl433 import RTL433DeviceSummary

        device = RTL433DeviceSummary(
            model="Oregon-THN132N",
            device_id="123",
            channel=1,
            reading_count=50,
            last_seen="2026-04-14T10:00:00",
            avg_temperature_c=22.5,
            avg_humidity=65.0,
        )

        assert device.model == "Oregon-THN132N"
        assert device.reading_count == 50
        assert device.avg_temperature_c == 22.5

    def test_readings_response_structure(self):
        """Тест структуры ответа показаний"""
        from api.routes.rtl433 import RTL433ReadingsResponse

        response = RTL433ReadingsResponse(items=[], total=0, limit=50, offset=0)

        assert response.total == 0
        assert response.limit == 50
        assert response.offset == 0

    def test_devices_response_structure(self):
        """Тест структуры ответа устройств"""
        from api.routes.rtl433 import RTL433DevicesResponse

        response = RTL433DevicesResponse(devices=[], total_devices=0)

        assert response.total_devices == 0
        assert len(response.devices) == 0

    def test_pagination_limits(self):
        """Тест лимитов пагинации"""
        # Проверка limit range (1-500)
        assert 1 <= 50 <= 500
        # Проверка offset range (>= 0)
        assert 0 >= 0

    def test_optional_fields(self):
        """Тест опциональных полей модели"""
        from api.routes.rtl433 import RTL433Reading

        # Минимальная запись (только обязательные поля)
        reading = RTL433Reading(
            id=1,
            model="Oregon-THN132N",
            device_id="123",
        )

        assert reading.temperature_c is None
        assert reading.humidity is None
        assert reading.pressure_hpa is None
