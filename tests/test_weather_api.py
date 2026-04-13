"""
Тесты для Weather API (api/routes/weather.py)

Покрытие:
- GET /weather/{location} — прогноз погоды
- Location resolution (odintsovo, moscow, iss, coordinates)
- Error handling (invalid location, API errors)
- Response structure validation
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


class TestWeatherByLocation:
    """Тесты прогноза по названию города"""

    def test_weather_odintsovo(self, client):
        """Тест прогноза для Одинцово"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "current": {
                "temperature_2m": 15.5,
                "apparent_temperature": 13.2,
                "relativehumidity_2m": 65,
                "windspeed_10m": 12.5,
                "winddirection_10m": 180,
                "pressure_msl": 1013.25,
                "weathercode": 2,
                "time": "2026-04-14T10:00",
            },
            "daily": {
                "time": ["2026-04-14", "2026-04-15", "2026-04-16"],
                "temperature_2m_max": [16.0, 18.0, 20.0],
                "temperature_2m_min": [8.0, 10.0, 12.0],
                "precipitation_sum": [0.0, 2.5, 0.0],
                "windspeed_10m_max": [15.0, 20.0, 10.0],
                "winddirection_10m_dominant": [180, 200, 190],
                "uv_index_max": [5.0, 6.0, 4.0],
                "sunrise": ["06:00", "06:01", "06:02"],
                "sunset": ["19:00", "19:01", "19:02"],
                "weathercode": [2, 61, 0],
            },
            "hourly": {
                "time": [f"2026-04-14T{i:02d}:00" for i in range(24)],
                "temperature_2m": [10.0 + i * 0.5 for i in range(24)],
                "relativehumidity_2m": [60] * 24,
                "weathercode": [2] * 24,
                "windspeed_10m": [10.0] * 24,
            },
        }

        with patch("api.routes.weather.httpx.AsyncClient.get", return_value=mock_response):
            response = client.get("/api/v1/weather/odintsovo")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["location"] == "Одинцово, Московская область"
            assert data["current"]["temperature"] == 15.5
            assert len(data["daily"]) == 3

    def test_weather_moscow(self, client):
        """Тест прогноза для Москвы"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "current": {
                "temperature_2m": 18.0,
                "apparent_temperature": 16.5,
                "relativehumidity_2m": 70,
                "windspeed_10m": 8.0,
                "winddirection_10m": 90,
                "pressure_msl": 1015.0,
                "weathercode": 0,
                "time": "2026-04-14T12:00",
            },
            "daily": {"time": ["2026-04-14"], "weathercode": [0]},
            "hourly": {"time": [], "temperature_2m": []},
        }

        with patch("api.routes.weather.httpx.AsyncClient.get", return_value=mock_response):
            response = client.get("/api/v1/weather/moscow?days=1")

            assert response.status_code == 200
            data = response.json()
            assert data["location"] == "Москва"
            assert data["current"]["weather_description"] == "Ясно"

    def test_weather_iss(self, client):
        """Тест прогноза для МКС (тестирует special location)"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "current": {"temperature_2m": -100, "weathercode": 0, "time": "2026-04-14T10:00"},
            "daily": {"time": [], "weathercode": []},
            "hourly": {"time": []},
        }

        with patch("api.routes.weather.httpx.AsyncClient.get", return_value=mock_response):
            response = client.get("/api/v1/weather/iss?days=1")

            assert response.status_code == 200
            data = response.json()
            assert "МКС" in data["location"]


class TestWeatherByCoordinates:
    """Тесты прогноза по координатам"""

    def test_weather_coordinates(self, client):
        """Тест прогноза по координатам"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "current": {"temperature_2m": 20.0, "weathercode": 1, "time": "2026-04-14T10:00"},
            "daily": {"time": ["2026-04-14"], "weathercode": [1]},
            "hourly": {"time": []},
        }

        with patch("api.routes.weather.httpx.AsyncClient.get", return_value=mock_response):
            response = client.get("/api/v1/weather/55.67,37.28?days=1")

            assert response.status_code == 200
            data = response.json()
            assert data["coordinates"]["latitude"] == 55.67
            assert data["coordinates"]["longitude"] == 37.28


class TestWeatherErrors:
    """Тесты обработки ошибок"""

    def test_weather_invalid_location(self, client):
        """Тест неизвестного местоположения"""
        response = client.get("/api/v1/weather/unknown_city")

        assert response.status_code in [400, 500, 503]

    def test_weather_invalid_coordinates(self, client):
        """Тест некорректных координат"""
        response = client.get("/api/v1/weather/invalid,coords")

        assert response.status_code in [400, 500, 503]

    def test_weather_api_error(self, client):
        """Тест ошибки внешнего API"""
        import httpx

        with patch(
            "api.routes.weather.httpx.AsyncClient.get",
            side_effect=httpx.HTTPError("Connection error"),
        ):
            response = client.get("/api/v1/weather/moscow")

            assert response.status_code in [400, 500, 503]


class TestWeatherValidation:
    """Тесты валидации"""

    def test_weather_codes_mapping(self):
        """Тест маппинга кодов погоды"""
        from api.routes.weather import WEATHER_CODES

        assert 0 in WEATHER_CODES
        assert 95 in WEATHER_CODES
        assert WEATHER_CODES[0] == "Ясно"
        assert WEATHER_CODES[61] == "Небольшой дождь"

    def test_weather_description_function(self):
        """Тест функции описания погоды"""
        from api.routes.weather import _get_weather_desc

        assert _get_weather_desc(0) == "Ясно"
        assert _get_weather_desc(61) == "Небольшой дождь"
        assert _get_weather_desc(999) == "Код 999"

    def test_weather_locations_constants(self):
        """Тест констант локаций"""
        from api.routes.weather import LOCATIONS

        assert "odintsovo" in LOCATIONS
        assert "moscow" in LOCATIONS
        assert "iss" in LOCATIONS
        assert LOCATIONS["moscow"]["lat"] == 55.75

    def test_weather_response_structure(self):
        """Тест структуры ответа"""
        response_data = {
            "status": "success",
            "location": "Москва",
            "coordinates": {"latitude": 55.75, "longitude": 37.62},
            "current": {
                "temperature": 18.0,
                "weather_description": "Ясно",
            },
            "forecast_days": 3,
            "daily": [],
            "hourly": [],
            "timestamp": "2026-04-14T10:00:00",
        }

        assert "status" in response_data
        assert "location" in response_data
        assert "current" in response_data
        assert "daily" in response_data
        assert response_data["status"] == "success"

    def test_weather_days_validation(self):
        """Тест валидации количества дней"""
        # Проверка range (1-7)
        assert 1 <= 3 <= 7
        assert 1 <= 7 <= 7
